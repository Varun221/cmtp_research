"""Gold-trace prefill analysis for CoT and C-MTP models.

Force-feed an increasing fraction ``x`` of a gold reasoning trace to the model
as its thinking prefix, let it finish (rest of the trace + answer), and measure
final accuracy as a function of ``x``.

  input  : question + gold CoT up to fraction x
  output : rest of CoT (model generated) + answer

Interpretation of x:
  - MTP  : fraction of the total spans available in the gold trace
           (spans = span_length-token chunks, padded with <lmot>, exactly as in
            training via utils.equal_segmentation).
  - CoT  : fraction of the total tokens in the gold trace.

Model loading and dataset handling mirror test_mtp.py / test_cot.py.

Set the configuration globals below, then run:  python misc/gold_trace_analysis.py
"""

import os
import re
import sys
import json
from typing import Optional

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from peft import LoraConfig, TaskType
from datasets import load_dataset
from safetensors.torch import load_file
from tqdm.auto import tqdm

# Import the cmtp package (src.model, scripts.math_grader).
CMTP_DIR = "/scratch/vy2142/cmtp_research/cmtp"
sys.path.insert(0, CMTP_DIR)
sys.path.insert(0, os.path.join(CMTP_DIR, "scripts"))
from src.model import TrainingModel, TrainingArguments, ModelArguments  # noqa: E402
from scripts.math_grader import grade_answer  # noqa: E402

# CONFIG

# Exp -- Running on CoT generations
CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp2mtp"
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp3mtp"
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp4mtp"

# NL
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cot/seminatural_llama_lora"
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp2mtp"
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp3mtp"
# CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp4mtp"

MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

MODE = "mtp"            # "mtp" or "cot"
COT_FORMAT = "structured"  # "structured" or "nl" -- how the gold cot is segmented

# Dataset with a gold CoT. Either a name handled by resolve_dataset() (e.g.
# "gsm8k-test") or a JSON file path (list of dicts with question/cot/answer).
# Dataset params (keys, cot/answer split) are set automatically by resolve_dataset.
# NL
# DATASET = "gsm8k-test"
# Exp
# DATASET = "/scratch/vy2142/cmtp_research/assets/generations/cot_lora_exp_correct_generations_gsm8k-test.json"
DATASET = "/scratch/vy2142/cmtp_research/assets/generations/cot_lora_exp_correct_generations_svamp.json"

SPAN_LENGTH = 2       # must match the checkpoint (MTP only)
BATCH_SIZE = 32
MAX_SAMPLES = None     # set to an int to subsample for a quick run

GREEDY = True          # greedy decoding
SAMPLE_MTP = False     # sample the aggregated thinking tokens (MTP only)
LORA_R = 128
LORA_ALPHA = 32
HF_TOKEN = None

# Generation budgets.
MAX_FREE_THINK = 500       # cap on model's own thinking after the forced prefix (MTP)
MAX_ANSWER_TOKENS = 300    # answer-generation budget (MTP)
MAX_NEW_TOKENS_COT = 512   # rest-of-cot + answer budget (CoT)

# X_VALUES = [round(0.05 * i, 2) for i in range(21)]  # 0.00 .. 1.00 step 0.05
X_VALUES = [round(0.1 * i, 2) for i in range(11)]  # 0.00 .. 1.00 step 0.10


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_boxed(text: str) -> Optional[str]:
    matches = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not matches:
        return None
    start = matches[-1] + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def extract_answer_str(sentence: str) -> str:
    boxed = _extract_boxed(sentence)
    if boxed is not None:
        return boxed
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    return pred[-1] if pred else ""


def compute_accuracy(gold: list, pred: list) -> float:
    acc = 0.0
    for p, g in zip(pred, gold):
        if grade_answer(str(p), str(g)):
            acc += 1
    return acc / len(gold)


def normalize_answer(val):
    if isinstance(val, bool):
        return val
    if val in ["True", "False"]:
        return val == "True"
    if val in "ABCDE":
        return val
    ans = val.split("####")[-1] if "####" in val else val
    return ans.replace(",", "").strip()


def format_cot(cot_text: str) -> str:
    """Reconstruct the cot string exactly as the training pipeline does."""
    if COT_FORMAT == "nl":
        return ". ".join(f"{cot_text}".split(". "))
    return "\n".join(f"{cot_text}".split(" "))


def load_model_and_tokenizer():
    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME_OR_PATH,
        ckpt_dir=CKPT_DIR,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_init=True,
        token=HF_TOKEN,
        train=False,
    )
    training_args = TrainingArguments(
        output_dir=CKPT_DIR,
        bf16=True,
        span_length=SPAN_LENGTH,
        greedy=GREEDY,
        sample_mtp=SAMPLE_MTP,
        compile_forward=False,
        max_token_num=2048,
    )

    name = MODEL_NAME_OR_PATH.lower()
    if any(n in name for n in ["llama", "mistral", "falcon", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "phi" in name:
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif "gpt2" in name:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME_OR_PATH}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    model = TrainingModel(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(CKPT_DIR, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(CKPT_DIR, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    if model.student.config.tie_word_embeddings:
        model.student.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        CKPT_DIR,
        token=HF_TOKEN,
        model_max_length=training_args.max_token_num + 10,
        padding_side="left",
        use_fast=False,
    )
    assert tokenizer.pad_token_id is not None
    assert model.bot_id == tokenizer.convert_tokens_to_ids("<lbot>")
    assert model.eot_id == tokenizer.convert_tokens_to_ids("<leot>")
    assert model.mot_id == tokenizer.convert_tokens_to_ids("<lmot>")

    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer, training_args


def resolve_dataset(name):
    """Load a dataset as {question, cot, answer} records, setting all params here.

    Only datasets that carry a gold CoT are supported. 'gsm8k-test' keeps its
    reasoning in the answer field before '####'; JSON files are expected to have
    explicit question/cot/answer keys. Datasets without a gold trace (gsm-hard,
    multi-arith, svamp) are rejected.
    """
    if name == "gsm8k-test":
        raw = load_dataset("gsm8k", "main")["test"]
        records = [
            {
                "question": ex["question"],
                "cot": ex["answer"].split("####")[0].strip(),
                "answer": ex["answer"].split("####")[-1],
            }
            for ex in raw
        ]
    elif ".json" in name:
        records = [
            {"question": ex["question"], "cot": ex["cot"], "answer": ex["answer"]}
            for ex in read_json(name)
        ]
    else:
        raise ValueError(
            f"Dataset '{name}' has no gold CoT. Use 'gsm8k-test' or a JSON file "
            "with question/cot/answer keys."
        )

    records = [
        {
            "question": f"{r['question']}".strip().replace("  ", " "),
            "cot": r["cot"],
            "answer": normalize_answer(r["answer"]),
        }
        for r in records
    ]
    if MAX_SAMPLES is not None:
        records = records[:MAX_SAMPLES]
    return records


def left_pad(seqs, pad_id):
    m = max(len(s) for s in seqs)
    rows = []
    for s in seqs:
        t = torch.tensor(s, dtype=torch.long)
        if len(s) < m:
            t = torch.cat([torch.full((m - len(s),), pad_id, dtype=torch.long), t])
        rows.append(t)
    return torch.stack(rows).to("cuda")


def main():
    model, tokenizer, training_args = load_model_and_tokenizer()
    records = resolve_dataset(DATASET)
    N = len(records)
    print(f"Mode={MODE} | format={COT_FORMAT} | span_length={SPAN_LENGTH} | samples={N}")

    bos_id = tokenizer.bos_token_id
    bot_id = model.bot_id
    eot_id = model.eot_id
    mot_id = model.mot_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    q_tok = [tokenizer.encode(r["question"], add_special_tokens=False) for r in records]
    answers = [r["answer"] for r in records]

    gen_kwargs = {"temperature": 0.2, "top_k": 40, "top_p": 0.95}

    def get_embedding(ids):
        return model.get_embd(model.student, model.model_name)(ids)

    def normal_agg(top_ids):
        return get_embedding(top_ids).mean(dim=0, keepdim=True)

    def _sample_from_logits(logits):
        shape = logits.shape[:-1]
        logits = logits.reshape(-1, logits.size(-1))
        if GREEDY:
            return torch.argmax(logits, dim=-1).reshape(shape)
        logits = logits / gen_kwargs["temperature"]
        if gen_kwargs["top_k"] > 1:
            top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
            logits[logits < top_k_values[:, -1].unsqueeze(-1)] = -float("inf")
        if gen_kwargs["top_p"] < 1.0:
            sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
            sorted_to_remove = cumulative_probs > gen_kwargs["top_p"]
            if sorted_to_remove.any():
                sorted_to_remove = sorted_to_remove.roll(1, dims=-1)
                sorted_to_remove[:, 0] = False
            for b in range(logits.size(0)):
                logits[b, sorted_indices[b, sorted_to_remove[b]]] = -float("inf")
        return (
            torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            .squeeze(-1)
            .reshape(shape)
        )

    def pred_multiple_tokens(hidden_states, sample=False):
        proj = model.mtp_projection(hidden_states)
        proj = proj.view(proj.size(0), SPAN_LENGTH, -1)
        if model.embeddings_tied:
            logits = proj @ model.get_embd(model.student, model.model_name).weight.t()
        else:
            logits = model.student.lm_head(proj)
        if sample:
            return _sample_from_logits(logits)
        return torch.argmax(logits, dim=-1)

    # Precompute gold spans (MTP) or gold token ids (CoT) per sample
    if MODE == "mtp":
        gold_spans = []  # per sample: list of span token-id lists
        for r in records:
            ids = tokenizer.encode(format_cot(r["cot"]), add_special_tokens=False)
            ids = ids + [mot_id] * ((-len(ids)) % SPAN_LENGTH)
            spans = [ids[i : i + SPAN_LENGTH] for i in range(0, len(ids), SPAN_LENGTH)]
            gold_spans.append(spans)
    else:
        gold_cot_ids = [
            tokenizer.encode(format_cot(r["cot"]), add_special_tokens=False)
            for r in records
        ]

    eot_tensor = torch.tensor(eot_id, device="cuda")

    # MTP
    def generate_mtp(input_ids, attn, gspans, nf):
        B = input_ids.size(0)
        gtens = [[torch.tensor(s, dtype=torch.long, device="cuda") for s in gsp] for gsp in gspans]
        finished_thinking = [False] * B
        finished = [False] * B
        span_k = [0] * B
        ans_count = [0] * B
        pred_tokens = [[] for _ in range(B)]
        new_mask_col = torch.ones((B, 1), dtype=attn.dtype, device="cuda")

        def decide(b, topids, topids_s):
            if not finished_thinking[b]:
                k = span_k[b]
                if k < nf[b]:
                    span_k[b] += 1
                    return normal_agg(gtens[b][k]).view(1, 1, -1)
                has_eot = bool((topids == eot_id).any())
                if has_eot or (k - nf[b]) >= MAX_FREE_THINK:
                    finished_thinking[b] = True
                    return get_embedding(eot_tensor).view(1, 1, -1)
                agg = topids_s if SAMPLE_MTP else topids
                span_k[b] += 1
                return normal_agg(agg).view(1, 1, -1)
            next_token = topids[0] if GREEDY else topids_s[0]
            pred_tokens[b].append(next_token.item())
            ans_count[b] += 1
            if next_token.item() == eos_id or ans_count[b] >= MAX_ANSWER_TOKENS:
                finished[b] = True
            return get_embedding(next_token).view(1, 1, -1)

        with torch.no_grad():
            out = model.student(
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
                past_key_values=None,
                output_hidden_states=True,
            )
            past = out.past_key_values
            hidden = out.hidden_states[-1][:, -1, :]
            mt = pred_multiple_tokens(hidden)
            mts = pred_multiple_tokens(hidden, sample=True)
            next_embd = torch.cat([decide(b, mt[b], mts[b]) for b in range(B)], dim=0)

            budget = max(nf) + MAX_FREE_THINK + MAX_ANSWER_TOKENS + 5
            for _ in range(budget):
                attn = torch.cat([attn, new_mask_col], dim=1)
                out = model.student(
                    inputs_embeds=next_embd,
                    attention_mask=attn,
                    use_cache=True,
                    past_key_values=past,
                    output_hidden_states=True,
                )
                past = out.past_key_values
                hidden = out.hidden_states[-1][:, -1, :]
                mt = pred_multiple_tokens(hidden)
                mts = pred_multiple_tokens(hidden, sample=True)
                next_inps = []
                for b in range(B):
                    if finished[b]:
                        next_inps.append(next_embd[b].view(1, 1, -1))
                    else:
                        next_inps.append(decide(b, mt[b], mts[b]))
                next_embd = torch.cat(next_inps, dim=0)
                if all(finished):
                    break

        return [
            extract_answer_str(tokenizer.decode(pt, skip_special_tokens=False))
            for pt in pred_tokens
        ]

    def eval_mtp_at_x(x):
        preds, filled = [], 0
        for start in tqdm(range(0, N, BATCH_SIZE), desc=f"x={x:.2f}", leave=False):
            idxs = list(range(start, min(start + BATCH_SIZE, N)))
            seqs, gspans, nf = [], [], []
            for i in idxs:
                S = len(gold_spans[i])
                n = int(round(x * S))
                if n == S:
                    filled += 1
                seqs.append([bos_id] + q_tok[i] + [bot_id])
                gspans.append(gold_spans[i])
                nf.append(n)
            input_ids = left_pad(seqs, pad_id)
            attn = (input_ids != pad_id).long()
            preds.extend(generate_mtp(input_ids, attn, gspans, nf))
            torch.cuda.empty_cache()
        return compute_accuracy(answers, preds), filled

    # CoT
    def generate_cot(input_ids, attn):
        B = input_ids.size(0)
        finished = [False] * B
        pred = [[] for _ in range(B)]
        cnt = [0] * B
        new_mask_col = torch.ones((B, 1), dtype=attn.dtype, device="cuda")
        with torch.no_grad():
            out = model.student(
                input_ids=input_ids, attention_mask=attn, use_cache=True, past_key_values=None
            )
            past = out.past_key_values
            nt = _sample_from_logits(out.logits[:, -1, :])
            for b in range(B):
                pred[b].append(nt[b].item())
                cnt[b] += 1
            for _ in range(MAX_NEW_TOKENS_COT - 1):
                attn = torch.cat([attn, new_mask_col], dim=1)
                out = model.student(
                    input_ids=nt.unsqueeze(1),
                    attention_mask=attn,
                    use_cache=True,
                    past_key_values=past,
                )
                past = out.past_key_values
                nt = _sample_from_logits(out.logits[:, -1, :])
                for b in range(B):
                    if not finished[b]:
                        pred[b].append(nt[b].item())
                        cnt[b] += 1
                        if nt[b].item() == eos_id or cnt[b] >= MAX_NEW_TOKENS_COT:
                            finished[b] = True
                if all(finished):
                    break
        return [
            extract_answer_str(tokenizer.decode(p, skip_special_tokens=False)) for p in pred
        ]

    def eval_cot_at_x(x):
        preds, filled = [], 0
        for start in tqdm(range(0, N, BATCH_SIZE), desc=f"x={x:.2f}", leave=False):
            idxs = list(range(start, min(start + BATCH_SIZE, N)))
            seqs = []
            for i in idxs:
                T = len(gold_cot_ids[i])
                n = int(round(x * T))
                if n == T:
                    filled += 1
                seqs.append([bos_id] + q_tok[i] + [bot_id] + gold_cot_ids[i][:n])
            input_ids = left_pad(seqs, pad_id)
            attn = (input_ids != pad_id).long()
            preds.extend(generate_cot(input_ids, attn))
            torch.cuda.empty_cache()
        return compute_accuracy(answers, preds), filled

    # run sweep
    eval_at_x = eval_mtp_at_x if MODE == "mtp" else eval_cot_at_x
    accs, fills = [], []
    for x in X_VALUES:
        acc, filled = eval_at_x(x)
        accs.append(acc)
        fills.append(filled)
        print(f"x={x:.2f}  acc={acc * 100:6.2f}%  fully_filled={filled}/{N}", flush=True)

    # table
    w = 8
    print("\n" + "=" * 40 + " RESULTS " + "=" * 40)
    print("x        " + "".join(f"{x:>{w}.2f}" for x in X_VALUES))
    print("acc(%)   " + "".join(f"{a * 100:>{w}.1f}" for a in accs))
    print("n_filled " + "".join(f"{f:>{w}d}" for f in fills))

    # save
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "generations")
    tag = f"{MODE}_{COT_FORMAT}_sp{SPAN_LENGTH}"
    ds_name = os.path.splitext(os.path.basename(DATASET))[0] if ".json" in DATASET else DATASET
    ds_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ds_name)
    out_path = os.path.join(out_dir, f"gold_trace_{tag}_{ds_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "mode": MODE,
                    "cot_format": COT_FORMAT,
                    "span_length": SPAN_LENGTH,
                    "ckpt_dir": CKPT_DIR,
                    "model_name_or_path": MODEL_NAME_OR_PATH,
                    "dataset": DATASET,
                    "num_samples": N,
                    "greedy": GREEDY,
                    "sample_mtp": SAMPLE_MTP,
                },
                "x_values": X_VALUES,
                "accuracy": accs,
                "n_filled": fills,
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
