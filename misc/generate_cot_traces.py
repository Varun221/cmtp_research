"""Generate CoT traces with a CoT checkpoint, keep the correct ones, and save.

Runs a trained CoT model over a dataset, decodes each chain-of-thought, grades
the extracted answer against the gold answer, and writes the correct traces as a
JSON list of {question, cot, answer} into assets/generations/. That JSON can then be fed
to gold_trace_analysis.py (DATASET=<the saved file>) as in-distribution gold
traces for the matching MTP checkpoint.

Model loading / dataset handling mirror test_cot.py. Set the globals below and:
    python misc/generate_cot_traces.py
"""

import os
import re
import sys
import json
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
from peft import LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file
from tqdm.auto import tqdm

CMTP_DIR = "/scratch/vy2142/cmtp_research/cmtp"
sys.path.insert(0, CMTP_DIR)
sys.path.insert(0, os.path.join(CMTP_DIR, "scripts"))
from src.model import TrainingModel, TrainingArguments, ModelArguments  # noqa: E402
from scripts.math_grader import grade_answer  # noqa: E402

# CONFIG
COT_CKPT_DIR = "/scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_llama_lora"
MODEL_NAME_OR_PATH = "meta-llama/Llama-3.2-1B-Instruct"

# Dataset to generate over: "gsm8k-test" / "gsm8k-hard" / "multi-arith" / "svamp",
# or a JSON file path (list of dicts). Only needs a question + gold answer.
# DATASET = "gsm8k-test"
DATASET = "svamp"

BATCH_SIZE = 32
MAX_SAMPLES = None      # int to subsample for a quick run
MAX_NEW_TOKENS = 512    # cot + answer generation budget
GREEDY = True

LORA_INIT = True        # True for LoRA checkpoints, False for full-finetuned ones
LORA_R = 128
LORA_ALPHA = 32
HF_TOKEN = None

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "generations")


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


def normalize_answer(val):
    if isinstance(val, bool):
        return val
    val = str(val)
    if val in ["True", "False"]:
        return val == "True"
    if val in "ABCDE":
        return val
    ans = val.split("####")[-1] if "####" in val else val
    return ans.replace(",", "").strip()


def load_model_and_tokenizer():
    model_args = ModelArguments(
        model_name_or_path=MODEL_NAME_OR_PATH,
        ckpt_dir=COT_CKPT_DIR,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_init=LORA_INIT,
        token=HF_TOKEN,
        train=False,
    )
    training_args = TrainingArguments(
        output_dir=COT_CKPT_DIR,
        bf16=True,
        greedy=GREEDY,
        compile_forward=False,
        max_token_num=2048,
    )

    lora_config = None
    if LORA_INIT:
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
        state_dict = load_file(os.path.join(COT_CKPT_DIR, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(COT_CKPT_DIR, "pytorch_model.bin"))
    model.load_state_dict(state_dict, strict=False)
    if model.student.config.tie_word_embeddings:
        model.student.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        COT_CKPT_DIR,
        token=HF_TOKEN,
        model_max_length=training_args.max_token_num + 10,
        padding_side="left",
        use_fast=False,
    )
    assert tokenizer.pad_token_id is not None
    assert model.bot_id == tokenizer.convert_tokens_to_ids("<lbot>")
    assert model.eot_id == tokenizer.convert_tokens_to_ids("<leot>")

    model = model.to("cuda").to(torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def resolve_gen_dataset(name):
    """Load a dataset as {question, answer} records (no gold CoT needed)."""
    if name == "gsm8k-test":
        data = load_dataset("gsm8k", "main")["test"]
        qk, ak = "question", "answer"
    elif name == "gsm8k-hard":
        data = load_dataset("juyoung-trl/gsm-hard")["train"]
        qk, ak = "instruction", "response"
    elif name == "multi-arith":
        data = load_dataset("ChilleD/MultiArith")["test"]
        qk, ak = "question", "final_ans"
    elif name == "svamp":
        svamp = load_dataset("ChilleD/SVAMP")
        data = concatenate_datasets([svamp["train"], svamp["test"]])
        qk, ak = "question_concat", "Answer"
    elif ".json" in name:
        data = read_json(name)
        qk, ak = "question", "answer"
    else:
        raise ValueError(f"Unknown dataset: {name}")

    records = [
        {
            "question": f"{ex[qk]}".strip().replace("  ", " "),
            "answer": normalize_answer(ex[ak]),
        }
        for ex in data
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
    model, tokenizer = load_model_and_tokenizer()
    records = resolve_gen_dataset(DATASET)
    N = len(records)
    print(f"Generating CoT traces | dataset={DATASET} | samples={N}")

    bos_id = tokenizer.bos_token_id
    bot_id = model.bot_id
    eot_id = model.eot_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    q_tok = [tokenizer.encode(r["question"], add_special_tokens=False) for r in records]

    gen_kwargs = {"temperature": 0.1, "top_k": 40, "top_p": 0.95}

    def _sample_from_logits(logits):
        if GREEDY:
            return torch.argmax(logits, dim=-1)
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
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).squeeze(-1)

    def generate(input_ids, attn):
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
            for _ in range(MAX_NEW_TOKENS - 1):
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
                        if nt[b].item() == eos_id or cnt[b] >= MAX_NEW_TOKENS:
                            finished[b] = True
                if all(finished):
                    break
        return pred

    kept = []
    n_correct = 0
    n_no_eot = 0
    for start in tqdm(range(0, N, BATCH_SIZE)):
        idxs = list(range(start, min(start + BATCH_SIZE, N)))
        seqs = [[bos_id] + q_tok[i] + [bot_id] for i in idxs]
        input_ids = left_pad(seqs, pad_id)
        attn = (input_ids != pad_id).long()
        preds = generate(input_ids, attn)
        for i, pred in zip(idxs, preds):
            gold = records[i]["answer"]
            decoded = tokenizer.decode(pred, skip_special_tokens=False)
            if not grade_answer(str(extract_answer_str(decoded)), str(gold)):
                continue
            n_correct += 1
            if eot_id not in pred:
                n_no_eot += 1
                continue
            cot_tokens = pred[: pred.index(eot_id)]
            cot = tokenizer.decode(cot_tokens, skip_special_tokens=True).strip()
            kept.append({"question": records[i]["question"], "cot": cot, "answer": gold})
        torch.cuda.empty_cache()


    ds_tag = os.path.splitext(os.path.basename(DATASET))[0] if ".json" in DATASET else DATASET
    out_path = os.path.join(OUTPUT_DIR, f"cot_lora_exp_correct_generations_{ds_tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)

    print(
        f"\ntotal={N}  correct={n_correct} ({100 * n_correct / N:.2f}%)  "
        f"correct_without_eot_dropped={n_no_eot}  saved={len(kept)}"
    )
    print(f"Saved traces to {out_path}")


if __name__ == "__main__":
    main()
