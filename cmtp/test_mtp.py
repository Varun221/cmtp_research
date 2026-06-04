"""Evaluation of MTP models with different eval datasets."""

import sys
import math
import re
import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
from peft import LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file
from tqdm.auto import tqdm
import numpy as np

from src.model import TrainingModel, TrainingArguments, DataArguments, ModelArguments

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.math_grader import grade_answer

# Path to the locally-prepared GSM8k-Aug validation file (see data_gsm/README.md).
# Update this to point at your own data_store after preparing the data.
EVAL_PATH = "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json"


@dataclass
class EvalArguments:
    evlist: str = field(
        default="",
        metadata={"help": "Comma-separated list of eval sets or a JSON file path."},
    )
    verbose: bool = field(
        default=False, metadata={"help": "Print verbose per-example output."}
    )
    dump: bool = field(
        default=False, metadata={"help": "Dump per-example records to a JSON file."}
    )


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None


def _extract_boxed(text: str) -> Optional[str]:
    """Extract last \\boxed{...} content, handling nested braces."""
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


def compute_accuracy(gold: list, pred: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if isinstance(p, list):
            if any(grade_answer(str(pi), str(g)) for pi in p):
                acc += 1
        else:
            if grade_answer(str(p), str(g)):
                acc += 1
    return acc / len(gold)


def main(model_args, data_args, training_args, eval_args):

    evlist = eval_args.evlist
    verbose = eval_args.verbose
    dump = eval_args.dump
    model_args.train = False

    task_type = TaskType.CAUSAL_LM
    if any(
        name in model_args.model_name_or_path.lower()
        for name in ["llama", "mistral", "falcon", "qwen"]
    ):
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
    elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
    elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    model = TrainingModel(model_args, training_args, lora_config)

    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

    model.load_state_dict(state_dict, strict=False)
    tie_embeds = model.student.config.tie_word_embeddings
    if tie_embeds:
        model.student.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.ckpt_dir,
        token=model_args.token,
        model_max_length=training_args.max_token_num + 10,
        padding_side="left",
        use_fast=False,
    )

    assert tokenizer.pad_token_id is not None
    bot_id = tokenizer.convert_tokens_to_ids("<lbot>")
    assert model.bot_id == bot_id

    model = model.to("cuda").to(torch.bfloat16)
    model.eval()

    gen_kwargs = {
        "max_new_tokens": training_args.max_token_num,
        "temperature": 0.2,
        "top_k": 40,
        "top_p": 0.95,
    }

    def _sample_from_logits(logits):
        shape = logits.shape[:-1]
        logits = logits.reshape(-1, logits.size(-1))
        if training_args.greedy:
            return torch.argmax(logits, dim=-1).reshape(shape)
        logits /= gen_kwargs["temperature"]
        if gen_kwargs["top_k"] > 1:
            top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
            logits[logits < top_k_values[:, -1].unsqueeze(-1)] = -float("inf")
        if gen_kwargs["top_p"] < 1.0:
            sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
            if sorted_indices_to_remove.any():
                sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                sorted_indices_to_remove[:, 0] = False
            for b in range(logits.size(0)):
                logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float(
                    "inf"
                )
        return (
            torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            .squeeze(-1)
            .reshape(shape)
        )

    def get_embedding(ids):
        return model.get_embd(model.student, model.model_name)(ids)

    def normal_agg(top_ids):
        return get_embedding(top_ids).mean(dim=0, keepdim=True)

    def pred_multiple_tokens(hidden_states, sample=False):
        proj_states = model.mtp_projection(hidden_states)
        proj_states = proj_states.view(
            proj_states.size(0), training_args.span_length, -1
        )
        if tie_embeds:
            logits = (
                proj_states @ model.get_embd(model.student, model.model_name).weight.t()
            )
        else:
            logits = model.student.lm_head(proj_states)
        if sample:
            return _sample_from_logits(logits)
        return torch.argmax(logits, dim=-1)

    def evaluate(test_set, question_name, answer_name, eval_name="eval"):
        question = [
            f"{example[question_name].strip().replace('  ', ' ')}"
            for example in test_set
        ]

        answer = []
        for example in test_set:
            val = example[answer_name]
            if isinstance(val, bool):
                answer.append(val)
                continue
            if val in ["True", "False"]:
                answer.append(val == "True")
                continue
            if val in "ABCDE":
                answer.append(val)
                continue
            ans = val.split("####")[-1] if "####" in val else val
            answer.append(ans.replace(",", "").strip())

        eval_step = math.ceil(len(question) / data_args.batch_size)
        question_data = []
        for i in range(eval_step):
            question_list = question[
                i * data_args.batch_size : (i + 1) * data_args.batch_size
            ]
            questions_tokenized = [
                tokenizer.encode(q, add_special_tokens=False) for q in question_list
            ]
            questions_tokenized = [
                torch.tensor([tokenizer.bos_token_id] + qt + [bot_id], dtype=torch.long)
                for qt in questions_tokenized
            ]
            max_len = max(len(qt) for qt in questions_tokenized)
            padded = []
            for qt in questions_tokenized:
                pads = torch.full(
                    (max_len - len(qt),), tokenizer.pad_token_id, dtype=torch.long
                )
                padded.append(torch.cat([pads, qt]))
            questions_inputids = torch.stack(padded)
            question_data.append(
                {
                    "input_ids": questions_inputids.to("cuda"),
                    "attention_mask": (questions_inputids != tokenizer.pad_token_id)
                    .long()
                    .to("cuda"),
                }
            )

        ans_pred_list = []
        len_cot = []
        dump_records = [] if dump else None
        FORCE_THINK = 0

        for step, batch in tqdm(enumerate(question_data), total=len(question_data)):
            batch_size = batch["input_ids"].size(0)
            attention_mask = batch["attention_mask"]

            with torch.no_grad():
                outputs = model.student(
                    input_ids=batch["input_ids"],
                    use_cache=True,
                    past_key_values=None,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                past_key_values = outputs.past_key_values

                think_count = torch.zeros(batch_size, dtype=torch.int, device="cuda")
                finished_thinking = torch.zeros(
                    batch_size, dtype=torch.bool, device="cuda"
                )
                finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
                pred_tokens = [[] for _ in range(batch_size)]
                mtp_think_tokens = (
                    [[] for _ in range(batch_size)] if (verbose or dump) else None
                )

                topids = pred_multiple_tokens(outputs.hidden_states[-1][:, -1, :])
                next_inps = []
                new_mask_col = torch.ones(
                    (batch_size, 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                for b in range(batch_size):
                    if model.eot_id in topids[b]:
                        finished_thinking[b] = True
                        next_inps.append(
                            get_embedding(
                                torch.tensor(model.eot_id, device="cuda")
                            ).view(1, 1, -1)
                        )
                    else:
                        if verbose or dump:
                            mtp_think_tokens[b].append(topids[b].tolist())
                        next_inps.append(normal_agg(topids[b]).view(1, 1, -1))
                next_embd = torch.cat(next_inps, dim=0)

                for i in range(gen_kwargs["max_new_tokens"] - 1):
                    attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)
                    out = model.student(
                        inputs_embeds=next_embd,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                    )
                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :]

                    for b in range(batch_size):
                        if FORCE_THINK > 0 and think_count[b] < FORCE_THINK:
                            logits[b, model.eot_id] = -float("inf")

                    topids = pred_multiple_tokens(out.hidden_states[-1][:, -1, :])
                    topids_sampled = pred_multiple_tokens(
                        out.hidden_states[-1][:, -1, :], sample=True
                    )
                    next_inps = []

                    for b in range(batch_size):
                        if finished[b]:
                            next_inps.append(next_embd[b].view(1, 1, -1))
                            continue
                        if not finished_thinking[b]:
                            has_eot = (topids[b] == model.eot_id).any()
                            if has_eot or think_count[b] >= 500:
                                finished_thinking[b] = True
                                next_inps.append(
                                    get_embedding(
                                        torch.tensor(model.eot_id, device="cuda")
                                    ).view(1, 1, -1)
                                )
                            else:
                                think_count[b] += 1
                                agg_inp = (
                                    topids_sampled[b]
                                    if training_args.sample_mtp
                                    else topids[b]
                                )
                                if verbose or dump:
                                    mtp_think_tokens[b].append(agg_inp.tolist())
                                next_inps.append(normal_agg(agg_inp).view(1, 1, -1))
                        else:
                            next_token = (
                                topids[b, 0]
                                if training_args.greedy
                                else topids_sampled[b, 0]
                            )
                            pred_tokens[b].append(next_token.item())
                            next_inps.append(get_embedding(next_token).view(1, 1, -1))
                            if next_token == tokenizer.eos_token_id:
                                finished[b] = True

                    next_embd = torch.cat(next_inps, dim=0)
                    if finished.all():
                        break

            for mini_step, pred_token in enumerate(pred_tokens):
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=False)
                if finished_thinking[mini_step]:
                    len_cot.append(think_count[mini_step].item())
                ans_pred = extract_answer_str(decoded_pred)
                ans_pred_list.append(ans_pred)
                if dump:
                    idx = step * data_args.batch_size + mini_step
                    dump_records.append(
                        {
                            "idx": idx,
                            "question": question[idx],
                            "mtp_think_tokens": mtp_think_tokens[mini_step],
                            "mtp_think_tokens_decoded": [
                                [
                                    tokenizer.decode(tok, skip_special_tokens=False)
                                    for tok in step_toks
                                ]
                                for step_toks in mtp_think_tokens[mini_step]
                            ],
                            "pred_tokens": pred_token,
                            "pred_decoded": decoded_pred,
                            "extracted_answer": ans_pred,
                            "ground_truth": answer[idx],
                            "think_count": think_count[mini_step].item(),
                            "finished_thinking": bool(
                                finished_thinking[mini_step].item()
                            ),
                        }
                    )
                if verbose:
                    idx = step * data_args.batch_size + mini_step
                    print(f"\n--- Example {idx} ---", flush=True)
                    print(f"Question: {question[idx]}", flush=True)
                    decoded_think = [
                        [
                            tokenizer.decode(tok, skip_special_tokens=False)
                            for tok in step_toks
                        ]
                        for step_toks in mtp_think_tokens[mini_step]
                    ]
                    think_str = " ".join(f"({' '.join(s)})" for s in decoded_think)
                    print(
                        f"MTP think tokens ({len(decoded_think)} steps): {think_str}",
                        flush=True,
                    )
                    print(f"Decoded prediction: {decoded_pred}", flush=True)
                    print(f"Extracted answer: {ans_pred}", flush=True)
                    print(f"Ground truth: {answer[idx]}", flush=True)

            del (
                outputs,
                out,
                past_key_values,
                next_embd,
                topids,
                topids_sampled,
                attention_mask,
            )
            torch.cuda.empty_cache()

        accuracy = compute_accuracy(answer, ans_pred_list)

        if dump:
            dump_path = os.path.join(
                model_args.ckpt_dir,
                f"dump_mtp_{eval_name}_span{training_args.span_length}.json",
            )
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"accuracy": accuracy, "records": dump_records},
                    f,
                    indent=2,
                    default=str,
                )
            print(f"Dumped {len(dump_records)} records to {dump_path}")

        return accuracy, len_cot

    # --- Run evals ---

    if len(evlist) == 0:
        evlist = "gsm8k-val,gsm8k-test,gsm8k-hard,multi-arith,svamp"

    for ev in evlist.split(","):
        if ev == "gsm8k-val":
            print("=== GSM8k-val ===")
            dataset_to_eval = read_json(EVAL_PATH)
            q_key, a_key = "question", "answer"
        elif ev == "gsm8k-test":
            print("=== GSM8k-test ===")
            dataset_to_eval = load_dataset("gsm8k", "main")["test"]
            q_key, a_key = "question", "answer"
        elif ev == "gsm8k-hard":
            print("=== GSM8k-hard ===")
            dataset_to_eval = load_dataset("juyoung-trl/gsm-hard")["train"]
            q_key, a_key = "instruction", "response"
        elif ev == "multi-arith":
            print("=== MultiArith ===")
            dataset_to_eval = load_dataset("ChilleD/MultiArith")["test"]
            q_key, a_key = "question", "final_ans"
        elif ev == "svamp":
            print("=== SVAMP ===")
            svamp = load_dataset("ChilleD/SVAMP")
            dataset_to_eval = concatenate_datasets([svamp["train"], svamp["test"]])
            q_key, a_key = "question_concat", "Answer"
        elif ".json" in ev:
            print(f"=== {ev} ===")
            dataset_to_eval = read_json(ev)
            q_key, a_key = "question", "answer"
        else:
            print(f"Unknown eval: {ev}")
            continue

        eval_name = os.path.splitext(os.path.basename(ev))[0] if ".json" in ev else ev

        per_run_accs = []
        all_len_cot = []
        num_times = training_args.inf_num_iterations
        for run_idx in range(num_times):
            if num_times > 1:
                print(f"  Run {run_idx + 1}/{num_times}...")
            acc, len_cot = evaluate(dataset_to_eval, q_key, a_key, eval_name=eval_name)
            per_run_accs.append(acc)
            all_len_cot.extend(len_cot)
            print(
                f"  [run {run_idx + 1}] acc={acc*100:.2f}%  think steps={np.mean(len_cot):.2f} ± {np.std(len_cot):.2f}"
            )

        print(
            f"Accuracy:   {np.mean(per_run_accs)*100:.2f}% ± {np.std(per_run_accs)*100:.2f}%"
        )
        print(f"Think steps: {np.mean(all_len_cot):.2f} ± {np.std(all_len_cot):.2f}")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, EvalArguments)
    )
    model_args, data_args, training_args, eval_args = (
        parser.parse_args_into_dataclasses()
    )
    main(model_args, data_args, training_args, eval_args)
