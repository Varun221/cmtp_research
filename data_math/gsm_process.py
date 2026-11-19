"""Process LLM inference outputs on GSM8k-Aug into training-ready data.

Reads data_store/gsmllm/infer/llama_train.json (raw vllm outputs of
Llama-3.2-1B-Instruct with the fields question, gold, pred, match, full_output,
token_length).

Writes to data_store/gsmllm/:
  llama_train_full.json     — train items whose full_output contains \\boxed{}
  llama_train_correct.json  — subset of the above where match == True
  llama_val.json            — held-out 1% slice, correct-filtered
  llama_{evalset}.json      — downstream eval sets with the chat template applied

The last 1% of the raw list is held out as the val split before filtering.
Downstream eval sets are chat-templated so they match the training prompt format.
"""

import json
import os
import re

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from prompts import build_llama_prompt

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

_HERE = os.path.dirname(os.path.abspath(__file__))
GSMLLM_DIR = os.path.join(_HERE, "..", "data_store", "gsmllm")

# Downstream eval sets: (name, hf spec) -> question key, answer key.
EVAL_SETS = [
    ("gsm8k-test", "gsm8k-test"),
    ("gsm8k-hard", "gsm8k-hard"),
    ("multi-arith", "multi-arith"),
    ("svamp", "svamp"),
]


def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))


def process_split(raw_items, build_prompt, tokenizer):
    full, correct = [], []
    skipped_no_boxed = 0
    for item in raw_items:
        cot = item["full_output"]
        if not has_boxed(cot):
            skipped_no_boxed += 1
            continue
        record = {
            "question": build_prompt(tokenizer, item["question"]),
            "cot": cot,
            "answer": item["pred"],
        }
        full.append(record)
        if item["match"]:
            correct.append(record)
    return full, correct, skipped_no_boxed


def load_eval_set(name):
    if name == "gsm8k-test":
        ds = load_dataset("gsm8k", "main")["test"]
        return ds, "question", "answer"
    if name == "gsm8k-hard":
        ds = load_dataset("juyoung-trl/gsm-hard")["train"]
        return ds, "instruction", "response"
    if name == "multi-arith":
        ds = load_dataset("ChilleD/MultiArith")["test"]
        return ds, "question", "final_ans"
    if name == "svamp":
        svamp = load_dataset("ChilleD/SVAMP")
        ds = concatenate_datasets([svamp["train"], svamp["test"]])
        return ds, "question_concat", "Answer"
    raise ValueError(f"Unknown eval set: {name}")


def main():
    build_prompt = build_llama_prompt

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    os.makedirs(GSMLLM_DIR, exist_ok=True)

    infer_path = os.path.join(GSMLLM_DIR, "infer", f"llama_train.json")
    print(f"Loading inference: {infer_path}")
    with open(infer_path) as f:
        raw = json.load(f)
    print(f"Total raw samples: {len(raw)}")

    val_size = len(raw) // 100
    train_raw = raw[:-val_size]
    val_raw = raw[-val_size:]
    print(f"Train raw: {len(train_raw)}, Val raw: {len(val_raw)}")

    train_full, train_correct, train_skipped = process_split(
        train_raw, build_prompt, tokenizer
    )
    _, val_correct, val_skipped = process_split(val_raw, build_prompt, tokenizer)

    full_path = os.path.join(GSMLLM_DIR, f"llama_train_full.json")
    correct_path = os.path.join(GSMLLM_DIR, f"llama_train_correct.json")
    val_path = os.path.join(GSMLLM_DIR, f"llama_val.json")
    with open(full_path, "w") as f:
        json.dump(train_full, f)
    with open(correct_path, "w") as f:
        json.dump(train_correct, f)
    with open(val_path, "w") as f:
        json.dump(val_correct, f)

    print(f"Train skipped (no \\boxed): {train_skipped}")
    print(f"train_full: {len(train_full)}")
    print(f"train_correct: {len(train_correct)}")
    print(f"val (correct-filtered): {len(val_correct)} (skipped {val_skipped})")
    print(f"Saved: {full_path}")
    print(f"Saved: {correct_path}")
    print(f"Saved: {val_path}")

    for name, spec in EVAL_SETS:
        ds, q_key, a_key = load_eval_set(spec)
        records = [
            {
                "question": build_prompt(tokenizer, ex[q_key]),
                "answer": str(ex[a_key]),
            }
            for ex in ds
        ]
        out_path = os.path.join(GSMLLM_DIR, f"llama_{name}.json")
        with open(out_path, "w") as f:
            json.dump(records, f)
        print(f"{name}: {len(records)} -> {out_path}")


if __name__ == "__main__":
    main()
