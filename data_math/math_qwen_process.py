"""Process Qwen inference outputs into training-ready data.

Reads mathinfer_outputs/qwen_train.json and data_store/math/math_test.json.
Writes to data_store/math/:
  qwen_train_full.json    — all train items where cot contains \\boxed{}
  qwen_train_correct.json — subset of above where model answered correctly
  qwen_test.json          — test split with qwen chat template applied to gold cot
"""

import json
import os
import re

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

_HERE = os.path.dirname(os.path.abspath(__file__))
INFER_TRAIN_PATH = os.path.join(_HERE, "mathinfer_outputs", "qwen_train.json")
TEST_PATH = os.path.join(_HERE, "..", "data_store", "math", "math_test.json")
OUT_DIR = os.path.join(_HERE, "..", "data_store", "math")


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def has_boxed(text: str) -> bool:
    return bool(re.search(r"\\boxed\{", text))


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # --- train ---
    print(f"Loading train inference: {INFER_TRAIN_PATH}")
    with open(INFER_TRAIN_PATH) as f:
        train_data = json.load(f)
    print(f"Total train samples: {len(train_data)}")

    train_full = []
    train_correct = []
    skipped_no_boxed = 0

    for item in train_data:
        cot = item["full_output"]
        if not has_boxed(cot):
            skipped_no_boxed += 1
            continue
        record = {
            "question": build_prompt(tokenizer, item["question"]),
            "cot": cot,
            "answer": item["pred"],
        }
        train_full.append(record)
        if item["match"]:
            train_correct.append(record)

    print(f"Skipped (no \\boxed in cot): {skipped_no_boxed}")
    print(f"train_full: {len(train_full)}")
    print(f"train_correct: {len(train_correct)}")

    full_path = os.path.join(OUT_DIR, "qwen_train_full.json")
    correct_path = os.path.join(OUT_DIR, "qwen_train_correct.json")
    with open(full_path, "w") as f:
        json.dump(train_full, f)
    with open(correct_path, "w") as f:
        json.dump(train_correct, f)
    print(f"Saved: {full_path}")
    print(f"Saved: {correct_path}")

    # --- test ---
    print(f"\nLoading test data: {TEST_PATH}")
    with open(TEST_PATH) as f:
        test_data = json.load(f)
    print(f"Total test samples: {len(test_data)}")

    test_results = [
        {
            "question": build_prompt(tokenizer, item["question"]),
            "cot": item["cot"],
            "answer": item["answer"],
        }
        for item in test_data
    ]
    print(f"test: {len(test_results)}")

    test_path = os.path.join(OUT_DIR, "qwen_test.json")
    with open(test_path, "w") as f:
        json.dump(test_results, f)
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
