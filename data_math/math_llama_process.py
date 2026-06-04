"""Process LLaMA inference outputs into training-ready data.

Reads mathinfer_outputs/llama_train.json and data_store/math/math_test.json.
Writes to data_store/math/:
  llama_train_full.json    — all train items where cot contains \\boxed{}
  llama_train_correct.json — subset of above where model answered correctly
  llama_test.json          — test split with llama chat template applied to gold cot
"""

import json
import os
import re

from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

PROMPT_TEMPLATE = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{{answer}}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem.\n\n"
    "Problem: {problem}\n"
)

_HERE = os.path.dirname(os.path.abspath(__file__))
INFER_TRAIN_PATH = os.path.join(_HERE, "mathinfer_outputs", "llama_train.json")
TEST_PATH = os.path.join(_HERE, "..", "data_store", "math", "math_test.json")
OUT_DIR = os.path.join(_HERE, "..", "data_store", "math")


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=question)}]
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

    full_path = os.path.join(OUT_DIR, "llama_train_full.json")
    correct_path = os.path.join(OUT_DIR, "llama_train_correct.json")
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

    test_path = os.path.join(OUT_DIR, "llama_test.json")
    with open(test_path, "w") as f:
        json.dump(test_results, f)
    print(f"Saved: {test_path}")


if __name__ == "__main__":
    main()
