"""MATH dataset preparation script.

Uses: https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark
Filters training data to <= 512 tokens (problem + cot + answer,
measured with LLaMA-3.2-1B tokenizer).

Run once to prepare train and test splits stored in data_store/math.
"""

import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer

TRAIN_NAME = "math_train.json"
TEST_NAME = "math_test.json"

TOKENIZER_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
MAX_TOKENS = 512


def to_records(df):
    return [
        {"question": row.problem, "answer": row.answer, "cot": row.solution}
        for _, row in df.iterrows()
    ]


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_store_dir = os.path.join(curr_dir, "..", "data_store", "math")
    if os.path.exists(data_store_dir):
        print(f"Warning: {data_store_dir} already exists and will be overwritten.")
    os.makedirs(data_store_dir, exist_ok=True)

    data = load_dataset("nlile/hendrycks-MATH-benchmark")
    print(f"Loaded dataset: {data}")

    traindf = data["train"].to_pandas()
    testdf = data["test"].to_pandas()
    print(f"Original: train={len(traindf)}, test={len(testdf)}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    def token_count(row):
        return (
            len(tokenizer.encode(row["problem"] + row["solution"] + row["answer"])) + 10
        )

    traindf["token_count"] = traindf.apply(token_count, axis=1)
    traindf = traindf[traindf["token_count"] <= MAX_TOKENS].reset_index(drop=True)
    print(f"After token filter (<={MAX_TOKENS}): train={len(traindf)}")

    train_data = to_records(traindf)
    test_data = to_records(testdf)

    with open(os.path.join(data_store_dir, TRAIN_NAME), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(data_store_dir, TEST_NAME), "w") as f:
        json.dump(test_data, f)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
