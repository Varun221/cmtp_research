"""Token length (mean ± std) of the CoT traces in each training set.

Length is len(tokenizer.encode(cot)) with the Llama-3.2-1B-Instruct tokenizer
(BOS included), over the unfiltered train file of each setting.

Usage (from the repo root):
    python misc/trace_length_stats.py
    python misc/trace_length_stats.py --math data_store/math/llama_train_correct.json

Output:
    setting                n  length (mean ± std)
    structured        380015  21.46 ± 10.47
    semi_natural      380013  56.08 ± 28.01
    gsm_realistic     206701  134.29 ± 42.50
    math                3587  242.67 ± 146.69
"""

import argparse
import json

import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

SETTINGS = {
    "structured": "data_store/gsm8kaug_train.json",
    "semi_natural": "data_store/gsm8kaug_nl_train.json",
    "gsm_realistic": "data_store/gsmllm/llama_train_correct.json",
    "math": "data_store/math/llama_train_correct.json",
}


def main():
    parser = argparse.ArgumentParser()
    for name, path in SETTINGS.items():
        parser.add_argument(f"--{name}", default=path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"{'setting':<15}{'n':>9}  length (mean ± std)")
    for name in SETTINGS:
        with open(getattr(args, name)) as f:
            cots = [example["cot"] for example in json.load(f)]
        lengths = np.array([len(ids) for ids in tokenizer(cots)["input_ids"]])
        print(f"{name:<15}{len(lengths):>9}  {lengths.mean():.2f} ± {lengths.std():.2f}")


if __name__ == "__main__":
    main()
