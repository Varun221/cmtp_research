"""Generate random subsets of data for Sample Efficiency experiments."""

import json
import os
import random

EXP_TRAIN_NAME = "gsm8kaug_train.json"
NL_TRAIN_NAME = "gsm8kaug_nl_train.json"

SIZES = [
    12500,
    25000,
    50000,
    60000,
    70000,
    88000,
    100000,
    120000,
    150000,
    200000,
    250000,
    300000,
    350000,
]
NAMES = [
    "12k",
    "25k",
    "50k",
    "60k",
    "70k",
    "88k",
    "100k",
    "120k",
    "150k",
    "200k",
    "250k",
    "300k",
    "350k",
]


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_store_dir = os.path.join(curr_dir, "..", "data_store")
    out_dir = os.path.join(data_store_dir, "randomsubsets")
    if os.path.exists(out_dir):
        print(f"Warning: {out_dir} already exists and will be overwritten.")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_store_dir, EXP_TRAIN_NAME)) as f:
        exp_data = json.load(f)
    with open(os.path.join(data_store_dir, NL_TRAIN_NAME)) as f:
        nl_data = json.load(f)

    print(f"Loaded exp train: {len(exp_data)}, nl train: {len(nl_data)}")

    random.seed(42)
    random.shuffle(exp_data)

    nl_by_question = {rec["question"]: rec for rec in nl_data}

    for size, name in zip(SIZES, NAMES):
        exp_subset = exp_data[:size]

        # Make corresponding NL set.
        nl_subset = []
        for rec in exp_subset:
            if rec["question"] in nl_by_question:
                nl_subset.append(nl_by_question[rec["question"]])

        exp_path = os.path.join(out_dir, f"gsm8kaug_train_{name}.json")
        nl_path = os.path.join(out_dir, f"gsm8kaug_nl_train_{name}.json")

        with open(exp_path, "w") as f:
            json.dump(exp_subset, f)
        with open(nl_path, "w") as f:
            json.dump(nl_subset, f)

        print(f"{name}: exp={len(exp_subset)}, nl={len(nl_subset)}")

    print("Done.")


if __name__ == "__main__":
    main()
