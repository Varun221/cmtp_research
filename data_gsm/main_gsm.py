"""GSM8k-Aug data preparation script.

CODI uses dataset available at: https://huggingface.co/datasets/zen-E/GSM8k-Aug
This does not have the validation split, hence we use the equivalent source: https://huggingface.co/datasets/whynlp/gsm8k-aug

Run once to prepare expressions (structured) and natural language
(semi-natural) versions of the dataset.
The prepared jsons will be stored in data_store/ directory.
"""

import os
from datasets import load_dataset
import json
import random

EXP_TRAIN_NAME = "gsm8kaug_train.json"
EXP_VAL_NAME = "gsm8kaug_val.json"

NL_TRAIN_NAME = "gsm8kaug_nl_train.json"
NL_VAL_NAME = "gsm8kaug_nl_val.json"


def make_cot_key(recs):
    for rec in recs:
        rec["cot"] = " ".join(rec.pop("steps").tolist())


def filter_bad_answers(recs):
    return_recs = []
    for rec in recs:
        answer = rec["answer"].split(" ")[-1]
        # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
        if not answer[0].isdigit():
            continue
        return_recs.append(rec)
    return return_recs


def main():
    # make a directory data_store if doesn't exist.
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_store_dir = os.path.join(curr_dir, "..", "data_store")
    os.makedirs(data_store_dir, exist_ok=True)

    exp_data = load_dataset("whynlp/gsm8k-aug")
    nl_data = load_dataset("whynlp/gsm8k-aug-nl")

    print(f"Loaded datasets from HuggingFace. {exp_data}, {nl_data}")

    train_recs = exp_data["train"].to_pandas().to_dict(orient="records")
    val_recs = exp_data["validation"].to_pandas().to_dict(orient="records")

    train_nl_recs = nl_data["train"].to_pandas().to_dict(orient="records")
    val_nl_recs = nl_data["validation"].to_pandas().to_dict(orient="records")
    print(f"Original Exp; train: {len(train_recs)}, val: {len(val_recs)}")
    print(f"Original NL; train: {len(train_nl_recs)}, val: {len(val_nl_recs)}")

    make_cot_key(train_recs)
    make_cot_key(val_recs)
    make_cot_key(train_nl_recs)
    make_cot_key(val_nl_recs)

    train_recs = filter_bad_answers(train_recs)
    val_recs = filter_bad_answers(val_recs)
    train_nl_recs = filter_bad_answers(train_nl_recs)
    val_nl_recs = filter_bad_answers(val_nl_recs)

    print(f"Filtered Exp; train: {len(train_recs)}, val: {len(val_recs)}")
    print(f"Filtered NL; train: {len(train_nl_recs)}, val: {len(val_nl_recs)}")

    # Add 1% extra data for slightly larger validation.
    random.seed(1000)
    random.shuffle(train_recs)

    frac = 0.99
    train_data = train_recs[: int(frac * len(train_recs))]
    eval_data = train_recs[int(frac * len(train_recs)) :]
    print(f"Split train into train: {len(train_data)}, val: {len(eval_data)}")

    # split train_nl using the questions in train_data
    train_questions = set([rec["question"] for rec in train_data])

    train_nl_data = []
    eval_nl_data = []
    for rec in train_nl_recs:
        if rec["question"] in train_questions:
            train_nl_data.append(rec)
        else:
            eval_nl_data.append(rec)
    print(f"Split NL train into train: {len(train_nl_data)}, val: {len(eval_nl_data)}")

    final_exp_train = train_data
    final_exp_val = eval_data + val_recs
    final_nl_train = train_nl_data
    final_nl_val = eval_nl_data + val_nl_recs

    print(f"\nFinal Exp; train: {len(final_exp_train)}, val: {len(final_exp_val)}")
    print(f"Final NL; train: {len(final_nl_train)}, val: {len(final_nl_val)}")

    if os.path.exists(os.path.join(data_store_dir, EXP_TRAIN_NAME)):
        print(
            f"Data seems to be already prepared at {data_store_dir}. Will overwrite the existing data."
        )

    # Save the final datasets
    with open(os.path.join(data_store_dir, EXP_TRAIN_NAME), "w") as f:
        json.dump(final_exp_train, f)
    with open(os.path.join(data_store_dir, EXP_VAL_NAME), "w") as f:
        json.dump(final_exp_val, f)
    with open(os.path.join(data_store_dir, NL_TRAIN_NAME), "w") as f:
        json.dump(final_nl_train, f)
    with open(os.path.join(data_store_dir, NL_VAL_NAME), "w") as f:
        json.dump(final_nl_val, f)

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
