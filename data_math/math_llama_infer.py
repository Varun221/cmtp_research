"""Inference for LLaMA-3.2-1B-Instruct on MATH

Generation config set to zero-shot version of:
https://huggingface.co/datasets/meta-llama/Llama-3.2-1B-Instruct-evals/viewer/Llama-3.2-1B-Instruct-evals__math__details

"""

import json
import re
import os
import argparse

from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_grader import grade_answer

NUM_TIMES = 1

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
# ../data_store/math
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "data_store",
    "math",
    "math_train.json",
)


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


def load_data(path: str):
    with open(path) as f:
        data = json.load(f)
    return data


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "user", "content": PROMPT_TEMPLATE.format(problem=question)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content, handling nested braces."""
    pattern = r"\\boxed\{"
    matches = [m.start() for m in re.finditer(pattern, text)]
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


def run_one_prompt(model: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(model=model, dtype="bfloat16", trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    warmup_prompt = build_prompt(tokenizer, "What is 1 + 1?")
    llm.generate([warmup_prompt], sampling_params)
    print("Model ready.")

    question = input("Question: ").strip()
    prompt = build_prompt(tokenizer, question)
    output = llm.generate([prompt], sampling_params)[0].outputs[0].text
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--one-prompt", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=5120)
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    if args.one_prompt:
        run_one_prompt(args.model, args.max_new_tokens)
        return

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_num_seqs=args.batch_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    print(f"Loading data: {args.data}")
    data = load_data(args.data)
    print(f"Samples: {len(data)}")

    prompts = [build_prompt(tokenizer, item["question"]) for item in data]

    per_run_accuracies = []
    all_token_lengths = []
    results = None

    for run_idx in range(NUM_TIMES):
        print(f"\nRun {run_idx + 1}/{NUM_TIMES} — running inference...")
        raw_outputs = llm.generate(prompts, sampling_params)

        correct = 0
        results = []
        for item, req_output in tqdm(zip(data, raw_outputs), total=len(data)):
            output = req_output.outputs[0].text
            token_len = len(req_output.outputs[0].token_ids)
            pred_raw = extract_boxed(output)
            gold = item["answer"]
            pred = pred_raw or ""
            match = grade_answer(pred, gold)
            if match:
                correct += 1
            all_token_lengths.append(token_len)
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "pred": pred,
                    "match": match,
                    "full_output": output,
                    "token_length": token_len,
                }
            )

        run_acc = correct / len(data) * 100
        per_run_accuracies.append(run_acc)
        print(f"  Run {run_idx + 1} accuracy: {correct}/{len(data)} = {run_acc:.2f}%")

    acc_mean = np.mean(per_run_accuracies)
    acc_std = np.std(per_run_accuracies)
    len_mean = np.mean(all_token_lengths)
    len_std = np.std(all_token_lengths)

    print(f"\n=== Results over {NUM_TIMES} run(s) x {len(data)} samples ===")
    print(f"Accuracy:     {acc_mean:.2f}% ± {acc_std:.2f}%")
    print(f"Token length: {len_mean:.1f} ± {len_std:.1f}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")

    for r in results[:3]:
        print("\n---")
        print(f"Q: {r['question'][:120]}...")
        print(f"Gold: {r['gold']}")
        print(f"Pred: {r['pred']}")
        print(f"Match: {r['match']}")


if __name__ == "__main__":
    main()
