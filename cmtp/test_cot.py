"""Evaluation of CoT models with different eval datasets."""

import logging
import math
import re
import os
import sys
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
import json

from tqdm.auto import tqdm
from peft import LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    TrainingModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from scripts.math_grader import grade_answer

do_print = False
DUMP = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def read_json(file_path):
    """
    Read a JSON file from the specified path and return the corresponding Python object.。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error occurred while reading the JSON file: {e}")
        return None


def evaluation(model_args, data_args, training_args):
    if model_args.lora_init:
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
            raise ValueError(
                f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}."
            )
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        lora_config = None

    model_args.train = False
    model = TrainingModel(model_args, training_args, lora_config)

    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

    model.load_state_dict(state_dict, strict=False)

    if model.student.config.tie_word_embeddings:
        model.student.tie_weights()

    tokenizer_path = model_args.ckpt_dir
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.max_token_num + 10,
        padding_side="left",
        use_fast=False,
    )

    assert tokenizer.pad_token_id is not None, "Already set for training."

    bot_id = tokenizer.convert_tokens_to_ids("<lbot>")
    assert model.bot_id == bot_id, "Bot id not match."
    eot_id = tokenizer.convert_tokens_to_ids("<leot>")

    device = "cuda"
    model = model.to("cuda")
    model.to(torch.bfloat16)

    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm8k-hard" == data_args.data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset["train"]
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_args.data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset["test"]
        answer_name = "final_ans"
    elif "svamp" == data_args.data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "gsm8k-test" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset["test"]
    elif "gsm8k-val" == data_args.data_name:
        test_set = read_json(
            "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json"
        )
    elif ".json" in data_args.data_name:
        test_set = read_json(data_args.data_name)
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = [example[question_name] for example in test_set]
    answer = []

    # get answer as string
    for example in test_set:
        example = example[answer_name]
        if isinstance(example, bool):
            answer.append(example)
            continue
        if example in ["True", "False"]:
            answer.append(example == "True")
            continue
        if example in "ABCDE":
            answer.append(example)
            continue
        if "####" in example:
            ans = example.split("####")[-1]
        else:
            ans = example
        ans = ans.replace(",", "").strip()
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question) / data_args.batch_size)
    logging.warning(
        f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
        f"eval steps: {eval_step}"
    )

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            question_list = question[
                i * data_args.batch_size : (i + 1) * data_args.batch_size
            ]
        else:
            question_list = question[i * data_args.batch_size :]

        questions_tokenized = [
            tokenizer.encode(q, add_special_tokens=False) for q in question_list
        ]
        questions_tokenized = [
            torch.tensor([tokenizer.bos_token_id] + qt + [bot_id], dtype=torch.long)
            for qt in questions_tokenized
        ]

        max_len = max(len(qt) for qt in questions_tokenized)
        padded_input_ids = []
        for qt in questions_tokenized:
            pad_len = max_len - len(qt)
            pads = torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            padded_input_ids.append(torch.cat([pads, qt]))

        questions_inputids = torch.stack(padded_input_ids)

        batch = {
            "input_ids": questions_inputids.to(device),
            "attention_mask": (questions_inputids != tokenizer.pad_token_id)
            .long()
            .to(device),
        }
        question_data.append(batch)

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 2048,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    dump_records = [] if DUMP else None
    len_cot = []
    len_ans = []
    model.eval()

    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    def _sample_from_logits(logits):
        if training_args.greedy:
            next_token_ids = torch.argmax(logits, dim=-1)
        else:
            logits /= gen_kwargs["temperature"]
            if gen_kwargs["top_k"] > 1:
                top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < min_top_k_value] = -float("inf")

            if gen_kwargs["top_p"] < 1.0:
                sorted_logit, sorted_indices = torch.sort(
                    logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                if sorted_indices_to_remove.any():
                    sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                    sorted_indices_to_remove[:, 0] = False

                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float(
                        "inf"
                    )

            probs = F.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token_ids

    for step, batch in tqdm(enumerate(question_data), total=len(question_data)):
        batch_size = batch["input_ids"].size(0)

        attention_mask = batch["attention_mask"]
        new_mask_col = torch.ones((batch_size, 1), dtype=attention_mask.dtype).to(
            attention_mask.device
        )

        with torch.no_grad():
            past_key_values = None
            outputs = model.student(
                input_ids=batch["input_ids"],
                use_cache=True,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            next_token_ids = _sample_from_logits(logits)

            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]

            for b in range(batch_size):
                pred_tokens[b].append(next_token_ids[b].item())

            for i in range(gen_kwargs["max_new_tokens"] - 1):
                attention_mask = torch.cat([attention_mask, new_mask_col], dim=1)

                seq_len += 1

                out = model.student(
                    input_ids=next_token_ids.unsqueeze(1),
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

                next_token_ids = _sample_from_logits(logits)
                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

            for mini_step, pred_token in enumerate(pred_tokens):

                if eot_id in pred_token:
                    len_cot.append(pred_token.index(eot_id) + 1)
                if tokenizer.eos_token_id in pred_token:
                    len_ans.append(pred_token.index(tokenizer.eos_token_id) + 1)

                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=False)

                if do_print:
                    print(f"Question {step*data_args.batch_size+mini_step} Starts...")
                    print(f"Q: {question[step*data_args.batch_size+mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step*data_args.batch_size+mini_step} Ends")
                    print(
                        f"Prediction={extract_answer_str(decoded_pred)}; Groundtruth={answer[step*data_args.batch_size+mini_step]}"
                    )
                    print("")
                ans_pred_list.append(extract_answer_str(decoded_pred))

                if DUMP:
                    idx = step * data_args.batch_size + mini_step
                    decoded_pred_clean = tokenizer.decode(
                        pred_token, skip_special_tokens=True
                    )
                    dump_records.append({
                        "idx": idx,
                        "question": question[idx],
                        "pred_tokens": pred_token,
                        "pred_tokens_decoded": [
                            tokenizer.decode([tok], skip_special_tokens=False)
                            for tok in pred_token
                        ],
                        "pred_decoded": decoded_pred,
                        "pred_decoded_clean": decoded_pred_clean,
                        "extracted_answer": ans_pred_list[-1],
                        "ground_truth": answer[idx],
                        "len_cot": pred_token.index(eot_id) + 1
                        if eot_id in pred_token
                        else None,
                        "len_ans": pred_token.index(tokenizer.eos_token_id) + 1
                        if tokenizer.eos_token_id in pred_token
                        else None,
                    })

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(
        f"adapter: {model_args.adapter_name_or_path} | {data_args.data_name} accuracy: {100*accuracy:.2f}% | "
    )

    print(
        "Avg Length of CoT (± std), Avg Length of Answer (± std): {:.2f} (±{:.2f}), {:.2f} (±{:.2f})".format(
            np.mean(len_cot), np.std(len_cot), np.mean(len_ans), np.std(len_ans)
        )
    )

    if DUMP:
        eval_name = re.sub(r"[^A-Za-z0-9._-]+", "_", data_args.data_name)
        if eval_name.endswith(".json"):
            eval_name = os.path.splitext(os.path.basename(eval_name))[0]
        dump_path = os.path.join(model_args.ckpt_dir, f"dump_cot_{eval_name}.json")
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(
                {"accuracy": accuracy, "records": dump_records},
                f,
                indent=2,
                default=str,
            )
        print(f"Dumped {len(dump_records)} records to {dump_path}")

    return 100 * accuracy, len_cot


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


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    accu_list = []
    cot_lens = []
    for i in range(training_args.inf_num_iterations):
        accu, len_cot = evaluation(model_args, data_args, training_args)
        accu_list.append(accu)
        cot_lens.extend(len_cot)

    print(
        f"Average accuracy over {training_args.inf_num_iterations} sampling: {sum(accu_list)/len(accu_list)}"
    )
    print(
        f"Average CoT length over {training_args.inf_num_iterations} sampling: {np.mean(cot_lens)} (±{np.std(cot_lens)})"
    )
