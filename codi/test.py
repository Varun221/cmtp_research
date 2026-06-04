import logging
import math
import re
import os
import sys
from typing import Optional

import torch
import transformers
from torch.nn import functional as F
from tqdm.auto import tqdm
import json

from peft import LoraConfig, TaskType
from datasets import load_dataset, concatenate_datasets
from safetensors.torch import load_file

import numpy as np

from src.model import (
    TrainingModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

from scripts.math_grader import grade_answer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Path to the locally-prepared GSM8k-Aug validation file (see data_gsm/README.md).
# Update this to point at your own data_store after preparing the data.
EVAL_PATH = "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json"

do_print = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def read_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        print(f"Error occurred while reading the JSON file: {e}")
        return None


def load_model(model_args, training_args):
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
        raise NotImplementedError

    model_args.train = False
    model = TrainingModel(model_args, training_args, lora_config)
    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer_path = model_args.model_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    model = model.to("cuda")
    model.to(torch.bfloat16)

    return model, tokenizer


def run_eval(model, tokenizer, model_args, data_args, training_args, data_name):
    ######################
    #      dataset       #
    ######################
    logging.warning("Downloading Data")
    question_name = "question"
    answer_name = "answer"
    if "gsm8k-hard" == data_name:
        dataset = load_dataset("juyoung-trl/gsm-hard")
        test_set = dataset["train"]
        question_name = "instruction"
        answer_name = "response"
    elif "multi-arith" == data_name:
        dataset = load_dataset("ChilleD/MultiArith")
        test_set = dataset["test"]
        answer_name = "final_ans"
    elif "svamp" == data_name:
        dataset = load_dataset("ChilleD/SVAMP")
        test_set = concatenate_datasets([dataset["train"], dataset["test"]])
        question_name = "question_concat"
        answer_name = "Answer"
    elif "gsm8k-test" == data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset["test"]
    elif "gsm8k-val" == data_name:
        test_set = read_json(EVAL_PATH)
    elif data_name.endswith(".json"):
        test_set = read_json(data_name)
    else:
        raise NotImplementedError

    logging.warning("Formatting inputs...")
    question = [
        f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set
    ]
    answer = []

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
        f"Total example: {len(question)} | eval batch size: {data_args.batch_size} | "
        f"eval steps: {eval_step}"
    )

    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i * data_args.batch_size : (i + 1) * data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i * data_args.batch_size :],
                return_tensors="pt",
                padding="longest",
            )

        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(
            batch["input_ids"].size(0), 1
        )
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat(
            (batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1
        )
        batch["input_len"] = len(batch["input_ids"][0])
        question_data.append(batch.to("cuda"))

    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }

    ans_pred_list = []
    len_cot = []

    for step, batch in tqdm(enumerate(question_data), total=len(question_data)):
        batch_size = batch["input_ids"].size(0)
        with torch.no_grad():
            past_key_values = None
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=batch["attention_mask"],
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            eot_emb = (
                model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device="cuda")
                )
                .unsqueeze(0)
                .to("cuda")
            )
            eot_emb = eot_emb.expand(batch["input_ids"].size(0), -1, -1)

            output = eot_emb
            seq_len = 0
            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]

            for i in range(gen_kwargs["max_new_tokens"]):
                seq_len += 1
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, : model.codi.config.vocab_size - 1]

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(
                            logits, gen_kwargs["top_k"], dim=-1
                        )
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(
                            logits, descending=True, dim=-1
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logit, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = (
                            cumulative_probs > gen_kwargs["top_p"]
                        )
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(
                                1, dims=-1
                            )
                            sorted_indices_to_remove[:, 0] = False
                        for b in range(logits.size(0)):
                            logits[
                                b, sorted_indices[b, sorted_indices_to_remove[b]]
                            ] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = (
                    model.get_embd(model.codi, model.model_name)(next_token_ids)
                    .unsqueeze(1)
                    .to("cuda")
                )

            for mini_step, pred_token in enumerate(pred_tokens):
                len_cot.append(len(pred_token))
                decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
                if do_print:
                    print(
                        f"Question {step * data_args.batch_size + mini_step} Starts..."
                    )
                    print(f"Q: {question[step * data_args.batch_size + mini_step]}")
                    print(decoded_pred)
                    print(f"Question {step * data_args.batch_size + mini_step} Ends")
                    print(
                        f"Prediction={extract_answer_str(decoded_pred)}; "
                        f"Groundtruth={answer[step * data_args.batch_size + mini_step]}"
                    )
                    print("")
                ans_pred_list.append(extract_answer_str(decoded_pred))

    accuracy = compute_accuracy(answer, ans_pred_list)
    print(
        f"adapter: {model_args.adapter_name_or_path} | {data_name} accuracy: {100 * accuracy:.2f}%"
    )
    print(f"average length of COT: {sum(len_cot) / len(len_cot)}")

    return 100 * accuracy


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
    m = re.search(r"[Tt]he answer is:?\s*(.+?)(?:\.|$)", sentence)
    if m is not None:
        return m.group(1).strip()
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

    eval_names = [name.strip() for name in data_args.data_name.split(",")]

    model, tokenizer = load_model(model_args, training_args)

    results = {}
    for data_name in eval_names:
        print(f"\n=== Running eval: {data_name} ===")
        accu_list = []
        for i in range(training_args.inf_num_iterations):
            accu = run_eval(
                model, tokenizer, model_args, data_args, training_args, data_name
            )
            accu_list.append(accu)
        avg = sum(accu_list) / len(accu_list)
        results[data_name] = avg
        print(
            f"Average accuracy over {training_args.inf_num_iterations} sampling [{data_name}]: {avg:.2f}% +- {np.std(accu_list):.2f}%"
        )

    print("\n=== Summary ===")
    for data_name, avg in results.items():
        print(f"  {data_name}: {avg:.2f}% +- {np.std(accu_list):.2f}%")
