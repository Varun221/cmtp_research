import logging
import re
from dataclasses import dataclass
from typing import Dict, Sequence
import torch
import json
import transformers
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from src.model import TrainingArguments
import json


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


IGNORE_INDEX = -100

# Counter({1: 62908, 2: 143578, 3: 104249, 4: 48198, 5: 17906, 6: 5666, 7: 2359, 8: 577, 9: 126, 10: 43, 11: 8, 12: 1, 13: 1})
DATA_PATHS = {
    "icot": {
        "train": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_train.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json",
    },
    "icot-subset": {
        "train": "/scratch/vy2142/cmtp_research/data_store/randomsubsets/gsm8kaug_train_100k.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json",
    },
    "icot-nl": {
        "train": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_nl_train.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_nl_val.json",
    },
    "icot-nl-subset": {
        "train": "/scratch/vy2142/cmtp_research/data_store/randomsubsets/gsm8kaug_nl_train_100k.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_nl_val.json",
    },
    "mathqwen": {
        "train": "/scratch/vy2142/cmtp_research/data_store/math/qwen_train_correct.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json",
    },
    "mathllama": {
        "train": "/scratch/vy2142/cmtp_research/data_store/math/llama_train_correct.json",
        "eval": "/scratch/vy2142/cmtp_research/data_store/math/llama_test.json",
    },
}


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""

    tokenized_list = [
        tokenizer.encode(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )[0]
        for text in strings
    ]

    return tokenized_list


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    segment = [sentence]
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r"-?\d+\.?\d*", pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float("inf")
    return pred_answer


def _tokenizer_prepends_bos(token_ids_list, bos_token_id):
    for t in token_ids_list:
        if len(t) > 0:
            return t[0].item() == bos_token_id
    return False


def get_answer_token_position(tokens, answer_prompts, tokenizer):
    # answer_prompt = torch.tensor([464, 3280, 318, 25])
    # breakpoint()

    # Takes last occurrence in case of multiple.
    try:
        match_indices = (
            (tokens.unfold(0, len(answer_prompts[0]), 1) == answer_prompts[0])
            .all(dim=1)
            .nonzero(as_tuple=True)[0][-1:]
            .item()
        )
        answer_token_id = match_indices + len(answer_prompts[0])
        return answer_token_id
    except Exception:
        breakpoint()


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    answers: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    bot_id: int,
    eot_id: int,
    training_args: TrainingArguments,
) -> Dict:
    print("Tokenizing inputs... This may take some time...")
    sources_id = _tokenize_fn(sources, tokenizer)
    cot_id = _tokenize_fn(targets, tokenizer)
    answers_id = _tokenize_fn(answers, tokenizer)

    # breakpoint()

    answers_id = [
        torch.tensor(x.numpy().tolist() + [tokenizer.eos_token_id], dtype=torch.long)
        for x in answers_id
    ]

    if _tokenizer_prepends_bos(cot_id, tokenizer.bos_token_id):
        cot_id = [x[1:] for x in cot_id]
        answers_id = [x[1:] for x in answers_id]

    if training_args.add_thinktoken_in_ref:
        # add bot to source
        sources_id = [
            torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long)
            for x in sources_id
        ]
        # add eot
        answers_id = [
            torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long)
            for x in answers_id
        ]

    ref_input_ids = [
        torch.cat([x, y, z]).to(torch.long)
        for x, y, z in zip(sources_id, cot_id, answers_id)
    ]
    ref_labels = []
    for x, y in zip(ref_input_ids, sources_id):
        z = x.clone()
        z[: len(y)] = -100
        ref_labels.append(z)

    if not training_args.add_thinktoken_in_ref:
        # add bot to source
        sources_id = [
            torch.tensor(x.numpy().tolist() + [bot_id], dtype=torch.long)
            for x in sources_id
        ]
        # add eot
        answers_id = [
            torch.tensor([eot_id] + x.numpy().tolist(), dtype=torch.long)
            for x in answers_id
        ]

    answer_prompts = [
        torch.tensor(tokenizer.encode("The answer is:")),
        torch.tensor(tokenizer.encode("The next step result is:")),
    ]
    if answer_prompts[0][0] == tokenizer.bos_token_id:  # remove the bos
        answer_prompts[0] = answer_prompts[0][1:]
        answer_prompts[1] = answer_prompts[1][1:]
    # breakpoint()
    ref_answer_position = [
        get_answer_token_position(x, answer_prompts, tokenizer)
        for i, x in enumerate(ref_input_ids)
    ]
    model_answer_position = [
        get_answer_token_position(x, answer_prompts, tokenizer) for x in answers_id
    ]

    ref_eos_position = [len(x) - 1 for x in ref_input_ids]
    model_eos_position = [len(x) - 1 for x in answers_id]

    return dict(
        encoder_input_ids=sources_id,
        decoder_input_ids=answers_id,
        ref_input_ids=ref_input_ids,
        labels=answers_id,
        ref_answer_position=ref_answer_position,
        model_answer_position=model_answer_position,
        ref_eos_position=ref_eos_position,
        model_eos_position=model_eos_position,
        ref_labels=ref_labels,
    )


class SupervisedDataset(Dataset):
    QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
    QUESTION_DA_PROMPT = (
        "\nAnswer the above question. Answer the final number directly in one number.\n"
    )

    def __init__(
        self, data_args, raw_data, tokenizer, bot, eot, training_args, eval=False
    ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Formatting inputs...")

        data_name = data_args.data_name
        self.data_name = data_name
        questions, cots, answers = [], [], []
        num_ops_list = []
        operators = ["+", "-", "*", "/"]

        for num_iter, example in tqdm(enumerate(raw_data), total=len(raw_data)):

            cot = example.get("cot", None)
            if cot is None:
                raise ValueError("COT is missing in the data example.")

            if (training_args.exp_mode) and (num_iter > training_args.exp_data_num):
                break

            question = f"{example['question']}"

            # if data_name in ("icot-nl", "icot-nl-subset"):  # icot-full (GSM8k-Aug-NL)
            if "icot-nl" in data_name:
                # bad data
                if example["answer"] is None:
                    continue

                if not eval:
                    # avoid OOM: remove very long data
                    token_num = len(
                        tokenizer.encode(
                            example["question"] + example["cot"] + example["answer"]
                        )
                    )
                    if token_num > training_args.max_token_num:
                        continue

                cot = f"{example['cot']}".split(". ")
                if not (training_args.include_last_cot):
                    cot = cot[:-1]

                answer = example["answer"].split(" ")[-1]
                if not eval:
                    if not answer[0].isdigit():
                        continue
                answer = f"The answer is: {answer}"
                answer = answer.replace("####", "")
                questions.append(question)

                if cot:
                    cot = ". ".join(cot) + ".\n"
                else:
                    cot = ""

                # For openr1 -- remove answer in cot.
                cot = cot.replace(answer, "").strip()

                cots.append(cot)
                answers.append(answer)

            elif data_name in ("icot", "icot-subset"):  # icot (GSM8k-Aug)
                if not eval:
                    # avoid OOM: remove very long data
                    token_num = len(
                        tokenizer.encode(
                            example["question"] + example["cot"] + example["answer"]
                        )
                    )
                    if token_num > training_args.max_token_num:
                        continue

                cot_list = []
                cot = f"{example['cot']}".split(" ")
                if not training_args.include_last_cot:
                    cot = cot[:-1]

                len_cot = len(cot)
                for i in range(training_args.num_latent):
                    cot_list.append(" ".join(cot[: max(0, len_cot - i)]))
                answer = example["answer"].split(" ")[-1]

                # some answers startwith the negative sign (-), bringing distillation problems for LLaMA
                if not eval:
                    if not answer[0].isdigit():
                        continue

                answer = f"The answer is: {answer}"
                answer = answer.replace("####", "")
                questions.append(question)
                cots.append(" ".join(cot))
                answers.append(answer)

            elif "math" in self.data_name:

                # bad data
                if example["answer"] is None:
                    continue

                if not eval:
                    # avoid OOM: remove very long data
                    token_num = len(
                        tokenizer.encode(
                            example["question"] + example["cot"] + example["answer"]
                        )
                    )
                    if token_num > training_args.max_token_num:
                        continue

                answer = example["answer"].strip()

                answer = f"The answer is: \\boxed{{{answer}}}"
                questions.append(question)

                cots.append(cot)
                answers.append(answer)
            else:
                raise NotImplementedError

            if training_args.exp_mode:
                questions = questions[: training_args.exp_data_num]
                cots = cots[: training_args.exp_data_num]
                answers = answers[: training_args.exp_data_num]

        print(f"{len(cots)} data in total...")
        logging.warning("Tokenizing inputs... This may take some time...")

        self.data_dict = preprocess(
            questions, cots, answers, tokenizer, bot, eot, training_args
        )
        self.keys = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_dict["encoder_input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: self.data_dict[key][i] for key in self.keys}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (
            encoder_input_ids,
            decoder_input_ids,
            ref_input_ids,
            labels,
            ref_answer_position,
            model_answer_position,
            ref_labels,
        ) = tuple(
            [instance.get(key, None) for instance in instances]
            for key in (
                "encoder_input_ids",
                "decoder_input_ids",
                "ref_input_ids",
                "labels",
                "ref_answer_position",
                "model_answer_position",
                "ref_labels",
            )
        )

        # pad left
        reversed_input_ids = [seq.flip(0) for seq in encoder_input_ids]
        encoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(1)
        # B, T [Flip back in 2nd dimension.]

        # pad
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        ref_labels = torch.nn.utils.rnn.pad_sequence(
            ref_labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            decoder_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            ref_input_ids=ref_input_ids,
            labels=labels,
            encoder_attention_mask=encoder_input_ids.ne(self.tokenizer.pad_token_id),
            ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
            model_answer_position=torch.tensor(model_answer_position, dtype=torch.long),
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            ref_labels=ref_labels,
        )


def create_data_module(
    model_args, data_args, training_args, pad_token_id, bot_id, eot_id
):
    """Create data module for training."""

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.max_token_num + 10,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.pad_token_id is not None:
        print("Removing existing pad token from tokenizer.")
        tokenizer._pad_token = None

    if len(tokenizer) < pad_token_id:
        num_gap = pad_token_id - len(tokenizer)
        print(f"Adding {num_gap} unused tokens to tokenizer.")
        tokenizer.add_tokens([f"<unused_{i}>" for i in range(num_gap)])

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        tokenizer.pad_token_id = pad_token_id
        if tokenizer.pad_token_id is None:  # error handling
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
        print(
            f"Did not find bos_token_id. Setting bos_token_id to {tokenizer.bos_token_id}."
        )

    logging.warning("Processing data...")
    if data_args.data_name in DATA_PATHS:

        train_path = DATA_PATHS[data_args.data_name]["train"]
        eval_path = DATA_PATHS[data_args.data_name]["eval"]

        if data_args.train_path_override is not None:
            train_path = data_args.train_path_override

        if data_args.eval_path_override is not None:
            eval_path = data_args.eval_path_override

        print(f"Loading train data from {train_path}...")
        train_dataset = SupervisedDataset(
            data_args=data_args,
            raw_data=read_json(train_path),
            tokenizer=tokenizer,
            bot=bot_id,
            eot=eot_id,
            training_args=training_args,
        )
        print(f"Loaded {len(train_dataset)} training samples.")

        print(f"Loading eval data from {eval_path}...")
        eval_dataset = SupervisedDataset(
            data_args=data_args,
            raw_data=read_json(eval_path),
            tokenizer=tokenizer,
            bot=bot_id,
            eot=eot_id,
            training_args=training_args,
            eval=True,
        )
        print(f"Loaded {len(eval_dataset)} evaluation samples.")

        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if data_args.data_name in DATA_PATHS else None,
        data_collator=data_collator,
    )

    return data_module, tokenizer


if __name__ == "__main__":
    pass
