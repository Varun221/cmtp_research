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
from src import utils


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


"""
Possible data path inputs

icot
icot-nl
icot-randsubset-xk ; x=12, 25, 50, 60, 70, 88, 100, 120, 150, 200, 250, 300, 350
icot-nl-randsubset-xk ; x=12, 25, 50, 60, 70, 88, 100, 120, 150, 200, 250, 300, 350
mathllama
mathqwen
"""


def get_data_paths(data_name):

    if "mathllama" in data_name:
        return (
            "/scratch/vy2142/cmtp_research/data_store/math/llama_train_correct.json",
            "/scratch/vy2142/cmtp_research/data_store/math/llama_test.json",
        )

    if "mathqwen" in data_name:
        return (
            "/scratch/vy2142/cmtp_research/data_store/math/qwen_train_correct.json",
            "/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json",
        )

    exp_eval_path = "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_val.json"
    nl_eval_path = "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_nl_val.json"

    if "nl" in data_name:
        if "randsubset" in data_name:
            subset_size = data_name.split("-")[-1]
            train_path = f"/scratch/vy2142/cmtp_research/data_store/randomsubsets/gsm8kaug_nl_train_{subset_size}.json"
        else:
            train_path = (
                "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_nl_train.json"
            )

        eval_path = nl_eval_path
    else:
        # Assume non-nl is expressions always.
        if "randsubset" in data_name:
            subset_size = data_name.split("-")[-1]
            train_path = f"/scratch/vy2142/cmtp_research/data_store/randomsubsets/gsm8kaug_train_{subset_size}.json"
        else:
            train_path = "/scratch/vy2142/cmtp_research/data_store/gsm8kaug_train.json"

        eval_path = exp_eval_path

    return train_path, eval_path


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""

    tokenized_list = [
        tokenizer.encode(
            text,
            truncation=True,
            add_special_tokens=False,
        )
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
    training_args: TrainingArguments,
) -> Dict:
    print("Tokenizing inputs... This may take some time...")

    # breakpoint()

    bot_id = tokenizer.convert_tokens_to_ids("<lbot>")
    mot_id = tokenizer.convert_tokens_to_ids("<lmot>")
    eot_id = tokenizer.convert_tokens_to_ids("<leot>")

    sources_id = _tokenize_fn(sources, tokenizer)
    cot_id = _tokenize_fn(targets, tokenizer)
    answers_id = _tokenize_fn(answers, tokenizer)
    # breakpoint()

    # add bos to sources
    sources_id = [[tokenizer.bos_token_id] + source for source in sources_id]

    # add eos to answers
    answers_id = [answer + [tokenizer.eos_token_id] for answer in answers_id]

    cot_id, persample_spanids = utils.equal_segmentation(
        cot_id=cot_id,
        bot_id=bot_id,
        eot_id=eot_id,
        mot_id=mot_id,
        span_length=training_args.span_length,
    )

    # fmt: off
    ref_input_ids = []
    span_segment_ids = []
    for x, y, sp, z in zip(sources_id, cot_id, persample_spanids, answers_id):
        ref_input_ids.append(x + y + z)
        span_segment_ids.append([0,]*len(x) + sp + [0,]*(len(z)))

    loss_masks = []
    for x, y in zip(sources_id, ref_input_ids):
        lm = [0,] * len(x) + [1,] * (len(y) - len(x))
        loss_masks.append(lm)
    # fmt: on

    # Convert to Tensors
    ref_input_ids = [torch.tensor(x, dtype=torch.long) for x in ref_input_ids]
    span_segment_ids = [torch.tensor(x, dtype=torch.long) for x in span_segment_ids]
    loss_masks = [torch.tensor(x, dtype=torch.long) for x in loss_masks]

    answer_prompts = [
        torch.tensor(tokenizer.encode("The answer is:", add_special_tokens=False)),
        torch.tensor(
            tokenizer.encode("The next step result is:", add_special_tokens=False)
        ),
    ]

    # breakpoint()
    ref_answer_position = [
        get_answer_token_position(x, answer_prompts, tokenizer)
        for i, x in enumerate(ref_input_ids)
    ]

    ref_eos_position = [len(x) - 1 for x in ref_input_ids]

    # breakpoint()

    return dict(
        ref_input_ids=ref_input_ids,
        loss_masks=loss_masks,
        span_segment_ids=span_segment_ids,
        ref_answer_position=ref_answer_position,
        ref_eos_position=ref_eos_position,
    )


class SupervisedDataset(Dataset):

    # Bad signature, fix later.
    def __init__(
        self,
        data_args=None,
        raw_data=None,
        tokenizer=None,
        training_args=None,
        data_dict=None,
        eval=False,
    ):
        super(SupervisedDataset, self).__init__()

        data_name = data_args.data_name
        questions, cots, answers = [], [], []

        if raw_data is None:
            raise ValueError("Raw data must be provided in the current code.")

        self.training_args = training_args

        for num_iter, example in tqdm(enumerate(raw_data), total=len(raw_data)):

            if "icot" in data_name:

                cot = example.get("cot", None)
                if cot is None:
                    raise ValueError("COT is missing in the data example.")

                if (training_args.exp_mode) and (num_iter > training_args.exp_data_num):
                    break

                question = f"{example['question']}"

                if not eval:
                    # avoid OOM: remove very long data
                    token_num = len(
                        tokenizer.encode(
                            example["question"] + example["cot"] + example["answer"]
                        )
                    )
                    if token_num > training_args.max_token_num:
                        logging.warning(
                            f"Ignoring sample due to excessive token length ({token_num} tokens)."
                        )
                        continue

                if "nl" in data_name:
                    # nl is sep by fullstop
                    cot = f"{example['cot']}".split(". ")
                else:
                    # expressions are sep by spaces
                    cot = f"{example['cot']}".split(" ")

                if not training_args.include_last_cot:
                    cot = cot[:-1]

                answer = example["answer"].split(" ")[-1]

                # some answers start with the negative sign (-), bringing distillation problems for LLaMA
                if not eval:
                    if not answer[0].isdigit():
                        logging.warning(
                            f"Ignoring sample due to non-numeric answer: {answer}"
                        )
                        continue

                answer = f"The answer is: {answer}"
                answer = answer.replace("####", "")

                questions.append(question)
                if "nl" in data_name:
                    # Add . between nl.
                    cots.append(". ".join(cot))
                else:
                    # Add \n between expressions.
                    cots.append("\n".join(cot))

                answers.append(answer)

            elif "math" in data_name:

                question = example["question"]

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
                answers.append(answer)

                questions.append(question)

                cots.append(example["cot"])

            else:
                raise ValueError(f"Data name {data_name} not recognized.")

        if training_args.exp_mode:
            questions = questions[: training_args.exp_data_num]
            cots = cots[: training_args.exp_data_num]
            answers = answers[: training_args.exp_data_num]

        print(f"{len(cots)} data in total...")
        logging.warning("Tokenizing inputs... This may take some time...")

        self.data_dict = preprocess(
            sources=questions,
            targets=cots,
            answers=answers,
            tokenizer=tokenizer,
            training_args=training_args,
        )
        self.keys = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_dict["ref_input_ids"])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {key: self.data_dict[key][i] for key in self.keys}


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        (
            ref_input_ids,
            loss_masks,
            span_segment_ids,
            ref_answer_position,
        ) = tuple(
            [instance.get(key, None) for instance in instances]
            for key in (
                "ref_input_ids",
                "loss_masks",
                "span_segment_ids",
                "ref_answer_position",
            )
        )

        # Pad Everything to Left.
        ref_input_ids = torch.nn.utils.rnn.pad_sequence(
            ref_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side="left",
        )
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            loss_masks,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        span_segment_ids = torch.nn.utils.rnn.pad_sequence(
            span_segment_ids,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )

        return dict(
            ref_input_ids=ref_input_ids,
            loss_masks=loss_masks,
            span_segment_ids=span_segment_ids,
            ref_attention_mask=ref_input_ids.ne(self.tokenizer.pad_token_id),
            ref_answer_position=torch.tensor(ref_answer_position, dtype=torch.long),
        )


def create_data_module(
    model_args, data_args, training_args, pad_token_id, bot_id, mot_id, eot_id
):
    """Create data module for training."""

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.max_token_num + 10,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is not None:
        print("Removing existing pad token from tokenizer.")
        tokenizer._pad_token = None

    if len(tokenizer) < pad_token_id:
        num_gap = pad_token_id - len(tokenizer)
        tokenizer.add_tokens([f"<unused_{i}>" for i in range(num_gap)])

        print(
            f"Added {num_gap} unused tokens to tokenizer to reach pad_token_id {pad_token_id}."
        )

    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.pad_token_id = pad_token_id
    tokenizer.add_tokens(["<lbot>", "<lmot>", "<leot>"])

    assert tokenizer.encode("<lbot><lmot><leot>", add_special_tokens=False) == [
        bot_id,
        mot_id,
        eot_id,
    ], "Special tokens are not correctly added."

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
        print(
            f"Did not find bos_token_id. Setting bos_token_id to {tokenizer.bos_token_id}."
        )

    logging.warning("Processing data...")
    print(f"Loading {data_args.data_name} dataset...")

    train_path, eval_path = get_data_paths(data_args.data_name)

    if data_args.train_path_override is not None:
        train_path = data_args.train_path_override
    if data_args.eval_path_override is not None:
        eval_path = data_args.eval_path_override

    logging.warning(f"Train path: {train_path}")
    logging.warning(f"Eval path: {eval_path}")

    train_dataset = SupervisedDataset(
        data_args=data_args,
        raw_data=read_json(train_path),
        tokenizer=tokenizer,
        training_args=training_args,
    )
    print(f"Loaded {len(train_dataset)} training samples.")
    eval_dataset = SupervisedDataset(
        data_args=data_args,
        raw_data=read_json(eval_path),
        tokenizer=tokenizer,
        training_args=training_args,
    )
    print(f"Loaded {len(eval_dataset)} evaluation samples.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    return data_module, tokenizer


if __name__ == "__main__":
    pass
