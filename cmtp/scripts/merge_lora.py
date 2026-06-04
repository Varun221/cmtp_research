"""
Merges LoRA adapter from a training checkpoint into the base model using the
same setup as train.py, then saves the merged model to the specified output dir.

Usage: See merge_lora.sh for launch.
"""

import os
import torch
import transformers
from peft import LoraConfig, TaskType

# Add cmtp directory.
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import (
    TrainingModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from src.dataset import create_data_module
from src import utils
from transformers import Trainer

torch._dynamo.config.capture_scalar_outputs = True

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, num_items_in_batch):
        return 0.0

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return {}


def pop_arg(argv, flag):
    if flag in argv:
        idx = argv.index(flag)
        value = argv[idx + 1]
        return value, argv[:idx] + argv[idx + 2 :]
    return None, argv


def merge():
    argv = sys.argv[1:]

    start_state, argv = pop_arg(argv, "--start_state")
    assert start_state is not None, "--start_state is required"
    sys.argv = [sys.argv[0]] + argv

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    utils.seed_everything(training_args.seed)

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
        elif any(
            name in model_args.model_name_or_path.lower()
            for name in ["gpt2", "gsm-cot"]
        ):
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

    model = TrainingModel(model_args, training_args, lora_config)

    data_module, tokenizer = create_data_module(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        pad_token_id=model.pad_token_id,
        bot_id=model.bot_id,
        mot_id=model.mot_id,
        eot_id=model.eot_id,
    )

    training_args.remove_unused_columns = False

    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    print(f"Loading checkpoint from: {start_state}")
    trainer._load_from_checkpoint(start_state)

    print("Merging LoRA adapter into student model...")
    merged = trainer.model.student.merge_and_unload()
    trainer.model.student = merged

    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    print(f"Merged model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    merge()
