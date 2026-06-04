"""Main training script. See scripts/ for launching examples."""

import os
import gc
import torch
import transformers
from transformers import Trainer
from math import ceil
from peft import LoraConfig, TaskType
from tqdm.auto import tqdm
from src.model import (
    TrainingModel,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from torch.amp import autocast

from src.dataset import create_data_module
from src import utils

# Compile settings.
torch._dynamo.config.capture_scalar_outputs = True


device_str = "cuda" if torch.cuda.is_available() else "cpu"


def _to_scalar(x):
    """Convert Tensor/number/None to python float (mean-reduced if needed)."""
    import torch

    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        # Detach, convert to float, take the mean if there are multiple elements, then call `item()`.
        return x.detach().float().mean().item()
    # It's already a number.
    return float(x)


IGNORE_INDEX = -100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Save code snapshot, assume train.py at root.
        project_dir = os.path.dirname(os.path.abspath(__file__))
        with open(
            os.path.join(self.args.output_dir, "codesnapshot.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(utils.get_encoded_files(project_dir))

    def compute_loss(self, model, inputs, num_items_in_batch):
        # Extract the global step from the optimizer
        step = self.state.global_step

        # Get total training steps
        batch_size = self.args.per_device_train_batch_size
        gradient_accumulation_steps = self.args.gradient_accumulation_steps
        num_epochs = self.args.num_train_epochs
        dataset_size = len(self.train_dataset)

        effective_batch_size = (
            batch_size * self.args.world_size * gradient_accumulation_steps
        )
        total_steps = ceil(dataset_size / effective_batch_size) * num_epochs

        # Add the step information to the inputs dictionary
        inputs["step_ratio"] = step / total_steps
        inputs["step"] = step

        with autocast(dtype=torch.bfloat16, device_type=device_str):
            outputs = model(**inputs)

        loss = outputs["loss"]
        if step % self.args.logging_steps == 0:
            logs = {
                **{
                    k: _to_scalar(v)
                    for k, v in outputs["metrics"].items()
                    if k != "loss"
                },
            }
            if not hasattr(self, "is_global_zero") or self.is_global_zero:
                self.log(logs)

        return loss

    def log(self, logs, start_time=None):
        if self.state.global_step is not None:
            super().log(logs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model
        model.eval()

        accumulated_metrics = {}
        num_batches = 0

        for inputs in tqdm(eval_dataloader, desc="Evaluating"):
            inputs = self._prepare_inputs(inputs)
            inputs["step"] = self.state.global_step
            inputs["step_ratio"] = 1.0

            with torch.no_grad():
                with autocast(dtype=torch.bfloat16, device_type=device_str):
                    outputs = model(**inputs)

            for k, v in outputs["metrics"].items():
                if isinstance(v, torch.Tensor):
                    v_detached = v.detach()
                    if k not in accumulated_metrics:
                        accumulated_metrics[k] = v_detached
                    else:
                        accumulated_metrics[k] += v_detached
            num_batches += 1

        # gpu-cpu sync point
        final_metrics = {}
        for k, v in accumulated_metrics.items():
            # calc average
            avg_value = (v / max(num_batches, 1)).item()
            final_metrics[f"{metric_key_prefix}_{k}"] = avg_value

        super().log(final_metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, final_metrics
        )

        model.train()

        del outputs
        del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_metrics


def train():
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

    # breakpoint()
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

    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        f"job_{utils.current_time()}",
    )
    training_args.remove_unused_columns = False

    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    # assert 0, 'run when confident'
    if training_args.resume_training_checkpoint is not None:
        print(
            f"Resuming training from checkpoint: {training_args.resume_training_checkpoint}"
        )
        trainer.train(resume_from_checkpoint=training_args.resume_training_checkpoint)
    else:
        trainer.train()

    trainer.evaluate()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
