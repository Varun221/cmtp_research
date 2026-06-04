import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model
from safetensors.torch import load_file

device_str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="mistralai/Mistral-7B-Instruct-v0.2")
    separate_decoder_name: str = field(default="")
    lora_r: int = field(default=128, metadata={"help": "lora rank"})
    lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
    train: bool = field(
        default=True,
        metadata={
            "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={
            "help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."
        },
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    ckpt_dir: Optional[str] = field(
        default=None, metadata={"help": "checkpoint dir for inference."}
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    debug_data: bool = field(
        default=False,
        metadata={
            "help": "Enable debug dataset to quickly verify the training process"
        },
    )
    batch_size: int = field(default=1, metadata={"help": "batch size during inference"})
    train_path_override: Optional[str] = field(
        default=None, metadata={"help": "Path to the training data override."}
    )
    eval_path_override: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data override."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=28000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    restore_from: str = field(
        default="",
        metadata={
            "help": "The checkpoint that should be restored from for fine-tuning"
        },
    )
    resume_training_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The checkpoint state to resume training from"}
    )
    per_device_train_batch_size: int = field(
        default=1,
    )
    per_device_eval_batch_size: int = field(
        default=1,
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    num_latent: int = field(
        default=5, metadata={"help": "The number of latent for training or inference."}
    )
    greedy: bool = field(
        default=False, metadata={"help": "Greedy decoding during inference."}
    )
    exp_mode: bool = field(
        default=False,
        metadata={
            "help": "Experimental model. Use partial number of data for debugging."
        },
    )
    exp_data_num: int = field(
        default=10000, metadata={"help": "The number of data used in exp mode"}
    )
    use_prj: bool = field(
        default=False,
        metadata={"help": "Use a prj module after the llm for latent generation."},
    )
    prj_dim: int = field(
        default=2048, metadata={"help": "The hidden dim of the projection module."}
    )
    prj_dropout: float = field(
        default=0.0, metadata={"help": "Dropout ratio of the projection module."}
    )
    distill_loss_div_std: bool = field(
        default=False,
        metadata={"help": "Divide the distillation loss by a std for normallisation."},
    )
    distill_loss_type: str = field(
        default="smooth_l1",
        metadata={"help": "Specify the distillation loss. Use smoothL1 by default."},
    )
    distill_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the distillation loss."}
    )
    explain_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the explain loss."}
    )
    ref_loss_factor: float = field(
        default=1.0, metadata={"help": "A multiplier of the distillation loss."}
    )
    inf_latent_iterations: int = field(default=1, metadata={"help": ""})
    inf_num_iterations: int = field(
        default=5, metadata={"help": "Run multiple times during inference"}
    )
    print_ref_model_stats: bool = field(
        default=False, metadata={"help": "Print some stats for the teacher task."}
    )
    include_last_cot: bool = field(
        default=False,
        metadata={"help": "Include the last CoT step in the training data."},
    )
    fix_attn_mask: bool = field(
        default=False, metadata={"help": "Correct a bug about attention mask."}
    )
    log_full: bool = field(default=False, metadata={"help": "Log all losses."})
    print_loss: bool = field(default=False)
    max_token_num: int = field(
        default=1000, metadata={"help": "Limit the longest data to avoid OOM."}
    )

    # Additions
    attn_impl: str = field(
        default="sdpa",
        metadata={"help": "sdpa (F.scaled_dot_product_attention) or flash_attention_2"},
    )
    compile_forward: bool = field(
        default=True,
        metadata={"help": "Whether to compile the forward pass."},
    )
    dynamo_disable_forward: bool = field(
        default=False,
        metadata={
            "help": "Disable torch._dynamo for the outer forward pass (use with flash_attn + unfreeze_emb)."
        },
    )

    # Setting otherwise leads to performance regression.
    add_thinktoken_in_ref: bool = field(
        default=False,
        metadata={"help": "Whether to add the end-of-think token in the reference."},
    )
    unfreeze_emb: bool = field(
        default=False,
        metadata={"help": "Whether to make the embedding layer trainable."},
    )


def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}"
    )
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)


def topk_mask(tensor: torch.Tensor, topk: int) -> torch.Tensor:
    """Return tensor with only the top-k values kept (others zeroed).
    Uses the kth-largest value as threshold; minor errors possible at ties."""
    topk_vals, _ = torch.topk(tensor, topk, dim=-1)
    threshold = topk_vals[..., -1:]
    return tensor * (tensor >= threshold).to(tensor.dtype)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes mean of tensor over all positions where mask is True."""
    # ensure same dtype for multiplication
    mask = mask.to(tensor.dtype)
    masked_sum = (tensor * mask).sum()
    count = mask.sum().clamp(min=1)
    return masked_sum / count


def cast_past(past, dtype):
    # See https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

    for layer in past.layers:
        layer.keys = layer.keys.to(dtype)
        layer.values = layer.values.to(dtype)

    return past


class TrainingModel(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.run_dtype = (
            torch.float16 if training_args.bf16 is False else torch.bfloat16
        )

        model_wrapper_class = AutoModelForCausalLM
        self.codi = model_wrapper_class.from_pretrained(
            self.model_name,
            torch_dtype=self.run_dtype,
            attn_implementation=training_args.attn_impl,
            resume_download=True,
        )

        ori_vocab_size = self.codi.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.eot_id = ori_vocab_size + 2

        self.codi.resize_token_embeddings(
            ori_vocab_size + 3
        )  # dummy values for mem tokens

        self.dim = self.codi.config.hidden_size
        self.num_latent = training_args.num_latent
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # LoRA
        if model_args.lora_init:
            self.codi = get_peft_model(self.codi, lora_config)

            if training_args.unfreeze_emb:
                print("unfreezing embedding layer.")
                # keep embedding layer unfrozen to train new tokens.
                for name, param in self.codi.named_parameters():
                    if "embed_tokens" in name:
                        param.requires_grad = True

        # Projection Layer
        self.use_prj = training_args.use_prj

        if self.use_prj:
            self.prj = nn.Sequential(
                nn.Dropout(training_args.prj_dropout),
                nn.Linear(self.dim, training_args.prj_dim, dtype=self.run_dtype),
                nn.GELU(),
                nn.Linear(training_args.prj_dim, self.dim, dtype=self.run_dtype),
                nn.LayerNorm(self.dim, dtype=self.run_dtype),
            )

        # Losses
        self.print_loss = training_args.print_loss
        self.ref_loss_factor = training_args.ref_loss_factor

        # Cross Entropy Loss
        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fct = partial(F.cross_entropy, ignore_index=-100)

        # Distillation Loss
        self.distill_loss_div_std = training_args.distill_loss_div_std
        self.distill_loss_type = training_args.distill_loss_type
        self.distill_loss_factor = training_args.distill_loss_factor
        if self.distill_loss_type == "smooth_l1":
            self.distill_loss_fct = nn.SmoothL1Loss()
        elif self.distill_loss_type == "l2":
            self.distill_loss_fct = nn.MSELoss()
        else:
            raise NotImplementedError

        # general
        self.fix_attn_mask = training_args.fix_attn_mask

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.tokenizer.pad_token_id = self.pad_token_id

        if self.training:
            self.init()

        if self.training_args.compile_forward:
            if self.training_args.dynamo_disable_forward:
                self.codi.forward = torch.compile(self.codi.forward, dynamic=True)
            else:
                self.forward = torch.compile(self.forward, dynamic=True)

    def get_embd(self, model, model_name):
        try:
            if "pythia" in model_name.lower():
                return model.get_base_model().gpt_neox.embed_in
            elif "gpt2" in model_name.lower():
                try:
                    return model.get_base_model().transformer.wte
                except Exception:  # no lora
                    return model.transformer.wte
            else:
                try:
                    return model.get_base_model().model.embed_tokens
                except Exception:  # no lora
                    return model.model.embed_tokens
        except AttributeError:
            if "pythia" in model_name:
                return model.gpt_neox.embed_in
            raise NotImplementedError

    def init(self):
        print_trainable_parameters(self)
        if (
            self.training_args.restore_from is not None
            and self.training_args.restore_from != ""
        ):
            print(
                f"Loading from the pretrained checkpoint: {self.training_args.restore_from}..."
            )
            state_dict = load_file(self.training_args.restore_from)
            self.load_state_dict(state_dict)
            print(f"Finished loading from {self.training_args.restore_from}")

    def forward(
        self,
        encoder_input_ids: torch.LongTensor = None,
        decoder_input_ids: torch.LongTensor = None,
        ref_input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        model_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        ref_labels: torch.LongTensor = None,
        step: int = None,
        step_ratio: float = None,
    ):
        if not self.fix_attn_mask:
            ref_attention_mask = None

        # Encode the question
        past_key_values = None
        outputs = self.codi(
            input_ids=encoder_input_ids,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=encoder_attention_mask,
        )
        past_key_values = outputs.past_key_values

        # Latent embedding for the next input
        # (64, 1, 2048)
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if self.use_prj:
            latent_embd = self.prj(latent_embd)

        dynamic_mask = None
        if self.fix_attn_mask:
            dynamic_mask = torch.ones(
                (encoder_attention_mask.size(0), self.num_latent),
                device=ref_labels.device,
            )

        # Iterate over the latent embeddings
        distill_loss_total = 0
        ce_loss_total = 0

        ref_outputs_with_grad = self.codi(
            input_ids=ref_input_ids,
            output_hidden_states=True,
            attention_mask=ref_attention_mask,
        )
        ref_outputs = [x.detach() for x in ref_outputs_with_grad.hidden_states]

        # Formatting for deprecated exps
        ref_outputs_list = [ref_outputs]
        ref_input_ids = [ref_input_ids]

        # Process the position tensor
        # Normalise the position definition
        if (
            "llama" in self.model_name.lower() or "qwen" in self.model_name.lower()
        ):  # there is one more token standing for " "
            model_answer_position = model_answer_position + 1
            ref_answer_position = ref_answer_position + 1

        # the model answer position is the position of the eot token to predict the first token of the response
        model_answer_position = model_answer_position - 1
        ref_answer_position = ref_answer_position - 1

        num_latent = self.num_latent
        if num_latent != 0:
            # Run first num_latent-1 latent steps
            for lidx in range(num_latent - 1):
                outputs = self.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values,
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.use_prj:
                    latent_embd = self.prj(latent_embd)

            # Final latent step
            outputs = self.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.use_prj:
                latent_embd = self.prj(latent_embd)

            # Decode the final answer in natural language
            embds = self.get_embd(self.codi, self.model_name)(decoder_input_ids)

            if dynamic_mask is not None:  # Prevent attending the paddings
                decoder_mask = torch.ones(
                    (embds.size(0), embds.size(1)), dtype=torch.bool
                ).to(dynamic_mask)
                dynamic_mask = torch.cat(
                    (encoder_attention_mask, dynamic_mask, decoder_mask), dim=1
                )
                dynamic_mask = dynamic_mask.bool()

            past_key_values = cast_past(past_key_values, self.run_dtype)
            outputs = self.codi(
                inputs_embeds=embds,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=dynamic_mask,
            )

            # Student task loss
            logits = outputs.logits
            effective_logits = logits[:, :-1, :]
            effective_logits = effective_logits.reshape(-1, logits.size(-1))
            target_ids = labels[:, 1:].reshape(-1)
            ce_loss = self.loss_fct(effective_logits, target_ids)
            ce_loss_total += ce_loss

            # Distillation loss

            ref_outputs = ref_outputs_list[0]

            distill_loss = 0
            for j, (out, ref_out) in enumerate(zip(outputs.hidden_states, ref_outputs)):
                ref_selected = ref_out.gather(
                    1,
                    ref_answer_position.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, ref_out.size(-1)),
                )
                out_selected = out.gather(
                    1,
                    model_answer_position.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(-1, -1, out.size(-1)),
                )

                distill_loss_tmp = self.distill_loss_fct(
                    out_selected, ref_selected.detach()
                )

                if self.distill_loss_div_std:
                    distill_loss_tmp /= ref_selected.std()
                distill_loss += distill_loss_tmp

            distill_loss /= len(outputs.hidden_states)

            if self.print_loss:
                print(
                    f"loss={ce_loss + distill_loss}, "
                    f"ce_loss={ce_loss}, "
                    f"distill_loss={distill_loss}, "
                    f"ce_loss_total={ce_loss_total}, "
                    f"distill_loss_total={distill_loss_total}, "
                    f"ref_ce_loss={ref_ce_loss}"
                )

            distill_loss_total += distill_loss

        # Calculate the CE loss for the teacher task
        ref_logits = ref_outputs_with_grad.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)
        ref_ce_loss *= self.ref_loss_factor

        # Weigh the distillation loss
        distill_loss *= self.distill_loss_factor
        distill_loss_total *= self.distill_loss_factor

        if self.print_loss:
            print(
                f"loss={ce_loss+distill_loss}, ce_loss={ce_loss}, distill_loss={distill_loss}, ce_loss_total={ce_loss_total}, distill_loss_total={distill_loss_total}, ref_ce_loss={ref_ce_loss}"
            )

        loss = ce_loss_total + distill_loss_total + ref_ce_loss

        return {
            "loss": loss,
            "metrics": {
                "ce_loss": ce_loss_total,
                "distill_loss": distill_loss_total,
                "ref_ce_loss": ref_ce_loss,
            },
        }
