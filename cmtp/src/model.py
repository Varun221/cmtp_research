"""Implementation of C-MTP training with testing."""

import transformers
from transformers import AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model
from safetensors.torch import load_file
from typing import Optional

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
    freeze_emb: bool = field(
        default=False, metadata={"help": "Whether to freeze the embedding layer."}
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    raw_data_path: str = field(default=None, metadata={"help": "Path to the raw data."})
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
        default=None, metadata={"help": "Path to the eval data override."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
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
    eval_fraction: float = field(
        default=0.2, metadata={"help": "Fraction of data used for evaluation."}
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    greedy: bool = field(
        default=False, metadata={"help": "Greedy decoding during inference."}
    )
    sample_mtp: bool = field(
        default=False, metadata={"help": "Sample tokens for MTP during inference."}
    )
    exp_mode: bool = field(
        default=False,
        metadata={
            "help": "Experimental mode. Use partial number of data for debugging."
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
    prj_no_ln: bool = field(
        default=False,
        metadata={"help": "Remove the Layer Norm layer for the projection module."},
    )
    inf_num_iterations: int = field(
        default=1, metadata={"help": "Number of iterations during inference."}
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
    include_last_cot: bool = field(
        default=True,
        metadata={"help": "Include the last CoT step in the training data."},
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
    text_cot_training: bool = field(
        default=False,
        metadata={"help": "Do simple text-based CoT training."},
    )

    span_length: int = field(
        default=2, metadata={"help": "The length of the masked span."}
    )
    textcot_teacher_ckpt: Optional[str] = field(
        default="",
        metadata={
            "help": "Path to the teacher model checkpoint for text-based CoT training."
        },
    )
    init_student_with_teacher: bool = field(
        default=False,
        metadata={
            "help": "Initialize the student model with the teacher model weights"
        },
    )
    no_init_teacher: bool = field(
        default=False, metadata={"help": "Whether to initialize the teacher model."}
    )

    compile_forward: bool = field(
        default=True,
        metadata={"help": "Whether to compile the forward pass."},
    )

    multitask_training: bool = field(
        default=False,
        metadata={"help": "Whether to train student to perform teacher task as well."},
    )
    multitask_loss_factor: float = field(
        default=0.5, metadata={"help": "Loss factor for multitask training."}
    )




def print_trainable_parameters(model):
    trainable_parameters = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()
    print(
        f"trainable params: {trainable_parameters} || "
        f"all params: {all_param} || "
        f"trainable%: {100 * trainable_parameters / all_param:.4f}"
    )
    for name, param in model.named_parameters():
        print(name, param.shape, param.requires_grad)


def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes mean of tensor over all positions where mask is True."""
    # ensure same dtype for multiplication
    mask = mask.to(tensor.dtype)
    masked_sum = (tensor * mask).sum()
    count = mask.sum().clamp(min=1)
    return masked_sum / count


def masked_std(tensor, mask, unbiased=True, eps=1e-5):
    # Ensure mask is same dtype as tensor
    mask = mask.to(tensor.dtype)

    # DYNAMO FIX: Handle broadcasting explicitly.
    # If tensor is [B, T, D] and mask is [B, T], unsqueeze to [B, T, 1]
    if tensor.dim() == 3 and mask.dim() == 2:
        mask = mask.unsqueeze(-1)

    # Calculate masked mean
    count = mask.sum()
    # sum() across all dims results in a scalar; safe for division
    mean = (tensor * mask).sum() / (count + eps)

    # Calculate variance
    # (tensor - mean) works, but ensuring mask is broadcasted
    # prevents the dimension mismatch s66 vs s18
    squared_diff = ((tensor - mean) ** 2) * mask
    variance_sum = squared_diff.sum()

    if unbiased:
        denom = torch.clamp(count - 1, min=eps)
    else:
        denom = torch.clamp(count, min=eps)

    variance = variance_sum / denom
    return torch.sqrt(variance + eps)


def cast_past(past, dtype):
    # See https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

    for layer in past.layers:
        layer.keys = layer.keys.to(dtype)
        layer.values = layer.values.to(dtype)

    return past


def aggregate_segments_single(
    acts: torch.Tensor, comp_segment_ids: torch.Tensor, reduceop="mean"
):
    """
    Aggregates a single acts (T, D) acts using comp_segment_ids (T,)
    using scatter and gather operations.

    Computes the mean of the compression segments specified by comp_segment_ids
    Places the reduction on all the positions of the segment.
    """

    # Ensure comp_segment_ids is same device and dtype
    comp_segment_ids = comp_segment_ids.to(acts.device)

    # Get symbolic shapes
    T, D = acts.size(0), acts.size(1)

    # Expand comp_segment_ids to (T, D) for scatter
    index_expand = comp_segment_ids.unsqueeze(-1).expand(-1, D)

    # Create segment_averages with symbolic shape
    segment_averages = torch.zeros_like(acts)

    # Scatter reduce
    segment_averages = segment_averages.scatter_reduce(
        dim=0,
        index=index_expand,
        src=acts,
        reduce=reduceop,
        include_self=False,
    )

    # Gather averages for each position in the segment
    out_vectors = torch.index_select(segment_averages, dim=0, index=comp_segment_ids)

    return out_vectors


class TrainingModel(torch.nn.Module):
    def __init__(self, model_args, training_args, lora_config):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.run_dtype = (
            torch.float16 if training_args.bf16 is False else torch.bfloat16
        )

        # breakpoint()
        model_wrapper_class = AutoModelForCausalLM
        self.student = model_wrapper_class.from_pretrained(
            self.model_name,
            torch_dtype=self.run_dtype,
            attn_implementation=training_args.attn_impl,
            # resume_download=True,
        )

        self.teacher = model_wrapper_class.from_pretrained(
            self.model_name,
            torch_dtype=(
                torch.float16 if training_args.bf16 is False else torch.bfloat16
            ),
            attn_implementation=training_args.attn_impl,
        )

        # breakpoint()

        ori_vocab_size = self.student.config.vocab_size
        self.training = self.model_args.train

        # special tokens to enclose the latent embeddings
        self.pad_token_id = ori_vocab_size
        self.bot_id = ori_vocab_size + 1
        self.mot_id = ori_vocab_size + 2
        self.eot_id = ori_vocab_size + 3

        # dummy values for extra tokens.
        self.student.resize_token_embeddings(ori_vocab_size + 4)
        self.teacher.resize_token_embeddings(ori_vocab_size + 4)

        self.dim = self.student.config.hidden_size

        # Load student with teacher weights if specified.
        if self.training_args.init_student_with_teacher:
            assert self.training_args.textcot_teacher_ckpt.lower() not in [
                "none",
                "null",
                "",
            ]
            teacher_wts = torch.load(
                self.training_args.textcot_teacher_ckpt, map_location="cpu"
            )

            # remove "student."
            new_weights = {}
            for k in list(teacher_wts.keys()):
                if k.startswith("student."):
                    new_weights[k[len("student.") :]] = teacher_wts.pop(k)

            # Using the original model.
            if len(new_weights) == 0:
                new_weights = teacher_wts

            self.student.load_state_dict(new_weights)

        self.embeddings_tied = self.student.config.tie_word_embeddings
        print(f"embeddings tied: {self.embeddings_tied}")

        # LoRA
        if model_args.lora_init:
            self.student = get_peft_model(self.student, lora_config)

            if not model_args.freeze_emb:
                print("unfreezing embedding layer.")
                # keep embedding layer unfrozen to train new tokens.
                for name, param in self.student.named_parameters():
                    if "embed_tokens" in name:
                        param.requires_grad = True
                    # Also unfreeze lm_head if not tied, to train new tokens.
                    elif (not self.embeddings_tied) and ("lm_head" in name):
                        param.requires_grad = True

        # Projection Layer
        self.mtp_projection = nn.Sequential(
            nn.Linear(self.dim, self.dim, dtype=self.run_dtype),
            nn.GELU(),
            nn.Linear(
                self.dim, self.dim * training_args.span_length, dtype=self.run_dtype
            ),
        )

        # Cross Entropy Loss
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

        if self.training:
            self.init()

        if self.training_args.compile_forward:
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

        # Load and freeze teacher
        if not self.training_args.no_init_teacher:
            if (
                self.training_args.textcot_teacher_ckpt.lower()
                not in ["none", "null", ""]
            ) and not self.training_args.text_cot_training:
                teacher_wts = torch.load(
                    self.training_args.textcot_teacher_ckpt, map_location="cpu"
                )
                # remove "student."
                new_teacher_wts = {}
                for k in list(teacher_wts.keys()):
                    if k.startswith("student."):
                        new_teacher_wts[k[len("student.") :]] = teacher_wts.pop(k)

                if len(new_teacher_wts) == 0:
                    new_teacher_wts = teacher_wts

                self.teacher.load_state_dict(new_teacher_wts)

        freeze_model(self.teacher)
        self.teacher.eval()

        print_trainable_parameters(self)

    def forward(
        self,
        ref_input_ids: torch.LongTensor = None,
        ref_answer_position: Optional[torch.LongTensor] = None,
        ref_attention_mask: Optional[torch.LongTensor] = None,
        loss_masks: Optional[torch.LongTensor] = None,
        span_segment_ids: Optional[torch.LongTensor] = None,
        step: int = None,
        step_ratio: float = None,
    ):
        """
        ref_input_ids: (B, T)
        ref_attention_mask: (B, T)
        loss_masks: (B, T)
        ref_answer_position: (B,)
        span_segment_ids: (B, T)

        Span Segment ids contain the spans to be used for training.
        0: non-span
        each non-zero integer indicates a different span.
        """
        # breakpoint()

        ref_labels = ref_input_ids.masked_fill(loss_masks != 1, -100)

        # Get Teacher outputs and loss
        if (self.training_args.text_cot_training) or (
            self.training_args.multitask_training
        ):
            teacher_model = self.student
        else:
            teacher_model = self.teacher

        teacher_output = teacher_model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            output_hidden_states=True,
        )

        ref_logits = teacher_output.logits
        effective_ref_logits = ref_logits[:, :-1, :]
        effective_ref_logits = effective_ref_logits.reshape(-1, ref_logits.size(-1))
        ref_target_ids = ref_labels[:, 1:].reshape(-1)
        ref_ce_loss = self.loss_fct(effective_ref_logits, ref_target_ids)

        acc_ele = (ref_target_ids != -100) & (
            effective_ref_logits.argmax(dim=-1) == ref_target_ids
        )
        acc = acc_ele.float().mean()

        # breakpoint()

        if self.training_args.text_cot_training:
            # Just return the teacher CE loss for text-based CoT training.
            return {
                "loss": ref_ce_loss,
                "metrics": {
                    "teacher_xent": ref_ce_loss,
                    "teacher_acc": acc,
                },
            }

        if self.training_args.multitask_training:
            ref_ce_loss = ref_ce_loss * self.training_args.multitask_loss_factor
        else:
            ref_ce_loss = 0.0

        # Figure out span start points
        B, T = ref_input_ids.size(0), ref_input_ids.size(1)
        non_cot_pos = span_segment_ids == 0
        span_starts = torch.roll(span_segment_ids, shifts=1, dims=1) != span_segment_ids

        # Prepare student input embeddings.
        embeds = self.get_embd(self.student, self.model_name)(ref_input_ids)

        with torch.no_grad():
            embeds_agg = torch.func.vmap(aggregate_segments_single)(
                embeds.detach(), span_segment_ids
            )

        # For non-span positions, use the original embeddings. For span positions, use the aggregated embeddings.
        embeds_student_input = torch.where(
            non_cot_pos.unsqueeze(-1), embeds, embeds_agg
        )

        # Position ids
        valid_token_positions = (
            torch.logical_or(span_starts, non_cot_pos) & ref_attention_mask.bool()
        )
        position_ids = torch.cumsum(valid_token_positions, dim=1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids.masked_fill(~valid_token_positions, 0)

        # Pass into Student model.
        student_outputs = self.student(
            inputs_embeds=embeds_student_input,
            attention_mask=valid_token_positions,
            position_ids=position_ids,
            output_hidden_states=True,
        )

        # LOSSES

        V = student_outputs.logits.size(-1)

        # Loss calculation.
        inp_tokens = ref_input_ids[:, :-1]  # (B, T)
        student_labels = ref_input_ids[:, 1:]  # (B, T)
        span_seg_labels = span_segment_ids[:, 1:]  # (B, T)
        span_seg_inp = span_segment_ids[:, :-1]  # (B, T)

        # DYNAMO FIX: Ensure contiguous tensors for indexing operations in the loss calculation.
        student_labels = student_labels.contiguous()

        # Create labels with unfold.
        unf = student_labels.unfold(1, self.training_args.span_length, 1)
        unf_pad = F.pad(
            unf, (0, 0, 0, student_labels.size(1) - unf.size(1)), "constant", 0
        )
        # +1 as we shift n-1 positions for n span length
        # each span-start predicts the *next* span's tokens.
        unf_pad_rolled = torch.roll(unf_pad, -self.training_args.span_length + 1, 1)
        # Use these only for span inputs.
        pred_labels = torch.where(
            span_seg_inp.unsqueeze(-1) > 0, unf_pad_rolled, unf_pad
        )
        # (B, T, span_length)

        # Separate MTP projections
        mtp_projection_out = self.mtp_projection(student_outputs.hidden_states[-1])
        # (B, T, dim * span_length)
        mtp_projection_out = mtp_projection_out.view(
            B, T, self.training_args.span_length, self.dim
        )
        # (B, T, span_length, dim)

        # Logits
        if self.embeddings_tied:
            student_logits = (
                mtp_projection_out
                @ self.get_embd(self.student, self.model_name).weight.t()
            )
        else:
            student_logits = mtp_projection_out @ self.student.lm_head.weight.t()
        # (B, T, span_length, V)

        student_logits = student_logits[:, :-1, ...].contiguous()

        # mtp_prediction_mask positions to calculate mtp loss.
        # bot -> span
        # span start -> next span
        # last span -> eot token at start.
        init_pred_mask = (
            (span_seg_labels > 0) & valid_token_positions[:, :-1] & loss_masks[:, 1:]
        )

        # Last span only needs to predict eot correctly.
        span_pred_mask = init_pred_mask.unsqueeze(-1).expand(
            -1, -1, self.training_args.span_length
        )
        tmp = span_pred_mask.all(dim=-1) * torch.arange(
            span_pred_mask.size(1), device=span_pred_mask.device
        )
        ind = tmp.max(dim=1).values
        mtp_prediction_mask = span_pred_mask.clone()
        mtp_prediction_mask[torch.arange(B, device=span_pred_mask.device), ind, 1:] = 0
        # B, T, span_length

        mtp_loss_ptoken = F.cross_entropy(
            student_logits.view(-1, V), pred_labels.view(-1), reduction="none"
        )
        mtp_loss_ptoken = mtp_loss_ptoken.view_as(pred_labels)
        mtp_loss_span = masked_mean(mtp_loss_ptoken, mtp_prediction_mask)

        # Normal token loss with the first projection.
        student_logits_first = student_logits[:, :, 0, :]
        # (B, T, V)
        ce_loss_ptoken = F.cross_entropy(
            student_logits_first.view(-1, V),
            student_labels.view(-1),
            reduction="none",
        )
        ce_loss_ptoken = ce_loss_ptoken.view_as(student_labels)

        # text_prediction_mask:
        # last question token -> bot
        # eot -> first answer token
        # answer tokens.
        text_prediction_mask = (
            (span_seg_labels == 0) & valid_token_positions[:, :-1] & loss_masks[:, 1:]
        )
        ce_loss = masked_mean(ce_loss_ptoken, text_prediction_mask)
        mtp_loss = mtp_loss_span + ce_loss

        # TESTING BREAKPOINT - See below for explanation of variables.
        # breakpoint()

        #### METRICS

        with torch.no_grad():

            span_pred_mask = (
                (span_seg_labels > 0)
                & valid_token_positions[:, :-1]
                & loss_masks[:, 1:]
            )
            text_pred_mask = (
                (span_seg_labels == 0)
                & valid_token_positions[:, :-1]
                & loss_masks[:, 1:]
            )

            # MTP metrics
            pred_indices = torch.argmax(student_logits, dim=-1)
            # (B, T, span_length)
            pred_is_hit = (pred_indices == pred_labels).float()

            acc_span_metrics = {}
            for dim in range(self.training_args.span_length):
                acc_span_metrics[f"acc_span_{dim}"] = masked_mean(
                    pred_is_hit[:, :, dim], mtp_prediction_mask[:, :, dim]
                )
            acc_span_metrics["acc_span_overall"] = masked_mean(
                pred_is_hit, mtp_prediction_mask
            )
            span_loss = mtp_loss_span

            # Text metrics
            pred_indices_text = torch.argmax(student_logits_first, dim=-1)
            pred_is_hit_text = pred_indices_text == student_labels
            text_acc = masked_mean(pred_is_hit_text.float(), text_pred_mask)
            text_loss = ce_loss

            total_acc = (
                text_acc * text_pred_mask.float().mean()
                + acc_span_metrics["acc_span_overall"]
                * mtp_prediction_mask.float().mean()
            )

        # Distillation Loss -- same as CODI.
        distill_loss = 0.0

        for j, (out, ref_out) in enumerate(
            zip(student_outputs.hidden_states, teacher_output.hidden_states)
        ):
            ref_selected = ref_out.gather(
                1,
                ref_answer_position.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, ref_out.size(-1)),
            )
            out_selected = out.gather(
                1,
                ref_answer_position.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, -1, out.size(-1)),
            )

            distill_loss_tmp = self.distill_loss_fct(
                out_selected, ref_selected.detach()
            )

            if self.distill_loss_div_std:
                distill_loss_tmp /= ref_selected.std()
            distill_loss += distill_loss_tmp

        distill_loss /= len(student_outputs.hidden_states)

        distill_loss *= self.distill_loss_factor
        loss = distill_loss + mtp_loss + ref_ce_loss
        # breakpoint()

        metrics = {
            "total_loss": loss,
            "teacher_xent": ref_ce_loss,
            "distill_loss": distill_loss,
            "student_xent": mtp_loss,
            "student_xent_text": text_loss,
            "student_xent_span": span_loss,
            "acc_text": text_acc,
            "acc_total": total_acc,
            **acc_span_metrics,
        }

        return {
            "loss": loss,
            "metrics": metrics,
        }


if __name__ == "__main__":

    # Test implementation correctness with custom inputs.

    from peft import LoraConfig, TaskType

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Test model functioning.
    # See utils.py for instructions on creating smallama
    model_args = ModelArguments(
        model_name_or_path="/scratch/vy2142/researchcodi/model_store/smallama",
        lora_r=128,
        lora_alpha=32,
        lora_init=True,
    )
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )
    training_args = TrainingArguments(
        bf16=True if torch.cuda.is_available() else False,
        output_dir="./tmp",
        textcot_teacher_ckpt="",
        compile_forward=False,
        # NotImplementedError: "smooth_l1_backward_cpu_out" not implemented for 'Half'
        distill_loss_type="l2",
        span_length=4,
    )
    model = TrainingModel(model_args, training_args, lora_config)

    model.to(device_str)

    # fmt: off

    # Test Input
    ref_input_ids = torch.tensor(
        [[5, 6, 7, 8, 9, # - 4
          model.bot_id, # - 5
          56, 58, 59, 60, # - 9
          61, 62, 63, 68, # - 13
          69, 70, 71, 72, # - 17
          73, 74, 75, 76, # - 21
          77, 78, model.mot_id, model.mot_id, # - 25
          model.eot_id, # - 26
          10, 11, 12, 1000 # - 30
          ]]
    )
    ref_answer_position = torch.tensor([28,])
    ref_attention_mask = torch.ones_like(ref_input_ids)
    loss_masks = torch.tensor([[0] * 5 + [1] * 26])
    span_segment_ids = torch.tensor(
        [[0] * 6 + [1, 1, 1, 1, 
                    2, 2, 2, 2, 
                    3, 3, 3, 3,
                    4, 4, 4, 4, 
                    5, 5, 5, 5] + [0] * 5]
    )

    # Verification for pred_labels
    # Test at breakpoint.

    # Test output - verify.
    """

    ### MTP Labels: pred_labels
    position i holds the span of tokens to predict from i
    
assert pred_labels[0, 5].squeeze().tolist() == [56, 58, 59, 60]
assert pred_labels[0, 6].squeeze().tolist() == [61, 62, 63, 68]
assert pred_labels[0, 10].squeeze().tolist() == [69, 70, 71, 72]
assert pred_labels[0, 14].squeeze().tolist() == [73, 74, 75, 76]
assert pred_labels[0, 18].squeeze().tolist() == [77, 78, model.mot_id, model.mot_id]
assert pred_labels[0, 22][0].item() == model.eot_id



    ### MTP Prediction Mask: mtp_prediction_mask

    only span-start positions are active
assert mtp_prediction_mask[0].any(dim=-1).nonzero().squeeze().tolist() == [5, 6, 10, 14, 18, 22]

    interiors of spans have no loss, such as span 1 and span 2
assert not mtp_prediction_mask[0, 7].any()
assert not mtp_prediction_mask[0, 11].any()

    bot_id (token 5) predicts 4 spans, so all positions
assert mtp_prediction_mask[0, 5].all()
    
    tokens 6, 10, 14, 18 are span starts, so all their 4 slots active
assert mtp_prediction_mask[0, 6].all()
assert mtp_prediction_mask[0, 10].all()
assert mtp_prediction_mask[0, 14].all()
assert mtp_prediction_mask[0, 18].all()

    for span 5 i.e. token 22 only first slot is true as we only eot_id
assert mtp_prediction_mask[0, 22].squeeze().tolist() == [True, False, False, False]



    ### Student Labels: student_labels

    standard NTP shift — position i predicts token i+1.
    last question predictions bot, eot predicts first answer token, subsequent answer tokens predict next token.
assert student_labels[0, 4].item() == model.bot_id
assert student_labels[0, 26].item() == 10
assert student_labels[0, 27].item() == 11
assert student_labels[0, 28].item() == 12
assert student_labels[0, 29].item() == 1000



    ### Student Prediction Mask: text_prediction_mask

    exclude question tokens and span tokens as they are covered by MTP.
assert not text_prediction_mask[0, :4].any()
assert not text_prediction_mask[0, 5:25].any()

    only active positions are last question token to bot, eot to answer, then answer NTP.
assert text_prediction_mask[0].nonzero().squeeze().tolist() == [4, 26, 27, 28, 29]
assert text_prediction_mask[0, 4]   # last question token (9) -> BOT
assert text_prediction_mask[0, 26]  # eot_id → first answer token
assert text_prediction_mask[0, 27]  # 10 -> 11
assert text_prediction_mask[0, 28]  # 11 -> 12
assert text_prediction_mask[0, 29]  # 12 -> 1000

    """

    # fmt: on

    print(
        ref_input_ids.shape,
        ref_attention_mask.shape,
        loss_masks.shape,
        span_segment_ids.shape,
    )

    outputs = model(
        ref_input_ids=ref_input_ids.to(device_str),
        ref_answer_position=ref_answer_position.to(device_str),
        ref_attention_mask=ref_attention_mask.to(device_str),
        loss_masks=loss_masks.to(device_str),
        span_segment_ids=span_segment_ids.to(device_str),
    )
    print(outputs)
    outputs["loss"].backward()
