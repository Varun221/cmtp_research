#!/bin/bash

cd /scratch/vy2142/cmtp_research/cmtp

SAVE_DIR=/scratch/vy2142/cmtp_research/tempexps/

mkdir -p "$SAVE_DIR"

export TOKENIZERS_PARALLELISM=false

EXPNAME=spanlength4_lora_mtp_noinit_multitask


# Add W&B configuration
export WANDB_WATCH="gradients"
export WANDB_ENTITY=""
export WANDB_MODE="online"
export WANDB_PROJECT=""

L1B_COT_CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/structured_llama_full/pytorch_model.bin"
L1B_COTNL_CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/seminatural_llama_full/pytorch_model.bin"
Qwen_COT_CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/structured_qwen_full/pytorch_model.bin"

# Models
# Qwen/Qwen2.5-1.5B-Instruct
# meta-llama/Llama-3.2-1B-Instruct

multi_task_args=(
    --init_student_with_teacher False
    --no_init_teacher True
    --multitask_training True
    --multitask_loss_factor 0.5
)

warmstart_args=(
    --textcot_teacher_ckpt "$L1B_COT_CKPT"
    --init_student_with_teacher True
    --multitask_training False
)

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXPNAME" \
    --run_name "$EXPNAME" \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--data_name icot \
    --bf16 \
	--seed 221 \
	--per_device_train_batch_size 32 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 8 \
	--learning_rate 5e-4 \
	--max_grad_norm 2.0 \
	--save_strategy "epoch" \
	--save_total_limit 5 \
	--save_safetensors False \
	--weight_decay 0.01 \
	--warmup_ratio 0.05 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to "wandb" \
	--logging_strategy "steps" \
	--distill_loss_div_std True \
	--exp_mode False \
	--max_token_num 300 \
    --eval_strategy "steps" \
	--eval_steps 500 \
	--per_device_eval_batch_size 64 \
	--do_eval \
    --span_length 4 \
    --attn_impl "flash_attention_2" \
    "${warmstart_args[@]}"
