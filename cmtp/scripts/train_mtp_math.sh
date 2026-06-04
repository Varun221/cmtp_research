#!/bin/bash


cd /scratch/vy2142/cmtp_research/cmtp

SAVE_DIR=/scratch/vy2142/cmtp_research/tempexps/

mkdir -p "$SAVE_DIR"

export TOKENIZERS_PARALLELISM=false

EXPNAME=l1bmath_sp2_mtp_lora_correct_8ep


# Add W&B configuration
export WANDB_WATCH="gradients"
export WANDB_ENTITY=""
export WANDB_MODE="online"
export WANDB_PROJECT=""


# Math Trained
L1BMATH_COT_CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_llama_lora_correct_merged/pytorch_model.bin"
QwenMATH_COT_CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_qwen_lora_correct_merged/pytorch_model.bin"

# Model: meta-llama/Llama-3.2-1B-Instruct 
# Use `mathllama` data_name
# Use L1BMATH_COT_CKPT for textcot_teacher_ckpt

# Model: Qwen/Qwen2.5-1.5B-Instruct 
# Use `mathqwen` data_name
# Use QwenMATH_COT_CKPT for textcot_teacher_ckpt

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXPNAME" \
    --run_name "$EXPNAME" \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--data_name mathllama \
    --bf16 \
	--seed 221 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 8 \
	--learning_rate 5e-4 \
	--max_grad_norm 2.0 \
	--save_strategy "epoch" \
	--save_total_limit 1 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.05 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to "wandb" \
	--logging_strategy "steps" \
	--distill_loss_div_std True \
	--exp_mode False \
	--max_token_num 1024 \
    --eval_strategy "steps" \
	--eval_steps 500 \
	--per_device_eval_batch_size 8 \
	--do_eval \
    --attn_impl "flash_attention_2" \
	--span_length 2 \
	--init_student_with_teacher True \
    --textcot_teacher_ckpt "$L1BMATH_COT_CKPT"


