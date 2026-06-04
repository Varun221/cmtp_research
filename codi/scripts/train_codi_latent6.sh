#!/bin/bash

cd /scratch/vy2142/cmtp_research/codi

SAVE_DIR=/scratch/vy2142/cmtp_research/tempexps/

mkdir -p "$SAVE_DIR"

export TOKENIZERS_PARALLELISM=false

export WANDB_WATCH="gradients"
export WANDB_ENTITY="vy2142-new-york-university"
export WANDB_MODE="online"
export WANDB_PROJECT="latentexps"

EXPNAME=l1b_codi6

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXPNAME" \
    --run_name "$EXPNAME" \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct  \
	--data_name icot \
	--seed 221 \
	--bf16 \
	--per_device_train_batch_size 64 \
	--gradient_accumulation_steps 2 \
	--num_train_epochs 8 \
	--learning_rate 7e-4 \
	--max_grad_norm 2.0 \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "no" \
	--save_total_limit 1 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
    --eval_strategy "steps" \
	--eval_steps 500 \
	--per_device_eval_batch_size 128 \
	--do_eval \
	--num_latent 6 \
	--logging_strategy "steps" \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--distill_loss_factor 20 \
	--max_token_num 200 \
	--attn_impl flash_attention_2