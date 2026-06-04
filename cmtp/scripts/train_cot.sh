#!/bin/bash

cd /scratch/vy2142/cmtp_research/cmtp

SAVE_DIR=/scratch/vy2142/cmtp_research/tempexps/

mkdir -p "$SAVE_DIR"

export TOKENIZERS_PARALLELISM=false

EXPNAME=cottraining_lora


# Add W&B configuration
export WANDB_WATCH="gradients"
export WANDB_ENTITY=""
export WANDB_MODE="online"
export WANDB_PROJECT=""

# model_name_or_path: 
# Qwen/Qwen2.5-1.5B-Instruct 
# meta-llama/Llama-3.2-1B-Instruct

# data_name: icot or icot-nl (See src/dataset.py for more options.)
# Increase max_token_num if switching to 
# icot-nl (longer texts) or Qwen (tokenizes to more tokens)

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
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 4 \
    --learning_rate 5e-4 \
    --max_grad_norm 2.0 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --save_safetensors False \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --do_train \
    --report_to "wandb" \
    --logging_strategy "steps" \
    --distill_loss_div_std True \
    --exp_mode False \
    --max_token_num 200 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --per_device_eval_batch_size 128 \
    --do_eval \
    --attn_impl "flash_attention_2" \
    --text_cot_training True