#!/bin/bash

cd /scratch/vy2142/cmtp_research/cmtp


# Merging Qwen-Math CoT

SAVE_DIR=/scratch/vy2142/cmtp_research/tempexps
EXPNAME=merging

python scripts/merge_lora.py \
	--output_dir "$SAVE_DIR" \
	--expt_name "$EXPNAME" \
    --run_name "$EXPNAME" \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--data_name mathqwen \
    --bf16 \
	--seed 221 \
	--per_device_train_batch_size 16 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 5 \
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
	--exp_mode True \
	--max_token_num 2048 \
    --eval_strategy "steps" \
	--eval_steps 500 \
	--per_device_eval_batch_size 32 \
	--do_eval \
    --attn_impl "flash_attention_2" \
	--text_cot_training True \
	--start_state "/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_qwen_lora_correct" \
	--output_dir "/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_qwen_lora_correct_merged"



# Merging Llama-Math CoT

python scripts/merge_lora.py \
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
	--per_device_train_batch_size 16 \
	--gradient_accumulation_steps 4 \
	--num_train_epochs 5 \
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
	--exp_mode True \
	--max_token_num 2048 \
    --eval_strategy "steps" \
	--eval_steps 500 \
	--per_device_eval_batch_size 32 \
	--do_eval \
    --attn_impl "flash_attention_2" \
	--text_cot_training True \
	--start_state "/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_llama_lora_correct" \
	--output_dir "/scratch/vy2142/cmtp_research/hf_upload/ckpts/cot/realistic_llama_lora_correct_merged"

