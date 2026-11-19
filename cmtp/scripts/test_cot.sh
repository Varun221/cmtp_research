#!/bin/bash

cd /scratch/vy2142/cmtp_research/cmtp


## Realistic - MATH 

# Llama
CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/realistic_llama_lora_correct"
datasets=("/scratch/vy2142/cmtp_research/data_store/math/llama_test.json")
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
BATCH_SIZE=4

# Qwen
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/realistic_qwen_lora_correct"
# datasets=("/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json")
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# BATCH_SIZE=4


## Realistic GSM8k-Aug (chat-templated eval jsons from data_math/gsm_process.py)

# Llama
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/gsmrealistic_llama_10ep"
# datasets=("/scratch/vy2142/cmtp_research/data_store/gsmllm/llama_gsm8k-test.json" "/scratch/vy2142/cmtp_research/data_store/gsmllm/llama_gsm8k-hard.json" "/scratch/vy2142/cmtp_research/data_store/gsmllm/llama_multi-arith.json" "/scratch/vy2142/cmtp_research/data_store/gsmllm/llama_svamp.json")
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# BATCH_SIZE=16


## Semi-Natural - GSM8k-Aug-NL

# Llama
# full trained: /scratch/vy2142/cmtp_research_extra/ckpts/cot/seminatural_llama_full
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/seminatural_llama_lora"
# datasets=("gsm8k-test" "gsm8k-hard" "multi-arith" "svamp")
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# BATCH_SIZE=32


## Structured - GSM8k-Aug

# Llama
# # full trained: /scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_llama_full/
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_llama_lora"
# datasets=("gsm8k-test" "gsm8k-hard" "multi-arith" "svamp")
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# BATCH_SIZE=32


# Qwen
# # full trained: /scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_qwen_full/
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_qwen_lora"
# datasets=("gsm8k-test" "gsm8k-hard" "multi-arith" "svamp")
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# BATCH_SIZE=32



run_experiment() {
    python test_cot.py \
        --data_name "$1" \
        --output_dir "$SAVE_DIR" \
        --model_name_or_path $MODEL_NAME \
        --lora_r 128 --lora_alpha 32 --lora_init \
        --seed 11 \
        --max_token_num 2048 \
        --bf16 \
        --batch_size $BATCH_SIZE \
        --greedy False \
        --inf_num_iterations 5 \
        --ckpt_dir "$CKPT"
}



for dataset in "${datasets[@]}"; do
    run_experiment "$dataset"
done
