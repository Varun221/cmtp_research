#!/bin/bash

cd /scratch/vy2142/cmtp_research/cmtp


NUM_TIMES=1


## Realistic - MATH

# Llama
CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/realistic_llama_sp2mtp"
SPAN_LENGTH=2
MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
EVLIST="/scratch/vy2142/cmtp_research/data_store/math/llama_test.json"
MAX_TOKENS=2080
BATCH_SIZE=1

# Qwen
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/realistic_qwen_sp2mtp"
# SPAN_LENGTH=2
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json"
# MAX_TOKENS=2080
# BATCH_SIZE=1


## Semi-Natural - GSM8k-Aug-NL

# Llama - span 2
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp2mtp"
# SPAN_LENGTH=2
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Llama - span 3
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp3mtp"
# SPAN_LENGTH=3
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Llama - span 4
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/seminatural_llama_sp4mtp"
# SPAN_LENGTH=4
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32




## Structured - GSM8k-Aug

# Llama - span 2
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp2mtp"
# SPAN_LENGTH=2
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Llama - span 3
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp3mtp"
# SPAN_LENGTH=3
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Llama - span 4
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp4mtp"
# SPAN_LENGTH=4
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32



# Qwen - span 2
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_qwen_sp2mtp"
# SPAN_LENGTH=2
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Qwen - span 3
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_qwen_sp3mtp"
# SPAN_LENGTH=3
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32

# Qwen - span 4
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_qwen_sp4mtp"
# SPAN_LENGTH=4
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# MAX_TOKENS=200
# BATCH_SIZE=32



python test_mtp.py \
    --ckpt_dir "$CKPT" \
    --model_name_or_path "$MODEL_NAME" \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --span_length "$SPAN_LENGTH" \
    --max_token_num "$MAX_TOKENS" \
    --greedy True \
    --sample_mtp False \
    --inf_num_iterations "$NUM_TIMES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$CKPT" \
    --evlist "$EVLIST"