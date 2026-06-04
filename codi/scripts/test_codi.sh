#!/bin/bash

cd /scratch/vy2142/cmtp_research/codi

NUM_TIMES=1


## Realistic - MATH

# Llama - codi6
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_llamacodi6"
# INF_LATENT=6
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/llama_test.json"
# BATCH_SIZE=16

# Llama - codi10
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_llamacodi10"
# INF_LATENT=10
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/llama_test.json"
# BATCH_SIZE=16

# Llama - codi20
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_llamacodi20"
# INF_LATENT=20
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/llama_test.json"
# BATCH_SIZE=16

# Qwen - codi6
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_qwencodi6"
# INF_LATENT=6
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json"
# BATCH_SIZE=16

# Qwen - codi10
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_qwencodi10"
# INF_LATENT=10
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json"
# BATCH_SIZE=16

# Qwen - codi20
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/codi/realistic_qwencodi20"
# INF_LATENT=20
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="/scratch/vy2142/cmtp_research/data_store/math/qwen_test.json"
# BATCH_SIZE=16

## Semi-Natural - GSM8k-Aug-NL

# Llama - codi6
# CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/codi/semi_natural_llamacodi6"
# INF_LATENT=6
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32

## Structured - GSM8k-Aug

# Llama - codi6
# CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/codi/structured_llamacodi6"
# INF_LATENT=6
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32

# Qwen - codi6
# CKPT="/scratch/vy2142/cmtp_research/hf_upload/ckpts/codi/structured_qwencodi6"
# INF_LATENT=6
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32




######## Sim-CoT trained checkpoints
# During training we use a decoder to supervise intermediate latents, but during inference
# we discard the decoder leading to an identical checkpoint as codi.

# Llama - structured6
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/simcot/structured_llama6"
# INF_LATENT=6
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32

# Llama - seminatural6
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/simcot/seminatural_llama6"
# INF_LATENT=6
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32

# Qwen - structured6
# CKPT="/scratch/vy2142/cmtp_research_extra/ckpts/simcot/structured_qwen6"
# INF_LATENT=6
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# EVLIST="gsm8k-test,gsm8k-hard,multi-arith,svamp"
# BATCH_SIZE=32


python test.py \
    --ckpt_dir "$CKPT" \
    --model_name_or_path "$MODEL_NAME" \
    --lora_r 128 --lora_alpha 32 --lora_init \
    --inf_latent_iterations "$INF_LATENT" \
    --greedy False \
    --use_prj True \
    --inf_num_iterations "$NUM_TIMES" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$CKPT" \
    --data_name "$EVLIST"
