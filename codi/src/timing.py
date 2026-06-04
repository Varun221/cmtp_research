"""Time the forward pass of TrainingModel for different sequence lengths."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
import numpy as np
from peft import LoraConfig, TaskType
from src.model import TrainingModel, ModelArguments, TrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Args from training.sh ----------
model_args = ModelArguments(
    # model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
    model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    lora_r=128,
    lora_alpha=32,
    lora_init=True,
    full_precision=True,
    train=False,  # skip init (no checkpoint to restore)
)

training_args = TrainingArguments(
    output_dir="/tmp/time_model",
    bf16=True,
    num_latent=6,
    use_prj=True,
    prj_dim=2048,
    prj_dropout=0.0,
    distill_loss_div_std=True,
    exp_mode=False,
    distill_loss_factor=20,
    max_token_num=200,
    attn_impl="flash_attention_2",
    compile_forward=False,
    unfreeze_emb=True,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=0.1,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
    ],
    init_lora_weights=True,
)

# ---------- Build model ----------
print("Loading model...")
model = TrainingModel(model_args, training_args, lora_config)
model = model.cuda().train()

# ---------- Timing ----------
BATCH_SIZE = 32
SEQ_LENGTHS = [
    180,
]
NUM_WARMUP = 5
NUM_TRIALS = 20

vocab_size = model.codi.config.vocab_size

# Rough layout: 60% question, 30% cot, 10% answer
for seq_len in SEQ_LENGTHS:
    q_len = int(seq_len * 0.6)
    a_len = int(seq_len * 0.1)
    ref_len = seq_len  # full sequence for teacher (question + cot + answer)

    # Dummy tensors
    encoder_input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, q_len), device="cuda")
    decoder_input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, a_len), device="cuda")
    ref_input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, ref_len), device="cuda")
    labels = torch.randint(0, vocab_size, (BATCH_SIZE, a_len), device="cuda")
    ref_labels = torch.randint(0, vocab_size, (BATCH_SIZE, ref_len), device="cuda")
    encoder_attention_mask = torch.ones(
        BATCH_SIZE, q_len, dtype=torch.long, device="cuda"
    )
    ref_attention_mask = torch.ones(
        BATCH_SIZE, ref_len, dtype=torch.long, device="cuda"
    )
    ref_answer_position = torch.full(
        (BATCH_SIZE,), q_len + int(seq_len * 0.3), dtype=torch.long, device="cuda"
    )  # answer starts after question+cot in ref
    model_answer_position = torch.zeros(
        BATCH_SIZE, dtype=torch.long, device="cuda"
    )  # first position in decoder

    # Warmup
    print(f"\nSeq length {seq_len}: warming up ({NUM_WARMUP} iters)...")
    for _ in range(NUM_WARMUP):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_attention_mask,
                ref_attention_mask=ref_attention_mask,
                ref_labels=ref_labels,
                ref_answer_position=ref_answer_position,
                model_answer_position=model_answer_position,
            )
            output["loss"].backward()
        model.zero_grad()
        torch.cuda.synchronize()

    # Timed runs
    times = []
    print(f"Seq length {seq_len}: timing ({NUM_TRIALS} iters)...")
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_ids,
                ref_input_ids=ref_input_ids,
                labels=labels,
                encoder_attention_mask=encoder_attention_mask,
                ref_attention_mask=ref_attention_mask,
                ref_labels=ref_labels,
                ref_answer_position=ref_answer_position,
                model_answer_position=model_answer_position,
            )
            output["loss"].backward()
        model.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000  # ms
    std = np.std(times) * 1000
    print(f"  => {avg:.2f} +/- {std:.2f} ms")

print("\nDone.")
