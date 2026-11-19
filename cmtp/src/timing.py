"""Time the forward+backward pass of TrainingModel for different sequence lengths.

BATCH_SIZE and SEQ_LENGTHS are set at the top of the timing loop. codi_ms below
comes from codi/src/timing.py run with the same settings; speedup = codi_ms / cmtp_ms.
Measured on one H200 with Llama-3.2-1B-Instruct, SPAN_LENGTH=4 vs 6 CODI latents,
mean over 20 trials after 5 warmup iterations.

Output:
     batch  seq_len    cmtp_ms    codi_ms   speedup
        16       32     161.30     639.81      3.97
        16       64     179.92     633.37      3.52
        16       96     199.02     637.82      3.20
        16      128     225.62     635.21      2.82
        -------------------------------------------
        32       32     175.18     610.00      3.48
        32       64     224.79     615.10      2.74
        32       96     290.19     623.80      2.15
        32      128     381.28     661.60      1.74
        -------------------------------------------
        64       32     228.95     660.58      2.89
        64       64     386.14     705.59      1.83
        64       96     544.23     791.40      1.45
        64      128     722.30     880.25      1.22
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch

torch._dynamo.config.capture_scalar_outputs = True
import numpy as np
from peft import LoraConfig, TaskType
from src.model import TrainingModel, ModelArguments, TrainingArguments
from src.utils import equal_segmentation

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SPAN_LENGTH = 4

# Setup Args
model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
    # model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
    lora_r=128,
    lora_alpha=32,
    lora_init=True,
    train=False,
)

training_args = TrainingArguments(
    output_dir="/tmp/time_model",
    bf16=True,
    distill_loss_div_std=True,
    exp_mode=False,
    max_token_num=200,
    attn_impl="flash_attention_2",
    textcot_teacher_ckpt="",
    init_student_with_teacher=False,  # no checkpoint for timing
    span_length=SPAN_LENGTH,
    compile_forward=False,
    multitask_training=True,
    no_init_teacher=True
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


print("Loading model...")
model = TrainingModel(model_args, training_args, lora_config)
model = model.cuda().train()

vocab_size = model.student.config.vocab_size
bot_id = model.bot_id
eot_id = model.eot_id
mot_id = model.mot_id

# Config params. See paper for exact setup.
BATCH_SIZE = 16
SEQ_LENGTHS = [
    180,
]
NUM_WARMUP = 5
NUM_TRIALS = 20

# Rough layout: 60% question, 30% cot, 10% answer
for seq_len in SEQ_LENGTHS:
    q_len = int(seq_len * 0.6)
    a_len = int(seq_len * 0.1)
    cot_len = seq_len - q_len - a_len - 2  # -2 for bot/eot added by equal_segmentation

    # Build span_segment_ids using equal_segmentation
    dummy_cot = [
        list(range(cot_len))
    ]  # single-sample list of token ids (values don't matter)
    _, persample_spanids = equal_segmentation(
        cot_id=dummy_cot,
        bot_id=bot_id,
        eot_id=eot_id,
        mot_id=mot_id,
        span_length=SPAN_LENGTH,
    )
    # persample_spanids[0] includes [0] + span_ids + [0] for bot/eot
    span_ids = [0] * q_len + persample_spanids[0] + [0] * a_len
    actual_len = len(span_ids)

    # Dummy tensors
    ref_input_ids = torch.randint(
        0, vocab_size, (BATCH_SIZE, actual_len), device="cuda"
    )
    ref_attention_mask = torch.ones(
        BATCH_SIZE, actual_len, dtype=torch.long, device="cuda"
    )
    loss_masks = torch.zeros(BATCH_SIZE, actual_len, dtype=torch.long, device="cuda")
    loss_masks[:, q_len:] = 1  # loss on cot + answer
    span_segment_ids = (
        torch.tensor(span_ids, dtype=torch.long, device="cuda")
        .unsqueeze(0)
        .expand(BATCH_SIZE, -1)
    )
    ref_answer_position = torch.full(
        (BATCH_SIZE,),
        q_len + len(persample_spanids[0]),
        dtype=torch.long,
        device="cuda",
    )

    # Warmup
    print(
        f"\nSeq length {actual_len} (target {seq_len}): warming up ({NUM_WARMUP} iters)..."
    )
    for _ in range(NUM_WARMUP):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                loss_masks=loss_masks,
                span_segment_ids=span_segment_ids,
                ref_answer_position=ref_answer_position,
            )
            output["loss"].backward()
        model.zero_grad()
        torch.cuda.synchronize()

    # Timed runs
    times = []
    print(f"Seq length {actual_len}: timing ({NUM_TRIALS} iters)...")
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                ref_input_ids=ref_input_ids,
                ref_attention_mask=ref_attention_mask,
                loss_masks=loss_masks,
                span_segment_ids=span_segment_ids,
                ref_answer_position=ref_answer_position,
            )
            output["loss"].backward()
        model.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = np.mean(times) * 1000
    std = np.std(times) * 1000
    print(f"  => {avg:.2f} +/- {std:.2f} ms")

print("\nDone.")
