"""Compare single-inference latency of NTP vs MTP-2 vs MTP-3 decoding.

For a fixed sequence length we run one forward pass to obtain the final-token
activation, then form the next-token logits two ways:

  - NTP : one projection head (span_length=1) -> matmul with the vocab matrix.
  - MTP : one wider projection head (span_length=k) reshaped into k slots ->
          matmul with the vocab matrix, giving k tokens at once.

To keep the comparison fair, NTP also goes through the projection-then-matmul
head instead of the model's optimized lm_head. The point is to show the MTP
head adds negligible overhead on top of the (shared, dominant) forward pass.

Head logic mirrors `pred_multiple_tokens` / the forward in src/model.py.

Output (batch 32, seq_len 180, mean +- std over 20 trials):
    === Llama-3.2-1B (dim=2048, vocab=128256) ===
      method |   full step (ms) |   head only (ms)
         NTP |       34.02+-0.6 |       0.228+-0.0
       MTP-2 |       34.02+-0.2 |       0.247+-0.0
       MTP-3 |       34.10+-0.2 |       0.240+-0.0

    === Qwen2.5-1.5B (dim=1536, vocab=151936) ===
      method |   full step (ms) |   head only (ms)
         NTP |       44.35+-0.8 |       0.204+-0.0
       MTP-2 |       44.38+-0.8 |       0.221+-0.0
       MTP-3 |       44.50+-0.9 |       0.228+-0.0
"""

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

MODELS = {
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
}
SPAN_LENGTHS = [1, 2, 3]  # span_length 1 == NTP
SEQ_LEN = 180
BATCH_SIZE = 32
NUM_WARMUP = 5
NUM_TRIALS = 20
DTYPE = torch.bfloat16


def make_head(dim, span_length):
    """Same architecture as TrainingModel.mtp_projection."""
    return nn.Sequential(
        nn.Linear(dim, dim, dtype=DTYPE),
        nn.GELU(),
        nn.Linear(dim, dim * span_length, dtype=DTYPE),
    ).cuda()


def run_head(hidden, head, span_length, vocab_weight):
    """hidden: (B, dim) final activation -> (B, span_length) predicted tokens."""
    proj = head(hidden)  # (B, dim * span_length)
    proj = proj.view(proj.size(0), span_length, -1)  # (B, span_length, dim)
    logits = proj @ vocab_weight.t()  # (B, span_length, V)
    return torch.argmax(logits, dim=-1)


@torch.no_grad()
def time_span(model, vocab_weight, head, span_length, input_ids, attention_mask):
    """Return (full_step_ms, head_only_ms) averaged over NUM_TRIALS."""

    def full_step():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1][:, -1, :]
        return run_head(hidden, head, span_length, vocab_weight)

    # Cache a hidden state for the head-only measurement.
    hidden = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-1][:, -1, :]

    for _ in range(NUM_WARMUP):
        full_step()
        run_head(hidden, head, span_length, vocab_weight)
    torch.cuda.synchronize()

    full_times, head_times = [], []
    for _ in range(NUM_TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        full_step()
        torch.cuda.synchronize()
        full_times.append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_head(hidden, head, span_length, vocab_weight)
        torch.cuda.synchronize()
        head_times.append(time.perf_counter() - t0)

    return np.mean(full_times) * 1000, np.mean(head_times) * 1000, np.std(full_times) * 1000, np.std(head_times) * 1000


def main():
    for model_label, model_name in MODELS.items():
        print(f"\n=== {model_label} ({model_name}) ===")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPE, attn_implementation="flash_attention_2"
        ).cuda().eval()

        dim = model.config.hidden_size
        vocab_size = model.config.vocab_size
        vocab_weight = model.get_output_embeddings().weight  # lm_head weight

        input_ids = torch.randint(
            0, vocab_size, (BATCH_SIZE, SEQ_LEN), device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long, device="cuda")

        print(
            f"batch={BATCH_SIZE} seq_len={SEQ_LEN} dim={dim} vocab={vocab_size}"
        )
        print(f"{'method':>8} | {'full step (ms)':>16} | {'head only (ms)':>16}")
        for span_length in SPAN_LENGTHS:
            head = make_head(dim, span_length)
            full_ms, head_ms, full_std, head_std = time_span(
                model, vocab_weight, head, span_length, input_ids, attention_mask
            )
            label = "NTP" if span_length == 1 else f"MTP-{span_length}"
            print(f"{label:>8} | {full_ms:>16.3f}+-{full_std:>4.1f} | {head_ms:>16.3f}+-{head_std:>4.1f}")
            del head

        del model
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
