
# C-MTP: Continuous Latents using Multi Token Prediction

C-MTP is a direct supervision method for training ContinuousCoT models. See the paper for an in-depth explanation.

**Training.** C-MTP trains a pretrained LLM to perform ContinuousCoT via distillation from a CoT teacher. Instead of seeing raw CoT tokens, the student receives the chain-of-thought grouped into fixed-size spans of `span_length` tokens (2, 3, or 4). Each span is compressed into a single aggregated embedding, and a projection head learns to predict all `span_length` tokens of the next span from it. Training uses three losses jointly: a span prediction loss, a standard next-token prediction loss, and a layer-wise distillation loss against the teacher.


Two training modes are supported:
- Warmstart-Init: Use a frozen CoT as the teacher
- Multitask-Train: Train the same backbone for CoT and ContinuousCoT and use CoT forward pass as the teacher.

See `src/model.py` for the implementation.

**Inference.** At each step the model generates `span_length` tokens, aggregates them into a single embedding, and feeds that embedding as input to the next step. This repeats until an end-of-think token is produced. 

See `test_mtp.py` for the implementation.

---

## Directory Structure

```
cmtp/
├── train.py                # Main training entry point
├── test_mtp.py             # Evaluation for C-MTP models
├── test_cot.py             # Evaluation for CoT models (teacher / baseline)
├── src/
│   ├── model.py            # TrainingModel, ModelArguments, TrainingArguments
│   ├── dataset.py          # Dataset loading and tokenization
│   ├── utils.py            # Seeding, file encoding, timing utilities
│   └── timing.py           # Used for timing analysis
└── scripts/
    ├── train_mtp.sh         # Train C-MTP on structured/semi-natural traces
    ├── train_mtp_math.sh    # Train C-MTP on realistic (MATH) traces
    ├── train_cot.sh         # Train CoT on structured/semi-natural traces
    ├── train_cot_math.sh    # Train CoT on realistic (MATH) traces
    ├── test_mtp.sh          # Evaluate C-MTP across all settings
    ├── test_cot.sh          # Evaluate CoT teacher across all settings
    ├── merge_lora.py        # Merge LoRA weights into base model
    ├── merge_lora.sh        # Merge script for realistic CoT checkpoints
    └── math_grader.py       # Answer grading utility for MATH eval.
```


## Data Preparation

Use `data_gsm/` to prepare Structured and Semi-Natural training data, or `data_math/` for realistic (MATH) training data. After running the preparation scripts, modify the data paths in the `get_data_paths` function in `src/dataset.py`.

Supported `data_name` values:

| `data_name` | Description |
|---|---|
| `icot` | Full structured dataset (GSM8k-Aug expressions) |
| `icot-nl` | Full semi-natural dataset (GSM8k-Aug-NL) |
| `mathllama` | Realistic Llama CoT generations with Llama chat template |
| `mathqwen` | Realistic Qwen CoT generations with Qwen chat template |

Random subsets of the structured and semi-natural datasets can be loaded by appending the subset size to the data name (See function for details).

## Training

All scripts run from `cmtp_research/cmtp/`. Set `SAVE_DIR` and WandB env vars before running.

### Step 1: Train a CoT teacher (Warmstart mode only)

Skip this step if using Multitask mode.

- **Structured / Semi-natural:** `bash scripts/train_cot.sh` — use full model training (no LoRA) for the warmstart checkpoint; use LoRA training for the CoT baseline.
- **Realistic (MATH):** `bash scripts/train_cot_math.sh`

For the realistic setting, the MATH dataset is small enough that full fine-tuning overfits, so the CoT teacher is trained with LoRA. The adapter must be merged into the base model before it can be used as a warmstart checkpoint:

```bash
bash scripts/merge_lora.sh
```

### Step 2: Train C-MTP

**Warmstart-Init** — initialize the student from the CoT teacher checkpoint, then train with distillation:

- **Structured / Semi-natural:** `bash scripts/train_mtp.sh` (pass the CoT checkpoint via `warmstart_args`)
- **Realistic (MATH):** `bash scripts/train_mtp_math.sh` (pass the merged CoT checkpoint)

**Multitask-Train** — no CoT teacher needed; the same backbone is trained jointly on CoT and ContinuousCoT:

- **Structured / Semi-natural:** `bash scripts/train_mtp.sh` (use `multitask_args`)


## Evaluation

See `test_cot.sh` for CoT model evaluation and `test_mtp.sh` for MTP model evaluation.


## References

This code was built upon starting from CODI training codebase. [CODI](https://github.com/zhenyi4/codi/tree/main)
