# Training Continuous Chain of Thought Models: A Tale of Two Regimes

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)]() &nbsp;
[![OpenReview](https://img.shields.io/badge/OpenReview-Forum-8c1b13.svg)](https://openreview.net/forum?id=ofq7HnlZjs) &nbsp;
[![Poster](https://img.shields.io/badge/Poster-View-1081c2.svg)](https://github.com/Varun221/cmtp_research/blob/main/assets/poster_fin.pdf)

# Abstract
> Continuous Chain-of-Thought methods replace verbose reasoning traces with a short sequence of dense latent representations. Earlier continuous CoT methods indirectly supervise the latent representations such that its final state match that of verbose reasoning traces, requiring autoregressive, slow generation during training. We introduce C-MTP, a simpler, faster direct supervision approach that models each latent as an average of the embeddings in the CoT traces to be compressed. Our approach outperforms a prior direct supervision method that approximates the distribution of compressed tokens, and performs competitively to slower indirect supervision approaches in existing evaluation setup with simplified CoT traces (less than 100 tokens). Lastly, we extend the evaluation of Continuous CoT methods to complex tasks with longer reasoning traces (> few hundreds reasoning tokens). We find both direct and indirect supervision training methods perform poorly (roughly 65\% performance drop) in this setting, revealing the limitations of current continuous CoT methods.


## Key Findings

![Main Results](assets/mainplot.png)

Head-to-Head comparison of standard CoT finetuning with two ContinuousCoT training methods: **Direct Supervision** (C-MTP, our proposed method) and **Indirect Supervision** (CODI) on three datasets of different CoT traces. _Structured_ consists of compact mathematical expressions of ~25 tokens, _Semi-Natural_ contains sentence per-step explanations of ~62 tokens while _Realistic_ captures traces generated from off-the-shelf LLM (~350 tokens on average). Both methods remain competitive with CoT-SFT on structured/semi-natural traces but collapse to ~35\% of its accuracy on realistic traces.

| Trace type | Dataset | Best method |
|---|---|---|
| Structured (compact expressions) | GSM8k-Aug | Direct (C-MTP) — more efficient, better generalization |
| Semi-Natural (verbose text traces) | GSM8k-Aug-NL | Indirect (CODI) — recurrent objective handles long traces better |
| Realistic (LLM-generated traces) | MATH | Both underperform standard CoT fine-tuning |

## Todo for Repo

- [ ] Upload checkpoints for cot, cmtp and codi.
- [ ] Add arXiv paper link
- [ ] Fill in the citation


## Repository Structure

```
cmtp_research/
├── cmtp/          # C-MTP: Direct supervision method
├── codi/          # CODI: Indirect supervision method
├── data_gsm/      # Data preparation for Structured and Semi-Natural settings
├── data_math/     # Data preparation for Realistic (MATH) setting
└── misc/          # Generation analysis scripts (self-consistency, dumps)
```

## Installation

Create a conda environment with the right Python version and install the dependencies:

```bash
conda create -n cmtp python=3.13.5 -y
conda activate cmtp
pip install -r requirements.txt
```

`torch` and `flash-attn` are CUDA-specific — adjust their versions to match your CUDA build if needed.



## Data Preparation

Three levels of CoT trace difficulty are used. See [data_gsm/README.md](data_gsm/README.md) and [data_math/README.md](data_math/README.md) for preparation steps.

| Level | Dataset | `data_name` |
|---|---|---|
| Structured | GSM8k-Aug (arithmetic expressions) | `icot` |
| Semi-Natural | GSM8k-Aug-NL (text traces) | `icot-nl` |
| Realistic | MATH (LLM-generated traces) | `mathllama` / `mathqwen` |

See Table 7 in paper for examples for each type.

The data paths are hardcoded to a local `data_store/`. Once you have created the data files, update them to your own locations: the training/eval paths in the `get_data_paths` function in each project's `src/dataset.py`, and the `EVAL_PATH` constant near the top of the `test*.py` evaluation scripts.



## Running

- **C-MTP (Direct Supervision):** See [cmtp/README.md](cmtp/README.md)
- **CODI (Indirect Supervision):** See [codi/README.md](codi/README.md)
- **Baselines (SimCoT, CoLaR):** Run with their respective external codebases: [Sim-CoT](https://github.com/InternLM/SIM-CoT)and [CoLaR](https://github.com/xiaomi-research/colar/tree/main)

Both methods support `meta-llama/Llama-3.2-1B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`. All experiments use LoRA (r=128, α=32) with `flash_attention_2`.

The evaluation scripts reproduce the numbers reported in the paper.

**Hardware.** All runs use a single NVIDIA H200 GPU. C-MTP training runs take a few hours; CODI runs take up to a day.



## Acknowledgements

This codebase builds on [CODI](https://github.com/zhenyi4/codi). The baseline experiments use [Sim-CoT](https://github.com/InternLM/SIM-CoT) and [CoLaR](https://github.com/xiaomi-research/colar/tree/main).



## Citation

