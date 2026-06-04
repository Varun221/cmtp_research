# CODI: Indirect Supervision for Continuous CoT

CODI is an indirect supervision approach for training Continuous Chain-of-Thought models introduced in [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)

This is a fork of the [original CODI codebase](https://github.com/zhenyi4/codi/tree/main). The main change is that data loading is extracted into `src/dataset.py` with support for the following datasets: `icot`, `icot-nl`, `mathllama`, `mathqwen`, and random subset variants. The rest of the changes are simplifications and minor additions to the original code.

Training:
- Use `scripts/train_codi_latent6.sh` for Structured/Semi-Natural 
- Use `scripts/train_codi_math.sh` for Realistic

Evaluation:
- Use `scripts/test_codi.sh`
