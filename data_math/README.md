
## MATH Data Preparation

Use this folder to create training datasets for the `realistic` setting used in the paper.

For realistic generations, we generate CoT traces for MATH problems (hard relative to GSM8k) using instruction-tuned models, then train ContinuousCoT on them. The CoT baseline trains on the original model's predictions, so its performance remains unchanged (see Table 4 in the paper), while both direct and indirect supervision fail to learn effectively.

All commands are run from the repo root (`cmtp_research/`).

*Step 1*: Download and process initial MATH splits.

```bash
python data_math/main_math.py
```

This downloads train and test splits from [nlile/hendrycks-MATH-benchmark](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) and filters out train questions that would produce very long answers, using the LLaMA tokenizer and a max token cap.

Outputs:
- `data_store/math/math_train.json`
- `data_store/math/math_test.json`



*Step 2*: Run inference to get model generations on the train questions.

```bash
python data_math/math_llama_infer.py --data data_store/math/math_train.json --output data_math/mathinfer_outputs/llama_train.json

python data_math/math_qwen_infer.py --data data_store/math/math_train.json --output data_math/mathinfer_outputs/qwen_train.json
```

Inference outputs are written to `data_math/mathinfer_outputs/`.

A copy of the inference outputs is available at XXX.



*Step 3*: Process inference outputs into training-ready data.

```bash
python data_math/math_llama_process.py
python data_math/math_qwen_process.py
```

Outputs written to `data_store/math/`:
- `llama_train_full.json` / `qwen_train_full.json` — all train items where the CoT contains `\boxed{}`
- `llama_train_correct.json` / `qwen_train_correct.json` — subset of the above where the model answered correctly
- `llama_test.json` / `qwen_test.json` — test split with model-specific chat template applied to questions for evaluation

The final train files can be used with the `cmtp` or `codi` folder using `mathqwen` or `mathllama` as the dataname. See the launch commands in the corresponding project folder for more details.

