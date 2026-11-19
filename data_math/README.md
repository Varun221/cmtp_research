
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


## GSM8k-Aug with LLM traces (GSM-realistic)

The same infer/process pipeline produces Llama-generated CoT traces on GSM8k-Aug questions instead of MATH. Only Llama-3.2-1B-Instruct is used for this setting. Input is `data_store/gsm8kaug_train.json` (already shuffled).

*Step 1*: Run inference on all GSM8k-Aug train questions (one generation per question).

```bash
python data_math/math_llama_infer.py --data data_store/gsm8kaug_train.json \
    --output data_store/gsmllm/infer/llama_train.json
```

*Step 2*: Process into training-ready data and chat-templated eval sets.

```bash
python data_math/gsm_process.py
```

Outputs written to `data_store/gsmllm/`:
- `llama_train_full.json`: all train items whose generation contains `\boxed{}`
- `llama_train_correct.json`: subset of the above where the model answered correctly
- `llama_val.json`: held-out 1% slice (correct-filtered), used as loss-only eval during training
- `llama_{gsm8k-test,gsm8k-hard,multi-arith,svamp}.json`: downstream eval sets with the Llama chat template applied to questions

Sample counts:

| Stage | Count |
|---|---|
| GSM8k-Aug train questions (inference input) | 380,015 |
| Held out as val (last 1%, before filtering) | 3,800 |
| Train questions after holding out val | 376,215 |
| `llama_train_full.json` (contains `\boxed{}`) | 349,204 |
| `llama_train_correct.json` (correct answer, used for training) | 206,701 |
| `llama_val.json` (correct-filtered val) | 2,064 |

So the 206,701 training samples are filtered from 376,215 questions, not from the full 380,015.

Use `gsmllama` as the `data_name` in the `cmtp` or `codi` folder. The downstream eval jsons are passed as `.json` paths in the eval `EVLIST`. See the launch scripts in each project folder.
