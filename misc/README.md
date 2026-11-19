## Generation Analysis

Scripts in this folder analyze the per-example dumps produced during evaluation. They compare CoT against C-MTP models (span 2, 3, 4) on a single eval set.

### Generate dumps with evals

Run evaluation with dumping enabled so each checkpoint writes a `dump_*.json`:

- C-MTP: add `--dump True` to your `python test_mtp.py` eval call.
- CoT: set `DUMP = True` in `test_cot.py`
Both of them result in a json dump of all eval statistics in your checkpoint directory.

Each dump holds `{"accuracy": ..., "records": [...]}`, where every record has the decoded think tokens, extracted answer, and ground truth.

### Configuration

Both scripts are standalone and configured by a small block at the top:

- `evalname` — which eval set (`gsm8k-test`, `gsm8k-hard`, `multi-arith`, `svamp`)
- `mtp_paths`, `cot_path` — paths to the dump files
- `mtp_labels`, `span_lens` — labels and span lengths for the C-MTP models
- `save_name` — output filename (written to this folder)

They both print each model's accuracy and record count on load.

#### `read_generations.py`

Writes a flat text file (`save_name`) with one block per example — question, ground truth, and the decoded CoT / C-MTP think text (with a correctness flag). Use it to eyeball where and how the reasoning chains differ across models.

```bash
python read_generations.py
```

#### `self_consistency.py`

Parses each `<<lhs=rhs>>` arithmetic step out of the think text, checks whether `lhs` actually evaluates to `rhs`, and plots the fraction of self-consistent steps by step position for each model. The plot is saved as `{save_name}.png`.

```bash
python self_consistency.py
```


### Gold-trace prefill

We feed `x` fraction of a gold reasoning trace to the model as its thinking prefix and let the model generate rest of the trace along with the final answer. We measure accuracy, sweeping `x` from 0 to 1. For C-MTP, we vary fraction of spans while for CoT we vary fraction of tokens. We use greedy decoding for simplicity.

All generations are written to `assets/generations/`, where you can find existing results stored.

All three scripts are configured by the globals at the top of each file. The checkpoints are the released ones (see the top-level README):

| Setting | `COT_FORMAT` | `CKPT_DIR` (`MODE`) | `DATASET` |
|---|---|---|---|
| NL | `nl` | `cot/seminatural_llama_lora` (`cot`), `cmtp/seminatural_llama_sp{2,3,4}mtp` (`mtp`) | `gsm8k-test` |
| Structured | `structured` | `cmtp/structured_llama_sp{2,3,4}mtp` (`mtp`) | `assets/generations/cot_lora_exp_correct_generations_{gsm8k-test,svamp}.json` |

`SPAN_LENGTH` must match the C-MTP checkpoint. The NL results come from the GSM8k-test gold traces. For the structured setting, the gold traces are the structured CoT model's own correct generations, so CoT accuracy is 1.0 by construction. 

1. **Structured gold traces** (structured setting only). Set `COT_CKPT_DIR` to `cot/structured_llama_lora` and `DATASET` to `gsm8k-test` or `svamp`:

   ```bash
   python misc/generate_cot_traces.py
   ```

   This writes `assets/generations/cot_lora_exp_correct_generations_{dataset}.json`.

2. **Prefill sweep.** Run once per row of the table above:

   ```bash
   python misc/gold_trace_analysis.py
   ```

   This writes `assets/generations/gold_trace_{mode}_{format}_sp{span}_{dataset}.json`, containing `x_values`, `accuracy` and the run config.

3. **Plots and table.** Run:

   ```bash
   python misc/plot_gold_trace.py
   ```

   This writes `gold_trace_plot_{nl_gsm8k,structured_gsm8k,structured_svamp}.png` and `table.md` (accuracy % per `x`) to `assets/generations/`.
