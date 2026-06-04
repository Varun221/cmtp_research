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
