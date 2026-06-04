## GSM8k-Aug Data Preparation

Scripts in this folder prepare data for the Structured (GSM8k-Aug) and Semi-Natural (GSM8k-Aug-NL) experiments in the paper.

All commands are run from the repo root (`cmtp_research/`).

#### `main_gsm.py`
Download and prepare the main train and val splits.

Outputs:
- `data_store/gsm8kaug_train.json` — Structured train dataset
- `data_store/gsm8kaug_val.json` — Structured val dataset
- `data_store/gsm8kaug_nl_train.json` — Semi-Natural train dataset
- `data_store/gsm8kaug_nl_val.json` — Semi-Natural val dataset



#### `make_subsets.py`
Creates nested random subsets of the Structured and Semi-Natural datasets, where each smaller subset is contained within the larger ones. Used for sample efficiency and fixed wall-clock experiments in the paper.

Subset sizes: 12k, 25k, 50k, 60k, 70k, 88k, 100k, 120k, 150k, 200k, 250k, 300k, 350k

Outputs to `data_store/randomsubsets/`:
- `gsm8kaug_train_{size}.json` — Structured subset of the given size
- `gsm8kaug_nl_train_{size}.json` — corresponding Semi-Natural subset (matched by question)



#### `make_structures.py`
Analyzes problem structures in GSM8k-Aug and creates structure-based held-out splits for the generalization experiments. Uses the train split produced by `main_gsm.py`.

Each GSM8k-Aug problem has a chain-of-thought that performs a sequence of arithmetic operations. This script identifies the abstract *structure* of each reasoning chain by replacing concrete numbers with symbolic placeholders, leaving only the pattern of operations and how intermediate results feed into later steps. Problems that share the same abstract structure are grouped together.

The script:
- Parses and assigns a structure template to each question and CoT (items that cannot be parsed are dropped)
- Assigns each unique structure an integer ID ordered by frequency
- Prints how much of the dataset is covered by the top 10, 50, 100, 500, 1,000, 5,000, and 10,000 most common structures
- Optionally generates a cumulative structure coverage plot (Fig. 3 in paper; uncomment line 211 to enable)
- Creates structure-based held-out splits for problems with 2, 3, or 4 reasoning steps

For the held-out experiments: within each step-count group, problems are partitioned by structure into two roughly equal halves — set A and set B — with no structure appearing in both halves. Within each half, 2% of problems are held out as a test set. Training on one half and evaluating on both measures whether the model generalizes to problem structures it has never seen during training.

Outputs to `data_store/genstructures/`:
- `gsm8kaug_len{n}_set{a/b}_train.json`
- `gsm8kaug_len{n}_set{a/b}_test.json` 

for `n` in `{2, 3, 4}`
