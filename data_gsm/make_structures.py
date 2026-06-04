"""Generate structure-annotated datasets from GSM8k-Aug.

Parses CoT expressions into structural templates, assigns structure IDs,
partitions by structure length, and saves train/test splits.

Uncomment line 210 to generate cumulative coverage plot (Fig 3 in paper).
"""

import ast
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import plotly.graph_objects as go

EXP_TRAIN_NAME = "gsm8kaug_train.json"

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_strip_alnum = re.compile(r"[a-zA-Z0-9.]")


def parse_cot_steps(cot):
    pattern = r"<<([^>]+)>>"
    steps = []
    total = 0
    for match in re.finditer(pattern, cot):
        expr = match.group(1)
        if "=" not in expr:
            continue
        total += 1
        lhs, result = expr.rsplit("=", 1)
        lhs = lhs.strip().replace(",", "")
        result = result.strip()
        try:
            ast.parse(lhs, mode="eval")
            steps.append((lhs, result))
        except SyntaxError:
            pass
    return steps, total


def build_template(steps):
    val_to_var = {}
    results_map = {}
    fresh_idx = [0]

    def get_var(num_str):
        if num_str in results_map:
            return results_map[num_str]
        if num_str not in val_to_var:
            idx = fresh_idx[0]
            name = chr(ord("a") + idx % 26) + (str(idx // 26) if idx >= 26 else "")
            fresh_idx[0] += 1
            val_to_var[num_str] = name
        return val_to_var[num_str]

    parts = []
    key_parts = []
    for i, (lhs, result) in enumerate(steps):
        template_lhs = NUM_RE.sub(lambda m: get_var(m.group(0)), lhs)
        res_name = f"r{i}"
        results_map[result] = res_name
        parts.append(f"{template_lhs}={res_name}")
        key_parts.append(template_lhs)

    return ", ".join(parts), tuple(key_parts)


def greedy_partition(entries):
    set_a, set_b = [], []
    sum_a, sum_b = 0, 0
    for key, count in sorted(entries, key=lambda x: -x[1]):
        if sum_a <= sum_b:
            set_a.append((key, count))
            sum_a += count
        else:
            set_b.append((key, count))
            sum_b += count
    return set_a, set_b


def split_train_test(data, test_frac=0.02):
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * (1 - test_frac))
    return data[:cut], data[cut:]


def plot_cumulative_coverage(structure_counts):
    counts = sorted(structure_counts.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)
    cumulative = np.cumsum(counts) / sum(counts) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ranks,
            y=cumulative,
            mode="lines",
            name="Coverage",
            line=dict(color="steelblue", width=2),
        )
    )
    fig.add_hline(
        y=50, line_dash="dash", line_width=1, line_color="gray", annotation_text="50%"
    )
    fig.add_hline(
        y=80, line_dash="dash", line_width=1, line_color="orange", annotation_text="80%"
    )
    fig.add_hline(
        y=95, line_dash="dash", line_width=1, line_color="red", annotation_text="95%"
    )

    for pct, color in [(50, "gray"), (80, "orange"), (95, "red")]:
        try:
            n = next(i + 1 for i, c in enumerate(cumulative) if c >= pct)
            fig.add_annotation(
                x=n,
                y=pct,
                text=f"{n} structs",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                font=dict(color=color, size=11),
                ax=40,
                ay=-30,
            )
        except StopIteration:
            pass

    fig.update_layout(
        title="Cumulative coverage",
        xaxis_title="Top-N structures",
        yaxis_title="% of dataset covered",
        plot_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=50),
        width=600,
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray", zeroline=False)
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
        zeroline=False,
        range=[0, 105],
    )
    fig.show()


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_store_dir = os.path.join(curr_dir, "..", "data_store")
    out_dir = os.path.join(data_store_dir, "genstructures")
    combined_dir = os.path.join(out_dir, "combined")

    for d in [out_dir, combined_dir]:
        if os.path.exists(d):
            print(f"Warning: {d} already exists and will be overwritten.")
        os.makedirs(d, exist_ok=True)

    with open(os.path.join(data_store_dir, EXP_TRAIN_NAME)) as f:
        data = json.load(f)
    print(f"Loaded exp train: {len(data)}")

    # Parse CoT steps into structural templates
    parsed = []
    skipped_empty = skipped_partial = skipped_mismatch = 0
    for item in data:
        steps, total = parse_cot_steps(item["cot"])
        if not steps:
            skipped_empty += 1
            continue
        if len(steps) != total:
            skipped_partial += 1
            continue
        template, key = build_template(steps)
        actual_str = ", ".join(f"{lhs}={result}" for lhs, result in steps)
        if _strip_alnum.sub("", template) != _strip_alnum.sub("", actual_str):
            skipped_mismatch += 1
            continue
        parsed.append({"item": item, "steps": steps, "key": key, "template": template})
    print(
        f"Parsed: {len(parsed)}, skipped (no steps): {skipped_empty}, "
        f"skipped (partial): {skipped_partial}, skipped (mismatch): {skipped_mismatch}"
    )

    groups = defaultdict(list)
    for p in parsed:
        groups[p["key"]].append(p)
    structure_counts = Counter({k: len(v) for k, v in groups.items()})
    print(f"\nUnique structures: {len(groups)}")

    print("\nTop 10 most common structures:\n")
    for key, count in structure_counts.most_common(10):
        ex = groups[key][0]
        print(f"Count: {count:5d}  Template: {ex['template']}")
        print(f"          Q: {ex['item']['question']}")
        print(f"        CoT: {ex['item']['cot']}")
        print()

    counts = sorted(structure_counts.values(), reverse=True)
    cumulative = np.cumsum(counts) / sum(counts) * 100

    # Uncomment to show cumulative coverage plot
    # plot_cumulative_coverage(structure_counts)

    for n in [10, 50, 100, 500, 1000, 5000, 10000]:
        print(f"Top {n:>5} structures cover {cumulative[n - 1]:.1f}% of the dataset")

    # Assign integer IDs ordered by frequency (most common = 0)
    key_to_id = {key: i for i, (key, _) in enumerate(structure_counts.most_common())}
    for p in parsed:
        sid = key_to_id[p["key"]]
        p["structure_id"] = sid
        p["item"]["structure_id"] = sid
    for item in data:
        if "structure_id" not in item:
            item["structure_id"] = None
    print(f"Assigned {len(key_to_id)} unique structure IDs")

    data_parsed = [item for item in data if item["structure_id"] is not None]
    print(f"Original: {len(data)}, After removing skipped: {len(data_parsed)}")

    # Group structures by key length (number of steps)
    length_groups = defaultdict(list)
    for key, count in structure_counts.items():
        length_groups[len(key)].append((key, count))

    length_pop = sorted(
        [
            (length, sum(c for _, c in entries))
            for length, entries in length_groups.items()
        ],
        key=lambda x: -x[1],
    )
    print(f"\n{'Length':>8}  {'Total items':>12}  {'Unique structs':>14}")
    print("-" * 40)
    for length, total in length_pop:
        print(f"{length:>8}  {total:>12,}  {len(length_groups[length]):>14,}")

    # Greedy partition into two equal-sample sets for lengths 2, 3, 4
    splits = {}
    random.seed(42)
    for length in [2, 3, 4]:
        set_a, set_b = greedy_partition(length_groups[length])
        keys_a = {key for key, _ in set_a}
        keys_b = {key for key, _ in set_b}
        assert (
            len(keys_a & keys_b) == 0
        ), f"Length-{length} set_a/set_b have overlapping structures!"

        sum_a = sum(c for _, c in set_a)
        sum_b = sum(c for _, c in set_b)
        print(
            f"\nLength-{length}: set_a={len(set_a)} structs ({sum_a:,} items), "
            f"set_b={len(set_b)} structs ({sum_b:,} items)"
        )

        for tag, keys in [(f"len{length}_seta", keys_a), (f"len{length}_setb", keys_b)]:
            items = [p["item"] for p in parsed if p["key"] in keys]
            train, test = split_train_test(items)
            splits[f"{tag}_train"] = train
            splits[f"{tag}_test"] = test
            print(f"  {tag}: train={len(train):,}, test={len(test):,}")

    print("\nNo structure collisions between set_a and set_b for lengths 2, 3, 4.")

    # Combined datasets
    def combine(*tags):
        return sum((splits[t] for t in tags), [])

    combined = {
        "len2len4_train": combine(
            "len2_seta_train", "len2_setb_train", "len4_seta_train", "len4_setb_train"
        ),
        "len2len3_train": combine(
            "len2_seta_train", "len2_setb_train", "len3_seta_train", "len3_setb_train"
        ),
        "len2_test": combine("len2_seta_test", "len2_setb_test"),
        "len3_test": combine("len3_seta_test", "len3_setb_test"),
        "len4_test": combine("len4_seta_test", "len4_setb_test"),
    }

    len2_train = splits["len2_seta_train"] + splits["len2_setb_train"]
    len3_train = splits["len3_seta_train"] + splits["len3_setb_train"]
    oneperc = len(len2_train) // 100
    combined["len2_plus1p_len3_train"] = len2_train + random.sample(len3_train, oneperc)

    print("\nCombined sets:")
    for tag, items in combined.items():
        print(f"  {tag}: {len(items):,}")

    # Save all splits
    for tag, items in splits.items():
        path = os.path.join(out_dir, f"gsm8kaug_{tag}.json")
        with open(path, "w") as f:
            json.dump(items, f)

    for tag, items in combined.items():
        path = os.path.join(combined_dir, f"gsm8kaug_{tag}.json")
        with open(path, "w") as f:
            json.dump(items, f)

    print("Done.")


if __name__ == "__main__":
    main()
