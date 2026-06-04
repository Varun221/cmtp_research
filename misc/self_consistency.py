"""Read CoT/MTP dumps and produce the per-position step self-consistency plot."""

import os
import re
import json
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Edit the eval to generate the text file for:

# gsm8k-test, gsm8k-hard, multi-arith, svamp
evalname = "gsm8k-test"

mtp_paths = [
    f"/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp2mtp/dump_mtp_{evalname}_span2.json",
    f"/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp3mtp/dump_mtp_{evalname}_span3.json",
    f"/scratch/vy2142/cmtp_research_extra/ckpts/cmtp/structured_llama_sp4mtp/dump_mtp_{evalname}_span4.json",
]
cot_path = f"/scratch/vy2142/cmtp_research_extra/ckpts/cot/structured_llama_lora/dump_cot_{evalname}.json"

mtp_labels = ["C-MTP-2", "C-MTP-3", "C-MTP-4"]
span_lens = [2, 3, 4]

save_name = f"self_consistency_{evalname}.txt"

# Loading

mtp_dumps = [json.load(open(p)) for p in mtp_paths]
cot_dump = json.load(open(cot_path))

print(
    f"CoT accuracy: {cot_dump['accuracy']*100:.2f}% ({len(cot_dump['records'])} records)"
)
for label, d in zip(mtp_labels, mtp_dumps):
    print(f"{label} accuracy: {d['accuracy']*100:.2f}% ({len(d['records'])} records)")

# Build samples

cot_think_text_by_idx = {}
for r in cot_dump["records"]:
    ptd = r.get("pred_tokens_decoded", [])
    L = r.get("len_cot")
    think_tokens = ptd[: max(L - 1, 0)] if L else ptd
    cot_think_text_by_idx[r["idx"]] = "".join(think_tokens)

mtp_think_text = {}  # label -> {idx: text}
for label, dump in zip(mtp_labels, mtp_dumps):
    mtp_think_text[label] = {}
    for r in dump["records"]:
        chunks = r.get("mtp_think_tokens_decoded", [])
        mtp_think_text[label][r["idx"]] = "|".join("".join(c) for c in chunks)

cot_records_by_idx = {r["idx"]: r for r in cot_dump["records"]}
mtp_records_by_label_idx = {
    label: {r["idx"]: r for r in dump["records"]}
    for label, dump in zip(mtp_labels, mtp_dumps)
}


def is_correct(r):
    p, g = r.get("extracted_answer"), r.get("ground_truth")
    try:
        return float(p) == float(g)
    except (TypeError, ValueError):
        return p == g


samples = []
for idx, cot_text in cot_think_text_by_idx.items():
    cot_record = cot_records_by_idx[idx]
    s = {
        "idx": idx,
        "q": cot_record["question"],
        "gt": cot_record["ground_truth"],
        "cot_text": cot_text,
        "cot_correct": is_correct(cot_record),
    }
    for label in mtp_labels:
        mtp_record = mtp_records_by_label_idx[label].get(idx)
        if mtp_record is None:
            continue
        s[f"{label}_text"] = mtp_think_text[label][idx]
        s[f"{label}_correct"] = is_correct(mtp_record)
    samples.append(s)

print(f"Built {len(samples)} samples")

# Extract steps and check consistency.


def safe_eval(expr):
    expr = expr.strip()
    if not re.match(r"^[\d\s\+\-\*\/\.\(\)]+$", expr):
        return None
    try:
        return float(eval(expr))
    except Exception:
        return None


def parse_steps(text):
    """Return list of (lhs_str, rhs_str) for each <<...>> step."""
    text = text.replace("|", "").replace("<leot>", "").replace("<lmot>", "")
    steps = []
    for m in re.finditer(r"<<([^>]+)>>", text):
        expr = m.group(1).strip()
        if "=" not in expr:
            continue
        eq_pos = expr.index("=")
        lhs = expr[:eq_pos].strip()
        rhs = expr[eq_pos + 1 :].lstrip("=").strip()  # handle accidental ==
        steps.append((lhs, rhs))
    return steps


def is_consistent(lhs, rhs):
    lhs_val = safe_eval(lhs)
    if lhs_val is None:
        return False
    rhs_clean = re.match(r"-?[\d]+\.?[\d]*", rhs.replace(",", ""))
    if not rhs_clean:
        return False
    try:
        rhs_val = float(rhs_clean.group())
    except Exception:
        return False
    return abs(lhs_val - rhs_val) <= max(0.5, 0.01 * abs(rhs_val))


# Generating the consistency plot.


def analysis_step_consistency_simple(samples, max_pos=7):
    model_keys = ["CoT"] + list(mtp_labels)
    text_key = {m: ("cot_text" if m == "CoT" else f"{m}_text") for m in model_keys}

    model_vecs = defaultdict(list)
    for s in samples:
        for m in model_keys:
            if text_key[m] in s:
                steps = parse_steps(s[text_key[m]])
                if steps:
                    model_vecs[m].append(
                        [1 if is_consistent(lhs, rhs) else 0 for lhs, rhs in steps]
                    )

    print("=== Per-position step self-consistency ===")
    print(f"{'Model':<8}" + "".join(f"  step{i}(n)" for i in range(max_pos)))

    palette = ["steelblue", "darkorange", "tomato", "green", "purple", "brown"]
    colors = {"CoT": "black"}
    for i, label in enumerate(mtp_labels):
        colors[label] = palette[i % len(palette)]

    fig, ax = plt.subplots(figsize=(8, 4))
    for model in model_keys:
        vecs = model_vecs.get(model, [])
        pos_avgs, pos_ns = [], []
        for i in range(max_pos):
            vals = [v[i] for v in vecs if i < len(v)]
            pos_avgs.append(np.mean(vals) if vals else np.nan)
            pos_ns.append(len(vals))

        print(
            f"{model:<8}" + "".join(f"  {a:.2f}({n})" for a, n in zip(pos_avgs, pos_ns))
        )

        xs = [i for i, n in enumerate(pos_ns) if n >= 5]
        ys = [pos_avgs[i] for i in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                label=model,
                color=colors.get(model, "gray"),
                linewidth=2,
                markersize=5,
            )

    ax.set_xlabel("Step position")
    ax.set_ylabel("Fraction self-consistent")
    ax.set_xticks(range(max_pos))
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.splitext(save_name)[0] + ".png",
    )
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    analysis_step_consistency_simple(samples)
