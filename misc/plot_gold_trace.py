"""Plot gold-trace prefill analysis results (accuracy vs. prefill fraction x).

Reads the gold_trace_*.json results produced by gold_trace_analysis.py from
assets/generations/, renders the three comparison figures, and writes the
underlying accuracies to table.md, all in that same folder.
"""

import os
import json

import matplotlib.pyplot as plt

GEN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "generations")

TITLE_BASE = "Gold-trace prefill analysis for CoT and C-MTP models"

PLOTS = [
    {
        "heading": "Plot 1: NL format, GSM8K-test gold traces",
        "title": f"{TITLE_BASE}\nNL format, GSM8K-test gold traces",
        "out_name": "gold_trace_plot_nl_gsm8k.png",
        "series": [
            ("CoT", "gold_trace_cot_nl_sp2_gsm8k-test.json", "o"),
            ("C-MTP-2", "gold_trace_mtp_nl_sp2_gsm8k-test.json", "s"),
            ("C-MTP-3", "gold_trace_mtp_nl_sp3_gsm8k-test.json", "^"),
            ("C-MTP-4", "gold_trace_mtp_nl_sp4_gsm8k-test.json", "D"),
        ],
    },
    {
        "heading": "Plot 2: Structured format, GSM8K-test (CoT = 1.0)",
        "title": f"{TITLE_BASE}\nStructured format, GSM8K-test (CoT = 1.0)",
        "out_name": "gold_trace_plot_structured_gsm8k.png",
        "series": [
            ("C-MTP-2", "gold_trace_mtp_structured_sp2_cot_lora_exp_correct_generations_gsm8k-test.json", "s"),
            ("C-MTP-3", "gold_trace_mtp_structured_sp3_cot_lora_exp_correct_generations_gsm8k-test.json", "^"),
            ("C-MTP-4", "gold_trace_mtp_structured_sp4_cot_lora_exp_correct_generations_gsm8k-test.json", "D"),
        ],
    },
    {
        "heading": "Plot 3: Structured format, SVAMP (CoT = 1.0)",
        "title": f"{TITLE_BASE}\nStructured format, SVAMP (CoT = 1.0)",
        "out_name": "gold_trace_plot_structured_svamp.png",
        "series": [
            ("C-MTP-2", "gold_trace_mtp_structured_sp2_cot_lora_exp_correct_generations_svamp.json", "s"),
            ("C-MTP-3", "gold_trace_mtp_structured_sp3_cot_lora_exp_correct_generations_svamp.json", "^"),
            ("C-MTP-4", "gold_trace_mtp_structured_sp4_cot_lora_exp_correct_generations_svamp.json", "D"),
        ],
    },
]


def load(name):
    with open(os.path.join(GEN_DIR, name), "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["x_values"], d["accuracy"]


def make_plot(series, title, out_name):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, fname, style in series:
        x, acc = load(fname)
        ax.plot(x, acc, marker=style, linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Prefill fraction $x$ of gold trace")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([round(0.1 * i, 1) for i in range(11)])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(GEN_DIR, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


def table_section(heading, series):
    """Markdown table of accuracy (%) per x; the x=0 and x=1 rows are bolded."""
    labels = [label for label, _, _ in series]
    columns = [load(fname) for _, fname, _ in series]
    x_values = columns[0][0]
    lines = [
        f"## {heading}",
        "",
        "| $x$ | " + " | ".join(labels) + " |",
        "|-----|" + "|".join("-" * (len(label) + 2) for label in labels) + "|",
    ]
    for i, x in enumerate(x_values):
        cells = [f"{x:.1f}"] + [f"{acc[i] * 100:.2f}" for _, acc in columns]
        if i in (0, len(x_values) - 1):
            cells = [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_table(out_name="table.md"):
    sections = [
        "# Gold-trace prefill analysis — accuracy vs. prefill fraction $x$",
        "Values extracted from the `gold_trace_*.json` results plotted by `plot_gold_trace.py`. "
        "Accuracy is a percentage (%).",
    ]
    sections += [table_section(p["heading"], p["series"]) for p in PLOTS]
    out_path = os.path.join(GEN_DIR, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    for p in PLOTS:
        make_plot(p["series"], p["title"], p["out_name"])
    write_table()
