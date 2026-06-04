"""Read CoT/MTP dumps and write a flat text file of per-example generations for analysis."""

import os
import json

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

save_name = f"generations_{evalname}.txt"

# Loading

mtp_dumps = [json.load(open(p)) for p in mtp_paths]
cot_dump = json.load(open(cot_path))

print(f"CoT accuracy: {cot_dump['accuracy']*100:.2f}% ({len(cot_dump['records'])} records)")
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


# Write file for analysis.

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_name)
with open(out_path, "w") as f:
    for idx in cot_think_text_by_idx:
        cot_record = cot_records_by_idx[idx]
        cot_think = cot_think_text_by_idx[idx]

        f.write(f"IDX: {idx}\n")
        f.write(f"Q: {cot_record['question']}\n")
        f.write(f"GT: {cot_record['ground_truth']}\n")
        f.write(f"CoT Think ({is_correct(cot_record)}): {repr(cot_think)}\n")
        for label in mtp_labels:
            mtp_record = mtp_records_by_label_idx[label].get(idx)
            mtp_think = mtp_think_text[label].get(idx, "")
            f.write(
                f"{label} Think ({is_correct(mtp_record) if mtp_record else None}): "
                f"{repr(mtp_think)}\n"
            )
        f.write("-" * 50 + "\n")

print(f"Wrote {len(cot_think_text_by_idx)} examples to {out_path}")
