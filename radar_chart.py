# combined_radar.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_cell(cell):
    """
    Convert the string in 'Result File' (a one‐element list of dict)
    into that dict. Returns {} on failure.
    """
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}

# ─── 1) Load & filter ────────────────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)

# keep only successfully finished runs
df = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# ─── 2) Parse out the metrics dict ───────────────────────────────────────────────
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# ─── 3) Select the metrics we want (drop BLEU-4 & Final Score) ──────────────────
METRIC_ORDER = [
    "ROUGE-L",
    "Timing F1",
    "Timing AUC",
    "Action F1",
]

# ─── 4) Get submission identifiers ───────────────────────────────────────────────
for col in ("Submission#", "Submission #", "#"):
    if col in df.columns:
        sub_ids = df[col].tolist()
        break
else:
    sub_ids = list(range(1, len(df) + 1))

# ─── 5) Map run numbers to descriptive names ──────────────────────────────────────
run_names = {
    4:  "Post-processing w/ training-style prompts",
    11: "Basic post-processing",
    13: "Downsampling & 16-frame context",
    14: "Baseline VideoLLaMA3",
}

# ─── 6) Prepare angles for radar plot ─────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, len(METRIC_ORDER), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # close the loop

# ─── 7) Define GPT-4o baseline with actual eval values ───────────────────────────
baseline = {
    "ROUGE-L":    0.0755,
    "Timing F1":  0.3785,
    "Timing AUC": 0.5358,
    "Action F1":  0.2651,
}
bvals = [baseline[m] for m in METRIC_ORDER] + [baseline[METRIC_ORDER[0]]]

# ─── 8) Plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
ax = plt.subplot(polar=True)

# a small palette of light‐fill colors
palette    = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
fill_alpha = 0.04

# plot each run in the original order (4 → 11 → 13 → 14)
for i, sid in enumerate(sub_ids):
    row   = mdf.loc[i]
    vals  = [row.get(m, np.nan) for m in METRIC_ORDER] + [row.get(METRIC_ORDER[0], np.nan)]
    color = palette[i % len(palette)]
    label = f"Run {sid}: {run_names.get(sid, '')}"

    ax.plot(angles, vals, color=color, lw=1.5, label=label)
    ax.fill(angles, vals, color=color, alpha=fill_alpha)

# overlay GPT-4o baseline in a distinct red, no fill
ax.plot(
    angles,
    bvals,
    color="#e34a33",   # a different red
    lw=2.5,
    linestyle="--",
    label="GPT-4o Baseline"
)
# (no ax.fill for baseline)

# labels & legend
ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_ORDER)
ax.set_title(
    "Instruction-Generation Performance Across VideoLLaMA3 Variants\nvs. GPT-4o Baseline",
    pad=20
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
