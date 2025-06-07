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

# ─── 1) Load & filter ──────────────────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)

# keep only successfully finished runs
df = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# ─── 2) Parse out the metrics dict ────────────────────────────────────────────────
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

# reverse so index 0 is the most recent
sub_ids = sub_ids[::-1]
mdf      = mdf.iloc[::-1].reset_index(drop=True)

# ─── 5) Prepare angles for radar plot ─────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, len(METRIC_ORDER), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # close the loop

# ─── 6) Define a GPT-4o baseline with the true values ─────────────────────────────
baseline = {
    "ROUGE-L":    0.0755,
    "Timing F1":  0.3785,
    "Timing AUC": 0.5358,
    "Action F1":  0.2651,
}
bvals = [baseline[m] for m in METRIC_ORDER]
bvals += bvals[:1]  # close loop

# ─── 7) Plot ─────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 8))
ax = plt.subplot(polar=True)

# a small palette of light-fill colors
palette    = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
fill_alpha = 0.04

# each submission
for i, sid in enumerate(sub_ids):
    row  = mdf.loc[i]
    vals = [row.get(m, np.nan) for m in METRIC_ORDER]
    vals += vals[:1]
    color = palette[i % len(palette)]

    ax.plot(angles, vals, color=color, lw=1.5, label=f"Run {sid}")
    ax.fill(angles, vals, color=color, alpha=fill_alpha)

# overlay GPT-4o baseline in dashed black
ax.plot(angles, bvals,
        color="black", lw=2.5, linestyle="--",
        label="GPT-4o Baseline")
ax.fill(angles, bvals, color="black", alpha=fill_alpha)

# labels & legend
ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_ORDER)
ax.set_title("VideoLLaMA3 Instruction-Generation Performance vs. GPT-4o Baseline", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
