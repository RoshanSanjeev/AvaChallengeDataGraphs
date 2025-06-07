# radar_plot_flipped_labels.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def parse_metrics_cell(cell: str) -> dict:
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}

# 1) Load only finished submissions
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)
df   = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# 2) Pull the metrics JSON out into its own DataFrame
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# 3) Define the axes of our radar
METRIC_ORDER = [
    "Final Score",
    "BLEU-4",
    "ROUGE-L",
    "Timing F1",
    "Timing AUC",
    "Action F1",
]

# 4) Hard‐coded GPT-4o baseline values
baseline = {
    "Final Score": 0.2651,
    "BLEU-4":       0.0000,
    "ROUGE-L":      0.0755,
    "Timing F1":    0.3785,
    "Timing AUC":   0.5358,
    "Action F1":    0.3355,
}

# 5) Figure out our submission IDs
if "Submission #" in df.columns:
    subs = df["Submission #"].astype(str).tolist()
elif "Submission#" in df.columns:
    subs = df["Submission#"].astype(str).tolist()
else:
    subs = [str(i+1) for i in range(len(df))]

# 6) Reverse them so the most recent is first on the legend
subs_rev = subs[::-1]
mdf_rev  = mdf.iloc[::-1].reset_index(drop=True)

# 7) Build the “flip mapping”: most-recent→oldest to oldest→most-recent
#    e.g. {'14':'4', '13':'11', '11':'13', '4':'14'}
flip_map = { new: old for new, old in zip(subs_rev, subs) }

# 8) Radar setup
labels = METRIC_ORDER
N      = len(labels)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
ax.set_facecolor("white")
ax.grid(color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
plt.xticks(angles[:-1], labels, color="black", size=12)

# radial ticks
max_val   = max(mdf_rev[METRIC_ORDER].max().max(), max(baseline.values()))
tick_locs = np.linspace(0, max_val, 4)
tick_labels = [f"{t:.2f}" for t in tick_locs]
plt.yticks(tick_locs, tick_labels, color="gray", size=10)
plt.ylim(0, tick_locs[-1])

# 9) Pastel palette
pastel_colors = ["#66c2a5", "#8da0cb", "#fc8d62", "#e78ac3", "#ffd92f"]
color_cycle   = cycle(pastel_colors)

# 10) Plot each run, but label with the flipped ID
for sid in subs_rev:
    idx  = subs_rev.index(sid)
    row  = mdf_rev.loc[idx]
    vals = [row.get(m, 0.0) for m in METRIC_ORDER]
    vals += vals[:1]

    c = next(color_cycle)
    ax.plot(
        angles, vals,
        color=c, linewidth=2.0, marker="o", markersize=5,
        label=f"Run {flip_map[sid]}"
    )
    ax.fill(angles, vals, color=c, alpha=0.15)

# 11) GPT-4o Baseline
bvals = [baseline[m] for m in METRIC_ORDER] + [baseline[METRIC_ORDER[0]]]
ax.plot(
    angles, bvals,
    color="black", linewidth=2.5, linestyle="--",
    label="GPT-4o Baseline"
)
ax.fill(angles, bvals, color="black", alpha=0.07)

# 12) Finish up
plt.title("All Submissions vs. GPT-4o Baseline", size=14, y=1.10)
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.05), fontsize=10, frameon=False)
plt.tight_layout()
plt.show()
