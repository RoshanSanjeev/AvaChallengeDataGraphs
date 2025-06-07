# bar_graphs.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_cell(cell: str) -> dict:
    """
    The 'Result File' cell looks like:
      "[{'BLEU-4': 0.0, 'ROUGE-L': 0.0755, ...}]"
    We convert it into a Python dict.
    """
    try:
        # replace single‐quotes with double so json.loads can handle it
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        # if parsing fails, return an empty dict
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load & filter
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)

# keep only successfully finished runs
df = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Parse the metrics JSON‐snippet into its own DataFrame
# ─────────────────────────────────────────────────────────────────────────────
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# ─────────────────────────────────────────────────────────────────────────────
# 3) Select the metrics & order you want
# ─────────────────────────────────────────────────────────────────────────────
METRIC_ORDER = [
    "Final Score",
    "BLEU-4",
    "ROUGE-L",
    "Timing F1",
    "Timing AUC",
    "Action F1",
]

# baseline values for GPT-4o
baseline = {
    "Final Score": 0.2651,
    "BLEU-4":       0.0000,
    "ROUGE-L":      0.0755,
    "Timing F1":    0.3785,
    "Timing AUC":   0.5358,
    "Action F1":    0.3355,
}

# ─────────────────────────────────────────────────────────────────────────────
# 4) Figure out your submission identifiers
# ─────────────────────────────────────────────────────────────────────────────
# try common column names for a submission ID
for col in ("Submission #", "Submission#", "#"):
    if col in df.columns:
        sub_ids = df[col].astype(str).tolist()
        break
else:
    # fallback to a simple 1..N
    sub_ids = [str(i+1) for i in range(len(df))]

# reverse both so row 0 => most‐recent submission
sub_ids = sub_ids[::-1]
mdf      = mdf.iloc[::-1].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Build grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
n_metrics    = len(METRIC_ORDER)
n_submissions = len(sub_ids)
x = np.arange(n_metrics)

# width of each bar
bar_width = 0.8 / (n_submissions + 1)  # +1 leaves room for the baseline

plt.figure(figsize=(12, 6))
cmap = plt.get_cmap("tab20")

# plot each submission as a bar in the group
for i, sid in enumerate(sub_ids):
    # extract that row's metrics in the correct order (fill missing with NaN)
    vals = [mdf.loc[i].get(m, np.nan) for m in METRIC_ORDER]
    plt.bar(
        x + i*bar_width,
        vals,
        width=bar_width,
        color=cmap(i),
        label=f"Run {sid}"
    )

# plot the GPT-4o baseline as a thick dashed, black bar in the next slot
baseline_vals = [baseline[m] for m in METRIC_ORDER]
plt.bar(
    x + n_submissions*bar_width,
    baseline_vals,
    width=bar_width,
    color="black",
    label="GPT-4o Baseline",
    hatch="//",
    alpha=0.6
)

# ─────────────────────────────────────────────────────────────────────────────
# 6) Final formatting
# ─────────────────────────────────────────────────────────────────────────────
plt.xticks(
    ticks=x + (n_submissions/2)*bar_width,
    labels=METRIC_ORDER,
    rotation=30,
    ha="right"
)
plt.ylabel("Score")
plt.title("All Submission Metrics (grouped by metric)")
plt.ylim(0, max(max(baseline_vals), mdf[METRIC_ORDER].max().max()) * 1.15)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()
