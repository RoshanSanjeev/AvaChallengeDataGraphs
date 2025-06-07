# combined_plot_rev.py

import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_cell(cell):
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}

# ──────────────────────────────────────────────────────────────────────────────
# 1) load & keep only finished runs
# ──────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)
df   = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) parse the JSON‐snippet into a metrics DataFrame
# ──────────────────────────────────────────────────────────────────────────────
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# ──────────────────────────────────────────────────────────────────────────────
# 3) define which metrics and in what order to plot them on the x‐axis
# ──────────────────────────────────────────────────────────────────────────────
METRIC_ORDER = [
    "Final Score",
    "BLEU-4",
    "ROUGE-L",
    "Timing F1",
    "Timing AUC",
    "Action F1",
]

# ──────────────────────────────────────────────────────────────────────────────
# 4) extract submission IDs (or fallback to a simple counter)
# ──────────────────────────────────────────────────────────────────────────────
for col in ("Submission#", "Submission #", "#"):
    if col in df.columns:
        sub_ids = df[col].tolist()
        break
else:
        sub_ids = list(range(1, len(df) + 1))

# ──────────────────────────────────────────────────────────────────────────────
# 5) reverse so index 0 is the most recent
# ──────────────────────────────────────────────────────────────────────────────
sub_ids = sub_ids[::-1]
mdf     = mdf.iloc[::-1].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6) define the GPT-4o baseline metrics
# ──────────────────────────────────────────────────────────────────────────────
BASELINE_NAME    = "GPT-4o_Baseline"
BASELINE_METRICS = {
    "Final Score": 0.2651,
    "BLEU-4":      0.0000,
    "ROUGE-L":     0.0755,
    "Timing F1":   0.3785,
    "Timing AUC":  0.5358,
    "Action F1":   0.3355,
}

baseline_scores = [BASELINE_METRICS[m] for m in METRIC_ORDER]

# ──────────────────────────────────────────────────────────────────────────────
# 7) plot everything
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
cmap = plt.get_cmap("tab20")

# plot each submission
for i, sid in enumerate(sub_ids):
    row    = mdf.loc[i]
    scores = [row.get(m, np.nan) for m in METRIC_ORDER]

    plt.plot(
        METRIC_ORDER,
        scores,
        marker="o",
        linestyle="-",
        color=cmap(i % 20),
        label=f"Run {sid}"
    )

# overlay the baseline
plt.plot(
    METRIC_ORDER,
    baseline_scores,
    marker="s",
    linestyle="--",
    color="k",
    linewidth=2,
    label=BASELINE_NAME
)

plt.title("VideoLlama3 Performance")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.grid(alpha=0.3, linestyle=":")
plt.legend(title="Legend", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()
