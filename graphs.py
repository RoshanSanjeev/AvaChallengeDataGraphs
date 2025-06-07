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

# load & keep only “Finished”
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)
df   = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# pull out their metrics JSON
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# decide which metrics go on the x-axis
METRIC_ORDER = [
    "Final Score",
    "BLEU-4",
    "ROUGE-L",
    "Timing F1",
    "Timing AUC",
    "Action F1",
]

# build a list from most‐recent → oldest
n    = len(mdf)
subs = list(range(n, 0, -1))  # e.g. [n, n-1, ..., 1]

plt.figure(figsize=(10,6))
cmap = plt.get_cmap("tab20")

# now iterate: i=0→ newest, i=1→2nd newest, ..., label as Sub 1, Sub 2, ...
for i, sub in enumerate(subs):
    scores = [mdf.loc[sub-1].get(m, np.nan) for m in METRIC_ORDER]
    plt.plot(
        METRIC_ORDER,
        scores,
        marker="o",
        linestyle="-",
        color=cmap(i % 20),
        label=f"Sub {i+1}"
    )

plt.title("Submission Scores Across All Metrics")
plt.xlabel("Metric")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.grid(alpha=0.3)
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()
