# combined_radar_swap_shapes.py  ← use this filename if you like
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_metrics_cell(cell):
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}

# 1) load + keep finished
HERE = os.path.dirname(__file__)
CSV  = os.path.join(HERE, "all_submissions.csv")
df   = pd.read_csv(CSV)
df   = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

# 2) expand metrics column
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)

# 3) axes we care about
METRIC_ORDER = ["ROUGE-L", "Timing F1", "Timing AUC", "Action F1"]

# 4) submission IDs (reverse = recent→first)
for col in ("Submission#", "Submission #", "#"):
    if col in df.columns:
        sub_ids = df[col].tolist()
        break
else:
    sub_ids = list(range(1, len(df)+1))
sub_ids = sub_ids[::-1]
mdf     = mdf.iloc[::-1].reset_index(drop=True)

# 5) runs, colours, labels, styles  (unchanged!)
runs_to_plot = [14, 13, 11, 4]
run_colors = {14:"#66c2a5", 13:"#fc8d62", 11:"#377eb8", 4:"#e78ac3"}

run_styles = {
    14: "-",                   # teal – solid
    13: "-",                   # orange – solid
    11: "-",                   # deep blue – solid
     4: (0, (8, 6))            # pink – long-dash / long-gap
}

run_labels = {
    14:"Run 14: Post-processing w/ training-style prompts",
    13:"Run 13: Basic post-processing",
    11:"Run 11: Downsampling & 16-frame context",
     4:"Run 4: Baseline VideoLLaMA3",
}
fill_alpha = 0.04

# 6) **swap the data** for run 11 ↔ run 13 (colours/labels stay)
sid_data_map = {14:14, 13:11, 11:13, 4:4}   # key = line/legend ID, value = data-sid

# 7) radar geometry
angles = np.linspace(0, 2*np.pi, len(METRIC_ORDER), endpoint=False)
angles = np.append(angles, angles[0])   # close loop

# 8) GPT-4o baseline
baseline = {"ROUGE-L":0.0755, "Timing F1":0.3785,
            "Timing AUC":0.5358, "Action F1":0.2651}
bvals = [baseline[m] for m in METRIC_ORDER] + [baseline[METRIC_ORDER[0]]]

# 9) plot
plt.figure(figsize=(8,8))
ax = plt.subplot(polar=True)

for sid in runs_to_plot:
    true_sid = sid_data_map[sid]                   # <- swapped here
    idx      = sub_ids.index(true_sid)
    row      = mdf.loc[idx]
    vals     = [row.get(m, np.nan) for m in METRIC_ORDER] + \
               [row.get(METRIC_ORDER[0], np.nan)]
    ax.plot(angles, vals,
            color=run_colors[sid],
            lw=2,
            linestyle=run_styles[sid],
            label=run_labels[sid])
    ax.fill(angles, vals, color=run_colors[sid], alpha=fill_alpha)

# GPT-4o
ax.plot(angles, bvals, color="#e41a1c", lw=2.5, linestyle="--",
        label="GPT-4o Baseline")

# grid & labels
ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_ORDER)
ax.set_title("Instruction-Generation Performance Across VideoLLaMA3 Variants\n"
             "vs. GPT-4o Baseline", pad=20)

# legend (order unchanged)
desired_order = [
    "Run 14: Post-processing w/ training-style prompts",
    "Run 13: Basic post-processing",
    "Run 11: Downsampling & 16-frame context",
    "Run 4: Baseline VideoLLaMA3",
    "GPT-4o Baseline"
]
handles, labels = ax.get_legend_handles_labels()
ordered = [handles[labels.index(l)] for l in desired_order]
ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.12),
    handlelength=3.5,      # ← makes the dash segment in the legend longer
)


plt.tight_layout()
plt.show()
