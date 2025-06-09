# combined_radar_swap_shapes.py
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
def parse_metrics_cell(cell):
    """Parse the one-element JSON list stored in the Result-File column."""
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# 1) load + keep finished
# HERE = os.path.dirname()
CSV  = os.path.join("all_submissions.csv")
df   = pd.read_csv(CSV)
df   = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

print(df["Result File"])

# 2) expand metrics column
metrics = df["Result File"].apply(parse_metrics_cell)
mdf     = pd.json_normalize(metrics)
# mdf     = mdf.rename(columns={"Timing AUC": "Timing\nAUC"})  # <--- Insert here


# 3) axes we care about
METRIC_ORDER = ["ROUGE-L", "Timing F1", "Timing\nAUC", "Action F1"]

# 4) submission IDs (reverse = recent→first)
for col in ("Submission#", "Submission #", "#"):
    if col in df.columns:
        sub_ids = df[col].tolist()
        break
else:
    sub_ids = list(range(1, len(df)+1))

sub_ids = sub_ids[::-1]
mdf     = mdf.iloc[::-1].reset_index(drop=True)

# 5) runs, colours, labels, styles
runs_to_plot = [14, 13, 11, 4]
run_colors   = {14:"#66c2a5", 13:"#fc8d62", 11:"#377eb8", 4:"#e78ac3"}
run_styles   = {
    14: "-",                  # teal – solid
    13: "-",                  # orange – solid
    11: "-",                  # deep blue – solid
     4: (0, (8, 6))           # pink – long-dash / long-gap
}

run_labels = {
    14: "PE",
    13: "PP",
    11: "DS",
     4: "MB",            # UPDATED
}
fill_alpha = 0.04

# 6) *swap* the data for run 11 ↔ run 13 (colours/labels stay)
sid_data_map = {14:14, 13:11, 11:13, 4:4}

# 7) radar geometry
angles = np.linspace(0, 2*np.pi, len(METRIC_ORDER), endpoint=False)
angles = np.append(angles, angles[0])   # close loop

# 8) GPT-4o baseline
baseline = {"ROUGE-L":0.0755, "Timing F1":0.3785,
            "Timing\nAUC":0.5358, "Action F1":0.2651}
bvals = [baseline[m] for m in METRIC_ORDER] + [baseline[METRIC_ORDER[0]]]

# 9) plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(polar=True)

for sid in runs_to_plot:
    true_sid = sid_data_map[sid]                # swapped here
    idx      = sub_ids.index(true_sid)
    row      = mdf.loc[idx]
    vals = [row.get(m.replace("\n", " "), np.nan) for m in METRIC_ORDER] + \
       [row.get(METRIC_ORDER[0].replace("\n", " "), np.nan)]


    ax.plot(angles, vals,
            color=run_colors[sid],
            lw=2,
            linestyle=run_styles[sid],
            label=run_labels[sid])
    ax.fill(angles, vals, color=run_colors[sid], alpha=fill_alpha)

# GPT-4o dashed baseline
ax.plot(angles, bvals, color="#e41a1c", lw=2.5, linestyle="--",
        label="GPT-4(Baseline)")

# grid & labels
ax.set_thetagrids(np.degrees(angles[:-1]), METRIC_ORDER, fontsize=24)
angles_deg = np.degrees(angles[:-1])  # Get the angles in degrees

for label, angle in zip(ax.get_xticklabels(), angles_deg):
    # Adjust only if the angle is near 0 or 180 degrees (horizontal)
    if abs(angle % 360) < 20 or abs((angle % 360) - 180) < 20:
        label.set_y(label.get_position()[1] - 0.15)  # More padding for horizontal
    else:
        label.set_y(label.get_position()[1] - 0.08)
ax.set_title("Instruction-Generation Performance Across Solutions", pad=50, fontsize= 30, fontweight='bold')

# legend (just reorder, labels already correct)
desired_order = [
    "PE",
    "PP",
    "DS",
    "MB",               # UPDATED
    "GPT-4(Baseline)"
]
handles, labels = ax.get_legend_handles_labels()
ordered_handles = [handles[labels.index(l)] for l in desired_order]

ax.legend(
    ordered_handles,
    desired_order,
    loc="upper left",                # Anchor legend's upper-left corner
    bbox_to_anchor=(0.65, 0.25),     # Increase 1.05 to move further right
    handlelength=3.5,
    frameon=True,
    fancybox=False,
    fontsize=20
)

# Set solid white background for the legend
legend = ax.get_legend()
legend.get_frame().set_facecolor("white")
# legend.set_facecolor("white")

plt.tight_layout()
plt.show()