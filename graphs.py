 # graphs.py
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV = "all_submissions.csv"   # <-- make sure this matches your filename

def parse_metrics_cell(cell):
    # each cell is like "[{'BLEU-4': ..., 'ROUGE-L': ..., ...}]"
    # convert single-quotes to double-quotes so json.loads will parse
    try:
        arr = json.loads(cell.replace("'", '"'))
        return arr[0]
    except Exception:
        return {}

def main():
    # 1) load and filter
    df = pd.read_csv(CSV)
    df = df[df["Status"].str.lower() == "finished"].reset_index(drop=True)

    # 2) extract metrics dict from the JSON snippet
    metrics = df["Result File"].apply(parse_metrics_cell)
    mdf = pd.json_normalize(metrics)  # gives columns like BLEU-4, ROUGE-L, Timing F1, etc.

    # 3) build an x-axis of 1,2,3...
    x = np.arange(1, len(df) + 1)

    # 4) plot BLEU-4 by itself
    plt.figure(figsize=(8, 3))
    plt.plot(x, mdf["BLEU-4"], marker="o", linestyle="-", label="BLEU-4")
    plt.title("BLEU-4 over Submissions")
    plt.xlabel("Submission #")
    plt.ylabel("BLEU-4 score")
    plt.xticks(x[::2])           # show every 2 ticks
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 5) plot all the other metrics together
    plt.figure(figsize=(8, 4))
    others = ["ROUGE-L", "Timing F1", "Timing AUC", "Action F1", "Final Score"]
    markers = ["s", "^", "x", "d", "o"]
    for name, mk in zip(others, markers):
        if name in mdf:
            plt.plot(x, mdf[name], marker=mk, linestyle="-", label=name)

    plt.title("Other Metrics over Submissions")
    plt.xlabel("Submission #")
    plt.ylabel("Score")
    plt.xticks(x[::2])
    plt.legend(loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
