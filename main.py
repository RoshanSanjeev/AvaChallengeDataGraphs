# main.py
import pandas as pd

# Since main.py and all_submissions.csv are in the same folder,
# just point pandas at the CSV filename (no extra folder prefix).
csv_path = "all_submissions.csv"
df = pd.read_csv(csv_path)

# Print a quick preview and list of columns
print(df.head())
print("Columns:", df.columns.tolist())
