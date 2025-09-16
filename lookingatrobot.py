import os
import pandas as pd
import shutil
import csv
import numpy as np

# ---------- Configuration ----------
source_excel     = "Shooter 2 Data.xlsx"
output_excel     = "Shooter 2 Proceed.xlsx"
folders          = [
    os.path.join("shook"),
    os.path.join("shook", "baseline"),
    os.path.join("noshook"),
    os.path.join("noshook", "baseline")
]
id_col_excel     = 0
look_percent_col = 9   # zero-based → 9th column in Excel

# Copy and prepare workbook
shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine='openpyxl')

# Make sure we have at least 9 columns
while df.shape[1] <= look_percent_col:
    df[f"Extra_{df.shape[1]+1}"] = np.nan

df.iloc[:, look_percent_col] = np.nan
df.columns.values[look_percent_col] = "% Looking At Robot"

not_found_csv       = []
not_found_look_col  = []
other_errors        = []

def find_matching_csv(index):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".csv") and filename.startswith(index):
                return os.path.join(folder, filename), folder
    return None, None

def find_column(header_row, target):
    """Return the first column index where target is a substring of the header."""
    target = target.strip().lower()
    for idx, col in enumerate(header_row):
        if isinstance(col, str) and target in col.strip().lower():
            return idx
    return -1

for i, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    index_str = (str(int(raw_id)).zfill(5)
                 if isinstance(raw_id, float) and raw_id.is_integer()
                 else str(raw_id).zfill(5))

    csv_path, folder = find_matching_csv(index_str)
    if not csv_path:
        print(f"⚠️ {index_str}: no CSV found")
        not_found_csv.append(index_str)
        continue

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                print(f"⚠️ {index_str}: empty CSV")
                not_found_look_col.append(index_str)
                continue

            # substring search for any header containing "lookingat"
            look_col = find_column(header, "lookingat")
            if look_col == -1:
                print(f"⚠️ {index_str}: 'lookingAt' column not found")
                not_found_look_col.append(index_str)
                continue

            robot_count = total = 0
            for r in reader:
                if len(r) <= look_col:
                    continue
                total += 1
                if r[look_col].strip().lower() == "robot":
                    robot_count += 1

            pct = (robot_count / total * 100) if total else np.nan
            df.iat[i, look_percent_col] = pct
            print(f"✅ {index_str}: {robot_count}/{total} → {pct:.2f}%")

    except Exception as e:
        print(f"❌ {index_str}: error ({e})")
        other_errors.append(index_str)

# Save
df.to_excel(output_excel, index=False, engine='openpyxl')

# Report issues
if not_found_csv:
    print("\nMissing CSV files for indices:", ", ".join(not_found_csv))
if not_found_look_col:
    print("Couldn't find 'lookingAt' column in:", ", ".join(not_found_look_col))
if other_errors:
    print("Errors processing indices:", ", ".join(other_errors))

print("\nDone! % Looking At Robot written in column 9.")  # user-visible column number