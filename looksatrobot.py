import os
import pandas as pd
import shutil
import csv
import numpy as np

# ---------- Configuration ----------
source_excel     = "Shooter 2 Data.xlsx"
output_excel     = "Shooter 2 Proceed.xlsx"
folders          = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshook",
    os.path.join("noshook", "baseline")
]
id_col_excel     = 0
looks_col        = 7  # zero-based index for the 8th column

# Duplicate the workbook and load it
shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine="openpyxl")

# Ensure at least 8 columns and set header for the look count
while df.shape[1] <= looks_col:
    df[f"Extra_{df.shape[1]+1}"] = np.nan
df.iloc[:, looks_col] = np.nan
df.columns.values[looks_col] = "Robot Look Count"

# Lists to collect any issues
not_found_csv      = []
not_found_look_col = []
other_errors       = []

def find_matching_csv(index):
    """Return (full_path, folder_name) for the CSV whose name starts with index."""
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            if fn.lower().endswith(".csv") and fn.startswith(index):
                return os.path.join(folder, fn), folder
    return None, None

def find_column(header_row, target):
    """Return the first column index where target is a substring of the header, case-insensitive."""
    target = target.strip().lower()
    for i, col in enumerate(header_row):
        if isinstance(col, str) and target in col.strip().lower():
            return i
    return -1

# Process each row in the DataFrame
for row_idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    # Zero-pad integer IDs to 5 digits
    if isinstance(raw_id, float) and raw_id.is_integer():
        index_str = str(int(raw_id)).zfill(5)
    else:
        index_str = str(raw_id).zfill(5)

    csv_path, folder = find_matching_csv(index_str)
    if not csv_path:
        print(f"⚠️ {index_str}: no CSV found")
        not_found_csv.append(index_str)
        continue

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                print(f"⚠️ {index_str}: CSV is empty")
                not_found_look_col.append(index_str)
                continue

            look_idx = find_column(header, "lookingat")
            if look_idx == -1:
                print(f"⚠️ {index_str}: 'LookingAt' column not found")
                not_found_look_col.append(index_str)
                continue

            robot_looks = 0
            in_look     = False
            for r in reader:
                if len(r) <= look_idx:
                    in_look = False
                    continue
                val = r[look_idx].strip().lower()
                if val == "robot":
                    if not in_look:
                        robot_looks += 1
                        in_look = True
                else:
                    in_look = False

            df.iat[row_idx, looks_col] = robot_looks
            print(f"✅ {index_str}: Robot look count = {robot_looks}")

    except Exception as e:
        print(f"❌ {index_str}: error ({e})")
        other_errors.append(index_str)

# Save the updated workbook
df.to_excel(output_excel, index=False, engine="openpyxl")

# Print any issues encountered
if not_found_csv:
    print("\nMissing CSV for indices:", ", ".join(not_found_csv))
if not_found_look_col:
    print("Missing 'LookingAt' column in:", ", ".join(not_found_look_col))
if other_errors:
    print("Errors processing indices:", ", ".join(other_errors))

print("\nDone! Robot look count saved in column 8 of Shooter 2 Proceed.xlsx.")