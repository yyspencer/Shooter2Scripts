import os
import pandas as pd
import shutil
import csv
import numpy as np

# ---------- Configuration ----------
source_excel = "Shooter 2 Data.xlsx"
output_excel = "Shooter 2 Proceed.xlsx"
folders = [
    os.path.join("shook"),
    os.path.join("shook", "baseline"),
    os.path.join("noshook"),
    os.path.join("noshook", "baseline")
]
id_col_excel = 0
looks_col = 12  # 13th column (zero-based index)

# Duplicate and load
shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine='openpyxl')

# Ensure enough columns
while df.shape[1] <= looks_col:
    df[f"Extra_{df.shape[1]+1}"] = np.nan
df.iloc[:, looks_col] = np.nan
df.columns.values[looks_col] = "Signage Look Count"

not_found_csv = []
not_found_look_col = []
other_errors = []

def find_matching_csv(index):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv") and filename[:5] == index:
                return os.path.join(folder, filename), folder
    return None, None

def find_column(header_row, target):
    """Finds first column where 'target' is a substring of the header (case-insensitive)."""
    t = target.strip().lower()
    for idx, col in enumerate(header_row):
        if isinstance(col, str) and t in col.strip().lower():
            return idx
    return -1

for idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    index_str = str(int(raw_id)).zfill(5) if isinstance(raw_id, float) and raw_id.is_integer() else str(raw_id).zfill(5)
    csv_path, found_folder = find_matching_csv(index_str)
    if not csv_path or not found_folder:
        df.iat[idx, looks_col] = np.nan
        print(f"{index_str}: Signage looks=None (no csv found)")
        not_found_csv.append((index_str, found_folder or "not found"))
        continue

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None:
                df.iat[idx, looks_col] = np.nan
                print(f"{index_str}: Signage looks=None (empty file)")
                not_found_look_col.append((index_str, found_folder))
                continue

            lookingat_col = find_column(header, "lookingat")
            if lookingat_col == -1:
                df.iat[idx, looks_col] = np.nan
                print(f"{index_str}: Signage looks=None (LookingAt column not found)")
                not_found_look_col.append((index_str, found_folder))
                continue

            signage_looks = 0
            in_look = False
            for rowdata in reader:
                if len(rowdata) <= lookingat_col:
                    in_look = False
                    continue
                looking_val = rowdata[lookingat_col]
                # Case-insensitive "Signage" substring check (including e.g. "Signage_1", "signageA", etc.)
                if isinstance(looking_val, str) and looking_val.strip().lower().startswith("signage"):
                    if not in_look:
                        signage_looks += 1
                        in_look = True
                else:
                    in_look = False

            df.iat[idx, looks_col] = signage_looks
            print(f"{index_str}: Signage_look_count={signage_looks}")
    except Exception as e:
        df.iat[idx, looks_col] = np.nan
        print(f"{index_str}: Signage looks=None (error reading csv) — {e}")
        other_errors.append((index_str, found_folder))

def print_issue_list(issue_list, description):
    if issue_list:
        print(f"\n{description} ({len(issue_list)}):")
        for index, folder in issue_list:
            print(f"{index}  {folder}")

df.to_excel(output_excel, index=False, engine='openpyxl')
print_issue_list(not_found_csv, "IDs with NO matching CSV file in any folder")
print_issue_list(not_found_look_col, "IDs where 'LookingAt' column NOT FOUND in CSV")
print_issue_list(other_errors, "IDs with OTHER errors (possibly corrupted CSV)")

print("Done! Signage look count calculated and saved in column", looks_col+1)