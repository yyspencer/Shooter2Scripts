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

# Column indices for output (21-28, 0-based index 20-27)
left_mean_col = 19
left_sd_col   = 20
left_max_col  = 21
left_min_col  = 22
right_mean_col = 23
right_sd_col   = 24
right_max_col  = 25
right_min_col  = 26

shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine='openpyxl')

# Ensure at least 28 columns and set headers
while df.shape[1] < 28:
    df[f"Extra_{df.shape[1]+1}"] = np.nan
df.iloc[:, left_mean_col]  = np.nan
df.iloc[:, left_sd_col]    = np.nan
df.iloc[:, left_max_col]   = np.nan
df.iloc[:, left_min_col]   = np.nan
df.iloc[:, right_mean_col] = np.nan
df.iloc[:, right_sd_col]   = np.nan
df.iloc[:, right_max_col]  = np.nan
df.iloc[:, right_min_col]  = np.nan

df.columns.values[left_mean_col]  = "Mean Left Pupil Size"
df.columns.values[left_sd_col]    = "SD Left Pupil Size"
df.columns.values[left_max_col]   = "Max Left Pupil Size"
df.columns.values[left_min_col]   = "Min Left Pupil Size"
df.columns.values[right_mean_col] = "Mean Right Pupil Size"
df.columns.values[right_sd_col]   = "SD Right Pupil Size"
df.columns.values[right_max_col]  = "Max Right Pupil Size"
df.columns.values[right_min_col]  = "Min Right Pupil Size"

not_found_csv = []
not_found_pupil_col = []
other_errors = []

def find_matching_csv(index):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".csv") and filename[:5] == index:
                return os.path.join(folder, filename), folder
    return None, None

def find_pupil_columns(header_row):
    # Trim whitespace from all header entries before comparing
    trimmed = [col.strip() for col in header_row]
    try:
        l_col = trimmed.index("leftPupilDiameter")
        r_col = trimmed.index("rightPupilDiameter")
        return l_col, r_col
    except ValueError:
        return -1, -1

for idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    if isinstance(raw_id, float) and raw_id.is_integer():
        index_str = str(int(raw_id)).zfill(5)
    else:
        index_str = str(raw_id).zfill(5)
    csv_path, found_folder = find_matching_csv(index_str)
    if not csv_path or not found_folder:
        for col in [left_mean_col, left_sd_col, left_max_col, left_min_col,
                    right_mean_col, right_sd_col, right_max_col, right_min_col]:
            df.iat[idx, col] = np.nan
        not_found_csv.append((index_str, found_folder if found_folder else "not found"))
        continue

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None:
                for col in [left_mean_col, left_sd_col, left_max_col, left_min_col,
                            right_mean_col, right_sd_col, right_max_col, right_min_col]:
                    df.iat[idx, col] = np.nan
                not_found_pupil_col.append((index_str, found_folder))
                continue
            l_col, r_col = find_pupil_columns(header)
            if l_col == -1 or r_col == -1:
                for col in [left_mean_col, left_sd_col, left_max_col, left_min_col,
                            right_mean_col, right_sd_col, right_max_col, right_min_col]:
                    df.iat[idx, col] = np.nan
                not_found_pupil_col.append((index_str, found_folder))
                continue
            left_vals = []
            right_vals = []
            for rowdata in reader:
                if len(rowdata) <= max(l_col, r_col):
                    continue
                try:
                    if rowdata[l_col] != "":
                        left_vals.append(float(rowdata[l_col]))
                except Exception:
                    pass
                try:
                    if rowdata[r_col] != "":
                        right_vals.append(float(rowdata[r_col]))
                except Exception:
                    pass
            # Left
            if len(left_vals) >= 2:
                df.iat[idx, left_mean_col] = np.mean(left_vals)
                df.iat[idx, left_sd_col]   = np.std(left_vals, ddof=1)
                df.iat[idx, left_max_col]  = np.max(left_vals)
                df.iat[idx, left_min_col]  = np.min([v for v in left_vals if v >= 0]) if any(v >= 0 for v in left_vals) else np.nan
            else:
                for col in [left_mean_col, left_sd_col, left_max_col, left_min_col]:
                    df.iat[idx, col] = np.nan
            # Right
            if len(right_vals) >= 2:
                df.iat[idx, right_mean_col] = np.mean(right_vals)
                df.iat[idx, right_sd_col]   = np.std(right_vals, ddof=1)
                df.iat[idx, right_max_col]  = np.max(right_vals)
                df.iat[idx, right_min_col] = np.min([v for v in right_vals if v >= 0]) if any(v >= 0 for v in right_vals) else np.nan
            else:
                for col in [right_mean_col, right_sd_col, right_max_col, right_min_col]:
                    df.iat[idx, col] = np.nan
    except Exception as e:
        for col in [left_mean_col, left_sd_col, left_max_col, left_min_col,
                    right_mean_col, right_sd_col, right_max_col, right_min_col]:
            df.iat[idx, col] = np.nan
        other_errors.append((index_str, found_folder))

def print_issue_list(issue_list, description):
    if issue_list:
        print(f"\n{description} ({len(issue_list)}):")
        for index, folder in issue_list:
            print(f"{index}  {folder}")

df.to_excel(output_excel, index=False, engine='openpyxl')
print_issue_list(not_found_csv, "IDs with NO matching CSV file in any folder")
print_issue_list(not_found_pupil_col, "IDs where pupil columns NOT FOUND in CSV")
print_issue_list(other_errors, "IDs with OTHER errors or not enough data for stats")

print("Done! Left/Right pupil stats (mean, sd, max, min) saved in columns 21-28.")