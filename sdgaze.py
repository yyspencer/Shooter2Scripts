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
sd_gaze_col = 13  # 15th column

shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine='openpyxl')

# Ensure at least 15 columns, and set header for SD gaze
while df.shape[1] < 15:
    df[f"Extra_{df.shape[1]+1}"] = np.nan
df.iloc[:, sd_gaze_col] = np.nan
df.columns.values[sd_gaze_col] = "SD Gaze [x,y,z]"

not_found_csv = []
not_found_gaze_col = []
other_errors = []

def find_matching_csv(index):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".csv") and filename[:5] == index:
                return os.path.join(folder, filename), folder
    return None, None

def find_gaze_columns(header_row):
    # Returns (x_col, y_col, z_col) or (-1, -1, -1) if missing
    x_col = y_col = z_col = -1
    for i, col in enumerate(header_row):
        col_clean = col.strip()
        if col_clean == "Gaze Visualizer.x":
            x_col = i
        elif col_clean == "Gaze Visualizer.y":
            y_col = i
        elif col_clean == "Gaze Visualizer.z":
            z_col = i
    return x_col, y_col, z_col

for idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    if isinstance(raw_id, float) and raw_id.is_integer():
        index_str = str(int(raw_id)).zfill(5)
    else:
        index_str = str(raw_id).zfill(5)
    csv_path, found_folder = find_matching_csv(index_str)
    if not csv_path or not found_folder:
        df.iat[idx, sd_gaze_col] = np.nan
        not_found_csv.append((index_str, found_folder if found_folder else "not found"))
        continue

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None:
                df.iat[idx, sd_gaze_col] = np.nan
                not_found_gaze_col.append((index_str, found_folder))
                continue
            x_col, y_col, z_col = find_gaze_columns(header)
            if x_col == -1 or y_col == -1 or z_col == -1:
                df.iat[idx, sd_gaze_col] = np.nan
                not_found_gaze_col.append((index_str, found_folder))
                continue
            gaze_x, gaze_y, gaze_z = [], [], []
            for rowdata in reader:
                if (len(rowdata) <= max(x_col, y_col, z_col) or
                        rowdata[x_col] == "" or rowdata[y_col] == "" or rowdata[z_col] == ""):
                    continue
                try:
                    gaze_x.append(float(rowdata[x_col]))
                    gaze_y.append(float(rowdata[y_col]))
                    gaze_z.append(float(rowdata[z_col]))
                except ValueError:
                    continue
            # Calculate sample standard deviation (ddof=1)
            if len(gaze_x) < 2 or len(gaze_y) < 2 or len(gaze_z) < 2:
                df.iat[idx, sd_gaze_col] = np.nan
                other_errors.append((index_str, found_folder))
                continue
            sd_x = np.std(gaze_x, ddof=1)
            sd_y = np.std(gaze_y, ddof=1)
            sd_z = np.std(gaze_z, ddof=1)
            # Write as string
            df.iat[idx, sd_gaze_col] = f"[{sd_x:.6f}, {sd_y:.6f}, {sd_z:.6f}]"
    except Exception as e:
        df.iat[idx, sd_gaze_col] = np.nan
        other_errors.append((index_str, found_folder))

def print_issue_list(issue_list, description):
    if issue_list:
        print(f"\n{description} ({len(issue_list)}):")
        for index, folder in issue_list:
            print(f"{index}  {folder}")

df.to_excel(output_excel, index=False, engine='openpyxl')
print_issue_list(not_found_csv, "IDs with NO matching CSV file in any folder")
print_issue_list(not_found_gaze_col, "IDs where Gaze columns NOT FOUND in CSV")
print_issue_list(other_errors, "IDs with OTHER errors or not enough data for SD")

print("Done! Gaze SD [x, y, z] calculated and saved in column 15.")