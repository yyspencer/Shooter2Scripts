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
mean_col = 14  # 16th column
sd_col = 15    # 17th column
max_col = 16   # 18th column
min_col = 17   # 19th column

shutil.copyfile(source_excel, output_excel)
df = pd.read_excel(output_excel, engine='openpyxl')

# Ensure at least 19 columns and set headers
while df.shape[1] < 19:
    df[f"Extra_{df.shape[1]+1}"] = np.nan
df.iloc[:, mean_col] = np.nan
df.iloc[:, sd_col] = np.nan
df.iloc[:, max_col] = np.nan
df.iloc[:, min_col] = np.nan
df.columns.values[mean_col] = "Mean Player-Robot Dist"
df.columns.values[sd_col] = "SD Player-Robot Dist"
df.columns.values[max_col] = "Max Player-Robot Dist"
df.columns.values[min_col] = "Min Player-Robot Dist"

not_found_csv = []
not_found_pos_col = []
other_errors = []

def find_matching_csv(index):
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.endswith(".csv") and filename[:5] == index:
                return os.path.join(folder, filename), folder
    return None, None

def find_player_robot_columns(header_row):
    # Trim headers and match against expected names
    trimmed = [col.strip() for col in header_row]
    try:
        px = trimmed.index("PlayerVR.x")
        py = trimmed.index("PlayerVR.y")
        pz = trimmed.index("PlayerVR.z")
        rx = trimmed.index("Robot.x")
        ry = trimmed.index("Robot.y")
        rz = trimmed.index("Robot.z")
        return px, py, pz, rx, ry, rz
    except ValueError:
        return -1, -1, -1, -1, -1, -1

for idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    if isinstance(raw_id, float) and raw_id.is_integer():
        index_str = str(int(raw_id)).zfill(5)
    else:
        index_str = str(raw_id).zfill(5)
    csv_path, found_folder = find_matching_csv(index_str)
    if not csv_path or not found_folder:
        for col in [mean_col, sd_col, max_col, min_col]:
            df.iat[idx, col] = np.nan
        not_found_csv.append((index_str, found_folder if found_folder else "not found"))
        continue

    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None:
                for col in [mean_col, sd_col, max_col, min_col]:
                    df.iat[idx, col] = np.nan
                not_found_pos_col.append((index_str, found_folder))
                continue
            px, py, pz, rx, ry, rz = find_player_robot_columns(header)
            if -1 in [px, py, pz, rx, ry, rz]:
                for col in [mean_col, sd_col, max_col, min_col]:
                    df.iat[idx, col] = np.nan
                not_found_pos_col.append((index_str, found_folder))
                continue
            # Read data rows
            distances = []
            for rowdata in reader:
                if len(rowdata) <= max(px, py, pz, rx, ry, rz):
                    continue
                try:
                    player_pos = np.array([
                        float(rowdata[px]),
                        float(rowdata[py]),
                        float(rowdata[pz])
                    ])
                    robot_pos = np.array([
                        float(rowdata[rx]),
                        float(rowdata[ry]),
                        float(rowdata[rz])
                    ])
                except Exception:
                    continue
                dist = np.linalg.norm(player_pos - robot_pos)
                distances.append(dist)
            # Calculate stats
            if len(distances) < 2:
                for col in [mean_col, sd_col, max_col, min_col]:
                    df.iat[idx, col] = np.nan
                other_errors.append((index_str, found_folder))
                continue
            mean_d = np.mean(distances)
            sd_d = np.std(distances, ddof=1)  # sample SD
            max_d = np.max(distances)
            min_d = np.min(distances)
            df.iat[idx, mean_col] = mean_d
            df.iat[idx, sd_col] = sd_d
            df.iat[idx, max_col] = max_d
            df.iat[idx, min_col] = min_d
    except Exception as e:
        for col in [mean_col, sd_col, max_col, min_col]:
            df.iat[idx, col] = np.nan
        other_errors.append((index_str, found_folder))

def print_issue_list(issue_list, description):
    if issue_list:
        print(f"\n{description} ({len(issue_list)}):")
        for index, folder in issue_list:
            print(f"{index}  {folder}")

df.to_excel(output_excel, index=False, engine='openpyxl')
print_issue_list(not_found_csv, "IDs with NO matching CSV file in any folder")
print_issue_list(not_found_pos_col, "IDs where Player/Robot position columns NOT FOUND in CSV")
print_issue_list(other_errors, "IDs with OTHER errors or not enough data for stats")

print("Done! Player-Robot distance stats (mean, sd, max, min) saved in columns 16-19.")