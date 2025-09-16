import os
import pandas as pd
import shutil
import csv
import numpy as np

# ---------- Configuration ----------
source_excel = "Shooter 2 Data.xlsx"
output_excel = "Shooter 2 Data Proceed.xlsx"

folders = [
    os.path.join("shook"),
    os.path.join("shook", "baseline"),
    os.path.join("noshookmodified"),
    os.path.join("noshookmodified", "baseline"),
]

id_col_excel = 0
output_col_excel = 4  # write crisis time here

# ---------- Duplicate Excel File ----------
shutil.copyfile(source_excel, output_excel)

# ---------- Load Excel ----------
df = pd.read_excel(output_excel, engine='openpyxl')

# ---------- Error Trackers ----------
not_found_csv = []
not_found_column = []
not_found_keyword = []
other_errors = []

# ---------- Helpers ----------
def classify_folder(folder_path: str) -> str:
    """Return 'shook', 'noshookmodified', 'noshook', or 'unknown'."""
    f = os.path.normpath(folder_path).lower()
    # order matters: check 'noshookmodified' before 'noshook'
    if "noshookmodified" in f:
        return "noshookmodified"
    if "noshook" in f:
        return "noshook"
    if "shook" in f:
        return "shook"
    return "unknown"

def find_matching_csv(index_5: str):
    """Return (csv_path, folder_type|folder_string) or (None, None)."""
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            if filename.lower().endswith(".csv") and filename[:5] == index_5:
                return os.path.join(folder, filename), classify_folder(folder)
    return None, None

def find_robot_event_col(header_row):
    for idx, col in enumerate(header_row):
        if "robotevent" in col.strip().lower():
            return idx
    return -1

def get_crisis_time(filepath, folder_type):
    """
    Determine crisis time from the robotEvent column:
      - shook: look for substring 'shook' → use that row's time
      - noshook: look for substring '0.2 seconds' → time + 0.229
      - noshookmodified: look for substring 'shook' (e.g., 'estimated shook');
                         if not found, also accept '0.2 seconds' → time + 0.229
    Returns (crisis_time, error_type or None)
    """
    try:
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header is None:
                return None, "no_header"

            event_col = find_robot_event_col([col.strip() for col in header])
            if event_col == -1:
                return None, "no_column"

            for row in reader:
                if len(row) <= event_col or not row:
                    continue
                # time assumed to be first column, as in your original script
                try:
                    time_val = float(row[0])
                except Exception:
                    continue

                ev = row[event_col]
                ev_norm = (ev or "").strip().lower()

                if folder_type == "shook":
                    if "shook" in ev_norm:
                        return time_val, None

                elif folder_type == "noshook":
                    if "0.2 seconds" in ev_norm:
                        return round(time_val + 0.229, 6), None

                elif folder_type == "noshookmodified":
                    # first prefer explicit 'shook' (e.g., 'estimated shook')
                    if "shook" in ev_norm:
                        return time_val, None
                    # fallback if someone left the old marker
                    if "0.2 seconds" in ev_norm:
                        return round(time_val + 0.229, 6), None

            return None, "no_keyword"

    except Exception:
        return None, "other"

# ---------- Main Processing ----------
for idx, row in df.iterrows():
    raw_id = row.iloc[id_col_excel]
    # Normalize to 5-char index
    if isinstance(raw_id, float) and raw_id.is_integer():
        index_str = str(int(raw_id)).zfill(5)
    else:
        index_str = str(raw_id).zfill(5)

    csv_path, folder_type = find_matching_csv(index_str)
    folder_report = folder_type if folder_type is not None else "not found"

    if not csv_path or not folder_type:
        df.iat[idx, output_col_excel] = np.nan
        not_found_csv.append((index_str, folder_report))
        continue

    crisis_time, error_type = get_crisis_time(csv_path, folder_type)

    if crisis_time is not None:
        df.iat[idx, output_col_excel] = crisis_time
    else:
        df.iat[idx, output_col_excel] = np.nan
        if error_type == "no_column":
            not_found_column.append((index_str, folder_report))
        elif error_type == "no_keyword":
            not_found_keyword.append((index_str, folder_report))
        else:
            other_errors.append((index_str, folder_report))

# ---------- Save Excel ----------
df.to_excel(output_excel, index=False, engine='openpyxl')

# ---------- Report ----------
def print_issue_list(issue_list, description):
    if issue_list:
        print(f"\n{description} ({len(issue_list)}):")
        for index, folder in issue_list:
            print(f"{index}  {folder}")

print_issue_list(not_found_csv, "IDs with NO matching CSV file in any folder")
print_issue_list(not_found_column, "IDs where 'robotEvent' column NOT FOUND in CSV")
print_issue_list(not_found_keyword, "IDs where keyword was NOT FOUND in 'robotEvent' column")
print_issue_list(other_errors, "IDs with OTHER errors (possibly corrupted CSV)")

if not (not_found_csv or not_found_column or not_found_keyword or other_errors):
    print("All indices processed successfully!")