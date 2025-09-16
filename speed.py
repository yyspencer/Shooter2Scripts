import os
import pandas as pd
import numpy as np

# ---------------- CONFIGURATION ----------------
EXCEL_FILE = "Shooter 2 Data.xlsx"
ID_COL = 0  # First column for index in Excel
FOLDERS = [
    os.path.join("shook"),
    os.path.join("shook", "baseline"),
    os.path.join("noshook"),
    os.path.join("noshook", "baseline"),
]
OUTPUT_DIR = "speed"

PLAYER_COLS = ["playervr.x", "playervr.y", "playervr.z"]
ROBOT_COLS  = ["robot.x", "robot.y", "robot.z"]
TIME_COL = "time"
ROBOT_EVENT_COL = "robotevent"
ROOM_EVENT_COL  = "roomevent"

def normalize(s):
    return str(s).strip().lower()

def find_col(df, name):
    name = normalize(name)
    for col in df.columns:
        if normalize(col) == name:
            return col
    return None

def find_matching_file(index):
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.lower().endswith(".csv") and f[:5] == index:
                return os.path.join(folder, f)
    return None

def euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Step 1: Read Excel and get zero-padded indices
df_excel = pd.read_excel(EXCEL_FILE, engine="openpyxl")
indices = df_excel.iloc[:, ID_COL].apply(
    lambda v: str(int(v)).zfill(5) if (isinstance(v, (float, int)) and float(v).is_integer()) else str(v).zfill(5)
)

for idx in indices:
    csv_path = find_matching_file(idx)
    if not csv_path:
        print(f"⚠️  No CSV found for index {idx}")
        continue

    try:
        # More robust: skip malformed lines, do not care about number of columns
        df = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
    except Exception as e:
        print(f"❌ Failed to read CSV {csv_path}: {e}")
        continue

    df.columns = [normalize(c) for c in df.columns]

    # Find each column one by one
    missing_column = None
    time_col = find_col(df, TIME_COL)
    if time_col is None:
        missing_column = TIME_COL
    robot_event_col = find_col(df, ROBOT_EVENT_COL)
    if robot_event_col is None:
        missing_column = ROBOT_EVENT_COL
    room_event_col = find_col(df, ROOM_EVENT_COL)
    if room_event_col is None:
        missing_column = ROOM_EVENT_COL

    player_cols = []
    for c in PLAYER_COLS:
        found = find_col(df, c)
        if found is None and missing_column is None:
            missing_column = c
        player_cols.append(found)
    robot_cols = []
    for c in ROBOT_COLS:
        found = find_col(df, c)
        if found is None and missing_column is None:
            missing_column = c
        robot_cols.append(found)

    # If any missing, output index and column, and skip this file
    all_needed = [time_col, robot_event_col, room_event_col] + player_cols + robot_cols
    if any(c is None for c in all_needed):
        print(f"Missing column for index {idx}: {missing_column}")
        continue

    try:
        times = df[time_col].astype(float).values
        player_xyz = df[player_cols].astype(float).values
        robot_xyz  = df[robot_cols].astype(float).values
    except Exception as e:
        print(f"❌ Failed to parse positions/time in {csv_path}: {e}")
        continue

    robot_event = df[robot_event_col].astype(str).values
    room_event  = df[room_event_col].astype(str).values
    n = len(df)

    # Survey room interval detection
    in_survey = np.zeros(n, dtype=bool)
    entry_idx = []
    exit_idx  = []
    for i, evt in enumerate(room_event):
        evt_norm = normalize(evt)
        if "robot entered survey room" in evt_norm:
            entry_idx.append(i)
        elif "robot exited survey room" in evt_norm:
            exit_idx.append(i)
    for start, end in zip(entry_idx, exit_idx):
        if start < end:
            in_survey[start:end+1] = True

    # Find crisis row
    crisis_row = None
    folder_type = os.path.normpath(csv_path).split(os.sep)[0].lower()
    if folder_type.startswith("shook"):
        for i, evt in enumerate(robot_event):
            if "shook" in normalize(evt):
                crisis_row = i
                break
    else:
        ref_time = None
        for i, evt in enumerate(robot_event):
            if "0.2 seconds" in normalize(evt):
                try:
                    ref_time = times[i] + 0.229
                except:
                    continue
                break
        if ref_time is not None:
            for i in range(n):
                if times[i] >= ref_time:
                    crisis_row = i
                    break

    if crisis_row is None:
        print(f"❌ No crisis event found in {csv_path}. Skipping index {idx}.")
        continue

    # Build valid row mask
    mask_valid = ~in_survey
    mask_valid[1:] &= (times[1:] > times[:-1])
    for arr in [player_xyz, robot_xyz]:
        mask_valid &= (arr != -1).all(axis=1)

    # Gather speed data
    speed_data = []
    prev_idx = None
    for i in range(n):
        if not mask_valid[i]:
            continue
        if prev_idx is None:
            prev_idx = i
            continue
        if i - prev_idx != 1:
            prev_idx = i
            continue
        dt = times[i] - times[prev_idx]
        if dt <= 0:
            prev_idx = i
            continue
        player_speed = euclidean(player_xyz[i], player_xyz[prev_idx]) / dt
        robot_speed  = euclidean(robot_xyz[i],  robot_xyz[prev_idx])  / dt
        speed_data.append((i, player_speed, robot_speed))
        prev_idx = i

    if not speed_data:
        print(f"❌ No valid speed data for index {idx}")
        continue

    # Write output file with header
    out_path = os.path.join(OUTPUT_DIR, f"{idx}.txt")
    with open(out_path, "w") as fout:
        fout.write("playerSpeed robotSpeed\n")
        crisis_written = False
        skip_next = False
        for j, (i, p_speed, r_speed) in enumerate(speed_data):
            if skip_next:
                skip_next = False
                continue
            fout.write(f"{p_speed} {r_speed}\n")
            if not crisis_written and i >= crisis_row:
                fout.write("\n")  # Empty line after crisis row
                crisis_written = True
                skip_next = True  # Skip first post-crisis line
        print(f"✅ Wrote {out_path}")

print("\nAll done.")