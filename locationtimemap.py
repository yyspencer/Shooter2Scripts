#!/usr/bin/env python3
import os
import re
import csv
import unicodedata
import numpy as np
import pandas as pd

# ----------------- Config (Shooter study) -----------------
EXCEL_FILE = "Shooter 2 Data.xlsx"
ID_COL = 0  # first column holds indices

# Search ONLY these top-level folders (no baselines)
SEARCH_FOLDERS = [
    "shook",
    "noshook",
]

ENTER_TAG = "Robot entered Survey Room"
EXIT_TAG  = "Robot exited Survey Room"

# Shooter crisis sphere stats (ABS positions), from your results:
CRISIS_MEAN = np.array([28.965761, 0.0, 47.078999], dtype=float)
CRISIS_VAR  = np.array([0.002806, 0.0, 0.004117], dtype=float)
CRISIS_STD  = np.sqrt(CRISIS_VAR)
CRISIS_RADIUS = 2.0 * float(np.linalg.norm(CRISIS_STD))  # Euclidean radius = 2σ

# Stillness tolerance (meters) for float jitter
EPS = 1e-6

# Keep ONLY deltas within this closed interval [min, max] seconds; others are outliers
DELTA_KEEP_MIN = -6.0
DELTA_KEEP_MAX = -4.0

# ----------------- Text/Regex Helpers -----------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower().replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", str(s).strip().lower())

def find_column_indices_relaxed(header, want_keys):
    normed = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        idx = -1
        for j, h in enumerate(normed):
            if k in h:
                idx = j
                break
        out[key] = idx
    return out

# ----------------- IO/Parsing Helpers -----------------
def load_csv_lenient(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

def coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def index_5char(v) -> str:
    if isinstance(v, (int, np.integer)):
        s = str(int(v)).zfill(5)
    elif isinstance(v, float) and float(v).is_integer():
        s = str(int(v)).zfill(5)
    else:
        s = str(v).strip()
        if s.isdigit() and len(s) < 5:
            s = s.zfill(5)
    return s[:5]

def find_csv_for_index(idx5: str):
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.lower().endswith(".csv"):
                continue
            if name[:5] == idx5:
                return os.path.join(folder, name), folder
    return None, None

# ----------------- Tag location (roomEvent or full scan) -----------------
def count_and_locate_tag(df: pd.DataFrame, room_event_idx: int, tag_text: str):
    needle = normalize_text(tag_text)

    if room_event_idx != -1:
        col = df.columns[room_event_idx]
        ser = df[col].astype(str).map(normalize_text)
        mask = ser.str.contains(re.escape(needle), regex=True, na=False)
        idxs = list(np.flatnonzero(mask.to_numpy()))
        return int(mask.sum()), idxs, False

    # FULL SCAN: normalize whole DF and check any cell per row
    try:
        df_norm = df.astype(str).map(normalize_text)  # pandas >= 2.2
    except AttributeError:
        df_norm = df.astype(str).applymap(normalize_text)  # older pandas
    contains = df_norm.apply(lambda c: c.str.contains(re.escape(needle), regex=True, na=False))
    row_any = contains.any(axis=1)
    idxs = list(np.flatnonzero(row_any.to_numpy()))
    return int(row_any.sum()), idxs, True

# ----------------- Crisis time detection (csv.reader) ----------
def detect_crisis_time_with_csv_reader(csv_path: str, folder_label: str):
    """
    Returns crisis_time (float seconds) or None.
    - shook  : first row whose robotEvent contains 'shook'
    - noshook: first row whose robotEvent contains '0.2 seconds', then +0.229
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, None)
            if header is None:
                return None
            colmap = find_column_indices_relaxed(header, ["time", "robotevent"])
            time_idx = colmap["time"] if colmap["time"] != -1 else 0
            evt_idx  = colmap["robotevent"]
            if evt_idx == -1:
                return None

            for row in r:
                if not row or len(row) <= max(time_idx, evt_idx):
                    continue
                try:
                    tval = float(row[time_idx])
                except Exception:
                    continue
                ev = normalize_text(row[evt_idx])

                if folder_label == "shook":
                    if "shook" in ev:
                        return tval
                elif folder_label == "noshook":
                    if "0.2 seconds" in ev:
                        return tval + 0.229
    except Exception:
        return None
    return None

# ----------------- Geometry / Stationarity -----------------
def abs_pos(x, y, z):
    try:
        fx = float(x); fy = float(y); fz = float(z)
    except Exception:
        return None
    if fx == -1 or fy == -1 or fz == -1:
        return None
    return np.array([abs(fx), abs(fy), abs(fz)], dtype=float)

def in_crisis_sphere(pos_abs: np.ndarray) -> bool:
    if pos_abs is None or not np.all(np.isfinite(pos_abs)):
        return False
    return float(np.linalg.norm(pos_abs - CRISIS_MEAN)) <= CRISIS_RADIUS

def is_still(prev_abs: np.ndarray, curr_abs: np.ndarray) -> bool:
    if prev_abs is None or curr_abs is None:
        return False
    return float(np.linalg.norm(curr_abs - prev_abs)) <= EPS

# ----------------- Per-index analysis -----------------
def analyze_index(idx5: str):
    """
    Returns:
      (folder_label, delta_seconds, is_outlier_bool)
      or (None, None, None) if skipped or delta can't be computed.
    """
    csv_path, folder_label = find_csv_for_index(idx5)
    if not csv_path:
        print(f"{idx5}: SKIP (no CSV found) [folder=none]")
        return None, None, None

    crisis_time = detect_crisis_time_with_csv_reader(csv_path, folder_label)
    if crisis_time is None:
        print(f"{idx5}: SKIP (no crisis tag found: {'shook' if folder_label=='shook' else '0.2 seconds'}) [folder={folder_label}]")
        return None, None, None

    # Load CSV for arrays and tag rows
    try:
        df = load_csv_lenient(csv_path)
        if df.shape[0] == 0:
            print(f"{idx5}: SKIP (empty CSV) [folder={folder_label}]")
            return None, None, None
    except Exception as e:
        print(f"{idx5}: SKIP (CSV read error: {e}) [folder={folder_label}]")
        return None, None, None

    # Columns
    cols_needed = ["time", "roomevent", "robot.x", "robot.y", "robot.z"]
    colmap = find_column_indices_relaxed(list(df.columns), cols_needed)
    time_idx = colmap["time"]
    if time_idx == -1:
        print(f"{idx5}: SKIP (missing 'Time' column) [folder={folder_label}]")
        return None, None, None
    rxi = colmap["robot.x"]; ryi = colmap["robot.y"]; rzi = colmap["robot.z"]
    if rxi == -1 or ryi == -1 or rzi == -1:
        print(f"{idx5}: SKIP (missing Robot.x/y/z columns) [folder={folder_label}]")
        return None, None, None

    # Enter/Exit rows (roomEvent or full scan)
    room_event_idx = colmap["roomevent"]
    enter_cnt, enter_rows, _ = count_and_locate_tag(df, room_event_idx, ENTER_TAG)
    exit_cnt,  exit_rows,  _ = count_and_locate_tag(df, room_event_idx, EXIT_TAG)

    if enter_cnt != 1 or exit_cnt != 1:
        why = []
        if enter_cnt != 1: why.append(f"{'missing' if enter_cnt==0 else 'duplicate'} enter (N={enter_cnt})")
        if exit_cnt  != 1: why.append(f"{'missing' if exit_cnt==0 else 'duplicate'} exit (N={exit_cnt})")
        src_note = "full-scan" if room_event_idx == -1 else "roomEvent"
        print(f"{idx5}: SKIP ({'; '.join(why)}) [{src_note}, folder={folder_label}]")
        return None, None, None

    i_enter = enter_rows[0]
    i_exit  = exit_rows[0]
    if i_exit < i_enter:
        print(f"{idx5}: SKIP (exit before enter) [folder={folder_label}]")
        return None, None, None

    # Arrays
    t  = coerce_float(df.iloc[:, time_idx]).to_numpy()
    rx = df.iloc[:, rxi].to_numpy()
    ry = df.iloc[:, ryi].to_numpy()
    rz = df.iloc[:, rzi].to_numpy()
    if not np.any(np.isfinite(t)):
        print(f"{idx5}: SKIP (no valid times) [folder={folder_label}]")
        return None, None, None

    # Scan window, build stationary sessions inside crisis sphere
    sessions = []  # (start_idx, end_idx)
    inside_prev = False
    last_still_pos  = None
    sess_start = None

    for i in range(i_enter, i_exit + 1):
        pos = abs_pos(rx[i], ry[i], rz[i])
        inside = in_crisis_sphere(pos)

        if inside:
            if last_still_pos is None or not is_still(last_still_pos, pos):
                if sess_start is not None:
                    sessions.append((sess_start, i - 1))
                sess_start = i
                last_still_pos = pos.copy()
        else:
            if sess_start is not None:
                sessions.append((sess_start, i - 1))
                sess_start = None
            last_still_pos = None

        inside_prev = inside

    if sess_start is not None:
        sessions.append((sess_start, i_exit))

    if not sessions:
        print(f"{idx5} [{folder_label}]: no stationary session inside crisis sphere")
        return folder_label, None, None

    def seg_duration(seg):
        s, e = seg
        ts = t[s] if np.isfinite(t[s]) else np.nan
        te = t[e] if np.isfinite(t[e]) else np.nan
        return float(te - ts) if (np.isfinite(ts) and np.isfinite(te)) else float("nan")

    durations = [seg_duration(seg) for seg in sessions]
    if all(not np.isfinite(d) for d in durations):
        print(f"{idx5} [{folder_label}]: no valid times for stationary sessions")
        return folder_label, None, None

    best_idx = int(np.nanargmax(durations))
    best_seg = sessions[best_idx]
    best_dur = durations[best_idx]

    # first move-again time after best session ends
    end_i = best_seg[1]
    ref_pos = abs_pos(rx[end_i], ry[end_i], rz[end_i])
    move_time = np.nan
    for j in range(end_i + 1, i_exit + 1):
        p = abs_pos(rx[j], ry[j], rz[j])
        if p is None or ref_pos is None or not np.isfinite(t[j]):
            continue
        if not is_still(ref_pos, p):
            move_time = float(t[j])
            break

    if not np.isfinite(move_time):
        print(f"{idx5} [{folder_label}]: longest_stationary={best_dur:.6f}s, no move-again found; delta=NaN")
        return folder_label, None, None

    crisis_time = float(crisis_time)
    delta = crisis_time - move_time

    # Band keep rule: keep ONLY if DELTA_KEEP_MIN <= delta <= DELTA_KEEP_MAX
    is_outlier = not (DELTA_KEEP_MIN <= delta <= DELTA_KEEP_MAX)

    status = "OUTLIER" if is_outlier else "OK"
    print(
        f"{idx5} [{folder_label}]: longest_stationary={best_dur:.6f}s, "
        f"move_again={move_time:.6f}s, crisis={crisis_time:.6f}s, "
        f"delta={delta:.6f}s  [{status}]"
    )

    return folder_label, delta, is_outlier

# ----------------- Main -----------------
def main():
    shook_deltas, nosh_deltas = [], []
    shook_outliers, nosh_outliers = [], []

    try:
        df_idx = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read Excel '{EXCEL_FILE}': {e}")
        return
    indices = df_idx.iloc[:, ID_COL].apply(index_5char)

    for idx5 in indices:
        label, delta, is_outlier = analyze_index(idx5)
        if label is None or delta is None:
            continue
        if label == "shook":
            (shook_outliers if is_outlier else shook_deltas).append((idx5, delta) if is_outlier else delta)
        elif label == "noshook":
            (nosh_outliers if is_outlier else nosh_deltas).append((idx5, delta) if is_outlier else delta)

    def summarize(label, kept_list, outliers_list):
        print(f"\n--- Summary: {label} ---")
        print(f"keep band: [{DELTA_KEEP_MIN}, {DELTA_KEEP_MAX}] seconds")
        kept = np.array(kept_list, dtype=float)
        print(f"kept={len(kept_list)}  outliers={len(outliers_list)}")
        if len(kept_list) > 0:
            mean = float(np.mean(kept))
            var  = float(np.var(kept, ddof=1)) if len(kept) > 1 else float('nan')
            print(f"mean delta = {mean:.6f}s")
            print(f"var  delta = {var:.6f}")
        else:
            print("No kept deltas.")
        if outliers_list:
            print("Outliers (excluded):")
            for idx, d in outliers_list:
                print(f"  {idx}: delta={float(d):.6f}s")

    summarize("shook",   shook_deltas, shook_outliers)
    summarize("noshook", nosh_deltas,  nosh_outliers)

if __name__ == "__main__":
    main()