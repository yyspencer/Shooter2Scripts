#!/usr/bin/env python3
import os
import re
import unicodedata
import numpy as np
import pandas as pd

# ========= Config =========
EXCEL_FILE = "Shooter 2 Data.xlsx"   # first sheet
ID_COL = 0                           # indices column (0-based)
CRISIS_COL = 3                       # "Crisis Time" lives in the 3rd col (0-based)

# Search both the modified folder and its baseline subfolder
CSV_ROOT = "noshookmodified"
SEARCH_FOLDERS = [CSV_ROOT, os.path.join(CSV_ROOT, "baseline")]

# Learned locations (ABS positions), means & variances (per-axis), and radii = 2σ
ENTER_MEAN = np.array([24.594621, 0.0, 45.237331], dtype=float)
ENTER_VAR  = np.array([ 0.000659, 0.0,  0.000252], dtype=float)

EXIT_MEAN  = np.array([33.631669, 0.0, 45.163783], dtype=float)
EXIT_VAR   = np.array([ 0.001266, 0.0,  0.001222], dtype=float)

CRISIS_MEAN = np.array([27.699638, 0.0, 46.760727], dtype=float)
CRISIS_VAR  = np.array([ 0.137175, 0.0,  0.406943], dtype=float)

R_ENTER  = 2.0 * np.sqrt(np.sum(ENTER_VAR))
R_EXIT   = 2.0 * np.sqrt(np.sum(EXIT_VAR))
R_CRISIS = 2.0 * np.sqrt(np.sum(CRISIS_VAR))

# Learned delta (crisis time relative to robot move after longest stationary session)
DELTA_MEAN = -4.989027
DELTA_VAR  =  0.003289
DELTA_SIG  = float(np.sqrt(DELTA_VAR))
DELTA_WIN  = 2.0 * DELTA_SIG   # must land within ±2σ of move_time + DELTA_MEAN

# Stationary tolerance (meters) for consecutive samples
EPS_STATIONARY = 10

# ========= Helpers =========
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", normalize_text(s).lower())

def find_columns(header, want_keys):
    """
    Relaxed (case-insensitive) substring match on normalized header tokens.
    Returns dict {key -> col_idx or -1}.
    """
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

def index5(v) -> str:
    """First 5 chars of the Excel value; zero-pad if purely numeric and <5 digits."""
    if isinstance(v, (int, np.integer)):
        s = str(int(v)).zfill(5)
    elif isinstance(v, float) and float(v).is_integer():
        s = str(int(v)).zfill(5)
    else:
        s = str(v).strip()
        if s.isdigit() and len(s) < 5:
            s = s.zfill(5)
    return s[:5]

def load_csv_lenient(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

def coerce_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()

def abs_pos_triplet(x, y, z):
    """Return abs() position triple; if any invalid -> None."""
    try:
        fx, fy, fz = float(x), float(y), float(z)
    except Exception:
        return None
    if fx == -1 or fy == -1 or fz == -1:
        return None
    return abs(fx), abs(fy), abs(fz)

def dist_euclid(p, q):
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))

def nearest_time_index(times: np.ndarray, target_time: float) -> int:
    diffs = np.full_like(times, np.inf, dtype=float)
    mask = np.isfinite(times)
    diffs[mask] = np.abs(times[mask] - target_time)
    return int(np.argmin(diffs))

def append_event(df: pd.DataFrame, row_idx: int, evt_col_name: str, text: str):
    """Append text to robotEvent cell (create column if needed)."""
    if evt_col_name not in df.columns:
        df[evt_col_name] = ""
    prev = "" if row_idx >= len(df) else str(df.at[row_idx, evt_col_name])
    prev = "" if prev == "nan" else prev
    df.at[row_idx, evt_col_name] = (prev + (" | " if prev else "") + text)

def find_first_in_sphere(times, rx, ry, rz, center, radius, start_idx=0):
    """Return first row index >= start_idx with abs(pos) inside sphere, else -1."""
    n = len(times)
    for i in range(int(start_idx), n):
        pos = abs_pos_triplet(rx[i], ry[i], rz[i])
        if pos is None:
            continue
        if dist_euclid(pos, center) <= radius:
            return i
    return -1

def longest_stationary_session_in_crisis(times, rx, ry, rz, enter_i, exit_i):
    """
    Find the longest stationary session *within the crisis sphere* in [enter_i, exit_i].
    A session = consecutive rows where:
      - inside crisis sphere, and
      - step distance to previous sample <= EPS_STATIONARY.
    Returns (start_i, end_i) or (None, None).
    """
    best = (None, None, -1.0)  # (s, e, duration)
    s = None
    prev_pos = None

    for i in range(enter_i, exit_i + 1):
        pos = abs_pos_triplet(rx[i], ry[i], rz[i])
        if pos is None:
            # break any ongoing session
            if s is not None and i - 1 >= s:
                dur = float(times[i - 1]) - float(times[s])
                if dur > best[2]:
                    best = (s, i - 1, dur)
            s = None
            prev_pos = None
            continue

        inside = (dist_euclid(pos, CRISIS_MEAN) <= R_CRISIS)
        if not inside:
            if s is not None and i - 1 >= s:
                dur = float(times[i - 1]) - float(times[s])
                if dur > best[2]:
                    best = (s, i - 1, dur)
            s = None
            prev_pos = None
            continue

        # inside crisis sphere
        if s is None:
            s = i
            prev_pos = pos
        else:
            step = dist_euclid(pos, prev_pos)
            if step <= EPS_STATIONARY:
                prev_pos = pos
            else:
                dur = float(times[i - 1]) - float(times[s])
                if dur > best[2]:
                    best = (s, i - 1, dur)
                s = i
                prev_pos = pos

    # close tail
    if s is not None and exit_i >= s:
        dur = float(times[exit_i]) - float(times[s])
        if dur > best[2]:
            best = (s, exit_i, dur)

    return (best[0], best[1]) if best[0] is not None else (None, None)

# ---- New: find window (proximity first; if not, fall back to roomEvent tags; else full file)
def find_window_indices(t, rx, ry, rz, df, colmap):
    n = len(t)

    # 1) Proximity-based enter/exit
    i_enter = find_first_in_sphere(t, rx, ry, rz, ENTER_MEAN, R_ENTER, start_idx=0)
    if i_enter != -1:
        i_exit = find_first_in_sphere(t, rx, ry, rz, EXIT_MEAN, R_EXIT, start_idx=i_enter + 1)
        if i_exit != -1 and i_exit > i_enter:
            return i_enter, i_exit, "proximity", True, True

    # 2) roomEvent textual tags (case-insensitive substring)
    room_idx = colmap.get("roomevent", -1)
    if room_idx != -1:
        room_col = df.columns[room_idx]
        ser = df[room_col].astype(str).map(lambda s: normalize_text(s).lower())

        ent_mask  = ser.str.contains("entered survey room", regex=False, na=False)
        exit_mask = ser.str.contains("exited survey room",  regex=False, na=False)

        ent_idxs  = np.flatnonzero(ent_mask.to_numpy())
        exit_idxs = np.flatnonzero(exit_mask.to_numpy())

        if ent_idxs.size > 0 and exit_idxs.size > 0:
            ent0 = int(ent_idxs[0])
            # first exit at/after enter
            exits_after = exit_idxs[exit_idxs >= ent0]
            if exits_after.size > 0:
                ex0 = int(exits_after[0])
                if ex0 > ent0:
                    return ent0, ex0, "roomEvent-tags", True, True
        # partial window: extend to file edges
        if ent_idxs.size > 0:
            ent0 = int(ent_idxs[0])
            return ent0, n - 1, "roomEvent-partial-enter", True, False
        if exit_idxs.size > 0:
            ex0 = int(exit_idxs[0])
            return 0, ex0, "roomEvent-partial-exit", False, True

    # 3) Fallback to full file if nothing found
    return 0, n - 1, "full-range", False, False

# ========= Main =========
def main():
    # Load Excel (first sheet)
    try:
        df_idx = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read Excel '{EXCEL_FILE}': {e}")
        return

    if df_idx.shape[1] <= CRISIS_COL:
        print(f"❌ Excel has no 3rd column (0-based index {CRISIS_COL}).")
        return

    # Build list of indices with blank Crisis Time (NaN / empty)
    def is_blank(v):
        if pd.isna(v):
            return True
        s = str(v).strip()
        return s == "" or s.lower() == "nan"

    todo_rows = []
    for i in range(len(df_idx)):
        if is_blank(df_idx.iat[i, CRISIS_COL]):
            todo_rows.append(i)

    if not todo_rows:
        print("All rows already have Crisis Time; nothing to do.")
        return

    print("Will process {} rows with blank Crisis Time, using CSVs in: {}".format(
        len(todo_rows),
        ", ".join([repr(f) for f in SEARCH_FOLDERS])
    ))

    for i in todo_rows:
        raw_id = df_idx.iat[i, ID_COL]
        idx5 = index5(raw_id)

        # find CSV in the search folders (noshookmodified and noshookmodified/baseline)
        csv_path = None
        found_folder = None
        for folder in SEARCH_FOLDERS:
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if name.lower().endswith(".csv") and name[:5] == idx5:
                    csv_path = os.path.join(folder, name)
                    found_folder = folder
                    break
            if csv_path:
                break

        if not csv_path:
            print(f"{idx5}: SKIP (no CSV found in {SEARCH_FOLDERS}).")
            continue

        # load CSV
        try:
            df = load_csv_lenient(csv_path)
        except Exception as e:
            print(f"{idx5}: SKIP (CSV read error: {e}) [folder={found_folder}]")
            continue
        if df.shape[0] == 0:
            print(f"{idx5}: SKIP (empty CSV) [folder={found_folder}]")
            continue

        # locate columns (now also looking for 'roomEvent' explicitly)
        need = ["time", "robotevent", "roomevent", "robot.x", "robot.y", "robot.z"]
        colmap = find_columns(list(df.columns), need)

        # Require time and robot position columns
        missing = [k for k in ["time", "robot.x", "robot.y", "robot.z"] if colmap[k] == -1]
        if missing:
            print(f"{idx5}: SKIP (missing columns: {', '.join(missing)}) [folder={found_folder}]")
            continue

        time_col = df.columns[colmap["time"]]
        evt_col  = df.columns[colmap["robotevent"]] if colmap["robotevent"] != -1 else "robotEvent"

        t  = coerce_float(df[time_col])
        rx = df[df.columns[colmap["robot.x"]]].to_numpy()
        ry = df[df.columns[colmap["robot.y"]]].to_numpy()
        rz = df[df.columns[colmap["robot.z"]]].to_numpy()

        n = len(t)
        if n == 0 or not np.any(np.isfinite(t)):
            print(f"{idx5}: SKIP (no valid Time values) [folder={found_folder}]")
            continue

        # --- choose the analysis window (proximity → roomEvent tags → full range)
        i_enter, i_exit, win_src, enter_found, exit_found = find_window_indices(t, rx, ry, rz, df, colmap)

        # Find longest stationary session inside crisis sphere within chosen window
        s_idx, e_idx = longest_stationary_session_in_crisis(t, rx, ry, rz, i_enter, i_exit)
        if s_idx is None or e_idx is None or e_idx < s_idx:
            print(f"{idx5}: SKIP (no stationary session inside crisis sphere in {win_src} window) [folder={found_folder}].")
            continue

        # time robot starts moving after longest stationary session
        if e_idx + 1 >= n or not np.isfinite(t[e_idx + 1]):
            print(f"{idx5}: SKIP (cannot find movement time after stationary session) [folder={found_folder}].")
            continue
        move_time = float(t[e_idx + 1])

        # predicted crisis time from learned delta
        target_time = move_time + DELTA_MEAN
        j = nearest_time_index(t, target_time)
        if not np.isfinite(t[j]) or abs(float(t[j]) - target_time) > DELTA_WIN:
            print(f"{idx5}: SKIP (no row within ±2σ of predicted crisis time; "
                  f"nearest diff={abs(float(t[j])-target_time):.6f}s) [folder={found_folder}].")
            continue

        # write/append events (only write enter/exit if we actually found them)
        if enter_found:
            append_event(df, i_enter, evt_col, "Robot entered Survey Room")
        if exit_found:
            append_event(df, i_exit,  evt_col, "Robot exited Survey Room")
        append_event(df, e_idx + 1, evt_col, "Robot stops moving after longest stationary session")
        append_event(df, j, evt_col, "estimated shook")

        # save back in-place
        try:
            df.to_csv(csv_path, index=False)
            ent_info = f"enter@{t[i_enter]:.3f}s" if enter_found else "enter@N/A"
            ex_info  = f"exit@{t[i_exit]:.3f}s"   if exit_found  else "exit@N/A"
            print(f"{idx5}: OK [{win_src}, folder={found_folder}] — {ent_info}, {ex_info}, "
                  f"move@{t[e_idx+1]:.3f}s, estimated_shook@{t[j]:.3f}s (Δ={t[j]-move_time:.3f}s).")
        except Exception as e:
            print(f"{idx5}: ❌ Failed to write CSV [folder={found_folder}]: {e}")

if __name__ == "__main__":
    main()