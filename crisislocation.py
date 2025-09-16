import os
import re
import csv
import unicodedata
import pandas as pd
import numpy as np

# ----------------- Config -----------------
EXCEL_FILE = "Shooter 2 Data.xlsx"
ID_COL = 0  # first column holds indices
SEARCH_FOLDERS = [
    os.path.join("shook"),
    os.path.join("shook", "baseline"),
    os.path.join("noshook"),
    os.path.join("noshook", "baseline"),
]

# ----------------- Text/Regex Helpers -----------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower().replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
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

def find_csv_for_index(idx5: str):
    """
    Find CSV path and classify folder_type.
    IMPORTANT: check 'noshook' BEFORE 'shook' so 'noshook' isn't misclassified.
    """
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.lower().endswith(".csv"):
                continue
            if name[:5] == idx5:
                fpath = os.path.join(folder, name)
                folder_lc = os.path.normpath(folder).lower()
                if "noshook" in folder_lc:
                    folder_type = "noshook"
                elif "shook" in folder_lc:
                    folder_type = "shook"
                else:
                    folder_type = "unknown"
                return fpath, folder_type
    return None, None

def nearest_time_index(times: np.ndarray, target_time: float) -> int:
    diffs = np.full_like(times, np.inf, dtype=float)
    mask = np.isfinite(times)
    diffs[mask] = np.abs(times[mask] - target_time)
    return int(np.argmin(diffs))

# ----------------- Crisis-time detection (csv.reader) -----------------
def detect_crisis_time_with_csv_reader(csv_path: str, folder_type: str):
    """
    Returns crisis_time (float seconds) or None.
    - shook  : first row whose robotEvent contains 'shook' (substring, case-insensitive)
    - noshook: first row whose robotEvent contains '0.2 seconds' (substring, case-insensitive), then +0.229
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
                    time_val = float(row[time_idx])
                except Exception:
                    continue
                event_text = normalize_text(row[evt_idx])

                if folder_type == "shook":
                    if "shook" in event_text:
                        return time_val
                elif folder_type == "noshook":
                    # within-cell substring search (not exact-cell match)
                    if "0.2 seconds" in event_text:
                        return time_val + 0.229
                else:
                    return None
    except Exception:
        return None
    return None

# ----------------- Core logic per index -----------------
def analyze_index(idx5: str):
    """
    Returns (folder_type, ratio_percent, (abs_x, abs_y, abs_z)) or (None, None, None) if skipped.
    Also prints per-index details.
    """
    csv_path, folder_type = find_csv_for_index(idx5)
    if not csv_path or folder_type not in ("shook", "noshook"):
        print(f"{idx5}: SKIP (no CSV found or unknown folder type)")
        return None, None, None

    crisis_time = detect_crisis_time_with_csv_reader(csv_path, folder_type)
    if crisis_time is None:
        print(f"{idx5}: SKIP (no crisis tag found: {'shook' if folder_type=='shook' else '0.2 seconds'})")
        return None, None, None

    try:
        df = load_csv_lenient(csv_path)
        if df.shape[0] == 0:
            print(f"{idx5}: SKIP (empty CSV)")
            return None, None, None
    except Exception as e:
        print(f"{idx5}: SKIP (CSV read error: {e})")
        return None, None, None

    cols_needed = ["time", "robot.x", "robot.y", "robot.z"]
    colmap = find_column_indices_relaxed(list(df.columns), cols_needed)
    if any(colmap[k] == -1 for k in cols_needed):
        missing = [k for k in cols_needed if colmap[k] == -1]
        print(f"{idx5}: SKIP (missing columns: {', '.join(missing)})")
        return None, None, None

    time_col = df.columns[colmap["time"]]
    rx_col   = df.columns[colmap["robot.x"]]
    ry_col   = df.columns[colmap["robot.y"]]
    rz_col   = df.columns[colmap["robot.z"]]

    t  = coerce_float(df[time_col]).to_numpy()
    rx = coerce_float(df[rx_col]).to_numpy()
    ry = coerce_float(df[ry_col]).to_numpy()
    rz = coerce_float(df[rz_col]).to_numpy()

    if not np.any(np.isfinite(t)):
        print(f"{idx5}: SKIP (no valid Time values)")
        return None, None, None

    i_crisis = nearest_time_index(t, crisis_time)

    # Crisis position (original values for logic)
    cx, cy, cz = rx[i_crisis], ry[i_crisis], rz[i_crisis]
    if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(cz)):
        print(f"{idx5}: SKIP (invalid robot position at crisis row)")
        return None, None, None

    # Dwell window: exact equality on original values
    i_start = i_crisis
    while i_start - 1 >= 0 and (rx[i_start - 1] == cx) and (ry[i_start - 1] == cy) and (rz[i_start - 1] == cz):
        i_start -= 1
    i_end = i_crisis
    nrows = len(t)
    while i_end + 1 < nrows and (rx[i_end + 1] == cx) and (ry[i_end + 1] == cy) and (rz[i_end + 1] == cz):
        i_end += 1

    start_time = float(t[i_start]) if np.isfinite(t[i_start]) else np.nan
    end_time   = float(t[i_end])   if np.isfinite(t[i_end])   else np.nan
    duration   = end_time - start_time if (np.isfinite(start_time) and np.isfinite(end_time)) else np.nan
    ratio_pct  = ((crisis_time - start_time) / duration * 100.0) if (np.isfinite(duration) and duration > 0) else np.nan

    # Report ABSOLUTE coordinates
    acx, acy, acz = abs(cx), abs(cy), abs(cz)
    pos_str = f"({acx}, {acy}, {acz})"

    if np.isfinite(ratio_pct):
        print(f"{idx5} [{folder_type}]: crisis_time={crisis_time:.6f}s, pos={pos_str}, "
              f"start={start_time:.6f}s, end={end_time:.6f}s, duration={duration:.6f}s, ratio={ratio_pct:.6f}%")
    else:
        print(f"{idx5} [{folder_type}]: crisis_time={crisis_time:.6f}s, pos={pos_str}, "
              f"start={start_time:.6f}s, end={end_time:.6f}s, duration={duration:.6f}s, ratio=NaN")

    return folder_type, (ratio_pct if np.isfinite(ratio_pct) else None), (acx, acy, acz)

# ----------------- Main -----------------
def main():
    try:
        df = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read Excel '{EXCEL_FILE}': {e}")
        return

    indices = df.iloc[:, ID_COL].apply(
        lambda v: str(int(v)).zfill(5) if (isinstance(v, float) and v.is_integer()) else str(v).zfill(5)
    )

    # Separate collectors for shook vs noshook
    shook_ratios, nosh_ratios = [], []
    shook_x, shook_y, shook_z = [], [], []
    nosh_x, nosh_y, nosh_z    = [], [], []

    for idx5 in indices:
        ftype, r_pct, pos = analyze_index(idx5)
        if ftype is None:
            continue
        if r_pct is not None:
            if ftype == "shook":
                shook_ratios.append(r_pct)
            elif ftype == "noshook":
                nosh_ratios.append(r_pct)
        if pos is not None:
            ax, ay, az = pos  # absolute values
            if ftype == "shook":
                shook_x.append(ax); shook_y.append(ay); shook_z.append(az)
            elif ftype == "noshook":
                nosh_x.append(ax);  nosh_y.append(ay);  nosh_z.append(az)

    # ---------- Aggregate outputs (separate) ----------
    print("\n--- Aggregate Statistics (Separate) ---")

    # Ratios
    def ratio_stats(label, arr):
        if arr:
            a = np.array(arr, dtype=float)
            mean = float(np.nanmean(a))
            var  = float(np.nanvar(a, ddof=1)) if len(a) > 1 else float('nan')
            print(f"{label} ratio: count={len(arr)}, mean={mean:.6f}%, sample variance={var:.6f}")
        else:
            print(f"{label} ratio: count=0 (no valid values)")

    ratio_stats("shook", shook_ratios)
    ratio_stats("noshook", nosh_ratios)

    # Crisis location (component-wise mean/var of ABS(x), ABS(y), ABS(z))
    def loc_stats(label, xs, ys, zs):
        if xs and ys and zs:
            X, Y, Z = np.array(xs), np.array(ys), np.array(zs)
            mx, my, mz = float(np.nanmean(X)), float(np.nanmean(Y)), float(np.nanmean(Z))
            vx = float(np.nanvar(X, ddof=1)) if len(X) > 1 else float('nan')
            vy = float(np.nanvar(Y, ddof=1)) if len(Y) > 1 else float('nan')
            vz = float(np.nanvar(Z, ddof=1)) if len(Z) > 1 else float('nan')
            print(f"{label} crisis location mean      : ({mx:.6f}, {my:.6f}, {mz:.6f})")
            print(f"{label} crisis location samp var  : ({vx:.6f}, {vy:.6f}, {vz:.6f})")
        else:
            print(f"{label} crisis location: no valid vectors")

    loc_stats("shook",  shook_x, shook_y, shook_z)
    loc_stats("noshook", nosh_x, nosh_y, nosh_z)

if __name__ == "__main__":
    main()