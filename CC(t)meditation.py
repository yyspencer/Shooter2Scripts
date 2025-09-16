#!/usr/bin/env python3
import os
import re
import shutil
import numpy as np
import pandas as pd

# ========= Config =========
SOURCE_XLSX  = "Shooter 2 Data.xlsx"
OUTPUT_XLSX  = "Shooter 2 Proceed.xlsx"
ID_COL       = 0                      # index column in sheet 1 (0-based)
CRISIS_COL   = 3                      # crisis time column in sheet 1 (0-based; 4th Excel column)

# UPDATED: include baseline subfolders for both shook and noshookmodified
SEARCH_DIRS  = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshookmodified",
    os.path.join("noshookmodified", "baseline"),
]

# Sheet order (1-based in Excel): 2..5 = pre_no, pre_med, post_no, post_med
SHEET_IDX = {
    "pre_no":  1,
    "pre_med": 2,
    "post_no": 3,
    "post_med":4,
}

# Output columns (Excel 1-based: O,P,Q → 15..17 → 0-based 14..16)
COL_BEST_LAG = 14
COL_CC_BEST  = 15
COL_CC_GLOB  = 16

# CSV column names (case-insensitive, normalized)
TIME_COL       = "time"
ROOMEVENT      = "roomevent"          # preferred for survey & meditation toggles
EVENT_FALLBACK = "event"              # fallback for toggles

PLAYER_COLS = ["playervr.x", "playervr.y", "playervr.z"]
ROBOT_COLS  = ["robot.x", "robot.y", "robot.z"]

# --------- Helpers ----------
def normalize_header_token(s: str) -> str:
    """Lowercase, strip, remove spaces/._- for robust header matching."""
    return re.sub(r"[ \t._-]+", "", str(s).strip().lower())

def find_cols(df: pd.DataFrame, names):
    """Return actual df column names for requested normalized names; None if missing."""
    norm2real = {normalize_header_token(c): c for c in df.columns}
    return [norm2real.get(normalize_header_token(name)) for name in names]

def find_col(df: pd.DataFrame, name):
    return find_cols(df, [name])[0]

def index5(val) -> str:
    """First 5 characters from the Excel cell."""
    return str(val).strip()[:5]

def find_csv_for_index(idx5: str):
    """Return (path,folder) for CSV whose name starts with idx5 (check SEARCH_DIRS in order)."""
    for folder in SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

def to_float_array(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()

def build_speed_pairs(times, p_xyz, r_xyz):
    """
    Compute speeds (distance / dt) between strictly consecutive rows (i-1 -> i).
    Drop any segment with non-finite times, non-positive dt, non-finite coords, or -1 sentinel.
    Returns list of (i, player_speed, robot_speed) where i is the ending row index.
    """
    n = len(times)
    pairs = []
    prev = None
    for i in range(n):
        if prev is None:
            prev = i
            continue
        if i - prev != 1:
            prev = i
            continue
        t0, t1 = times[prev], times[i]
        if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
            prev = i
            continue
        p0, p1 = p_xyz[prev], p_xyz[i]
        r0, r1 = r_xyz[prev], r_xyz[i]
        if (not np.isfinite(p0).all()) or (not np.isfinite(p1).all()) \
           or (not np.isfinite(r0).all()) or (not np.isfinite(r1).all()):
            prev = i; continue
        if (p0 == -1).any() or (p1 == -1).any() or (r0 == -1).any() or (r1 == -1).any():
            prev = i; continue
        dp = np.linalg.norm(p1 - p0)
        dr = np.linalg.norm(r1 - r0)
        pairs.append((i, dp/(t1 - t0), dr/(t1 - t0)))
        prev = i
    return pairs

def pearson_corr(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0: return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def cc_at_lag(x, y, lag):
    n = len(x)
    if lag > 0:
        xs, ys = x[:n - lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[:n + lag]
    else:
        xs, ys = x, y
    if len(xs) == 0 or len(ys) == 0: return 0.0
    return pearson_corr(xs, ys)

def build_survey_mask(df: pd.DataFrame, n: int):
    """Exclude Survey Room intervals via roomEvent (preferred) else Event)."""
    in_survey = np.zeros(n, dtype=bool)
    col = find_col(df, ROOMEVENT)
    if col is None:
        col = find_col(df, EVENT_FALLBACK)
    if col is None: 
        return in_survey
    ser = df[col].astype(str).str.lower()
    ent_mask  = ser.str.contains("entered survey room", regex=False, na=False) | ser.str.contains("robot entered survey room", regex=False, na=False)
    exit_mask = ser.str_contains("exited survey room",  regex=False, na=False) if hasattr(ser, "str_contains") else ser.str.contains("exited survey room",  regex=False, na=False)
    exit_mask = exit_mask | ser.str.contains("robot exited survey room",  regex=False, na=False)
    enters = np.flatnonzero(ent_mask.to_numpy())
    exits  = np.flatnonzero(exit_mask.to_numpy())
    for s, e in zip(enters, exits):
        if s < e:
            in_survey[s:e+1] = True
    return in_survey

def build_meditation_state(df: pd.DataFrame, n: int):
    """Boolean 'in_meditation' per row from 'Entered/Exited Meditation Area' (roomEvent preferred else Event)."""
    in_med = np.zeros(n, dtype=bool)
    col = find_col(df, ROOMEVENT)
    if col is None:
        col = find_col(df, EVENT_FALLBACK)
    if col is None: 
        return in_med
    ser = df[col].astype(str).str.lower()
    ent_mask  = ser.str.contains("entered meditation area", regex=False, na=False)
    exit_mask = ser.str.contains("exited meditation area",  regex=False, na=False)
    enters = set(np.flatnonzero(ent_mask.to_numpy()).tolist())
    exits  = set(np.flatnonzero(exit_mask.to_numpy()).tolist())
    state = False
    for i in range(n):
        if i in enters: state = True
        if i in exits:  state = False
        in_med[i] = state
    return in_med

def categorize_pairs(times, in_med, crisis_time, speed_pairs):
    """
    Assign each speed pair (ending at row i) to one of four categories based on
    time[i] (pre/post crisis) and in_med[i] (med/no-med).
    Returns dict: {cat: (p_array, r_array)}
    """
    cats = {"pre_no": [], "pre_med": [], "post_no": [], "post_med": []}
    for (i, ps, rs) in speed_pairs:
        if not np.isfinite(times[i]): 
            continue
        pre = times[i] <= crisis_time
        med = bool(in_med[i])
        if pre and not med: cats["pre_no"].append((ps, rs))
        elif pre and med:   cats["pre_med"].append((ps, rs))
        elif (not pre) and not med: cats["post_no"].append((ps, rs))
        else: cats["post_med"].append((ps, rs))
    out = {}
    for k, lst in cats.items():
        if lst:
            arr = np.array(lst, dtype=float)
            out[k] = (arr[:,0], arr[:,1])
        else:
            out[k] = (np.array([], dtype=float), np.array([], dtype=float))
    return out

# --------- Main ----------
def main():
    # Duplicate workbook
    shutil.copyfile(SOURCE_XLSX, OUTPUT_XLSX)

    # Read indices & crisis times from sheet 1
    xls = pd.ExcelFile(SOURCE_XLSX, engine="openpyxl")
    sheet_names = xls.sheet_names
    if len(sheet_names) < 5:
        print("❌ Expected at least 5 sheets (overview + 4 category sheets).")
        return

    df_over = xls.parse(sheet_names[0])
    if df_over.shape[1] <= max(ID_COL, CRISIS_COL):
        print("❌ Sheet 1 must contain indices (col 1) and crisis time (col 4).")
        return

    indices      = df_over.iloc[:, ID_COL].apply(index5).to_list()
    crisis_times = pd.to_numeric(df_over.iloc[:, CRISIS_COL], errors="coerce").to_numpy()

    # Prepare category DataFrames (sheets 2..5) and ensure cols to O–Q (0-based 14..16)
    cat_frames = {}
    for key, si in SHEET_IDX.items():
        df = pd.read_excel(OUTPUT_XLSX, sheet_name=sheet_names[si], engine="openpyxl")
        while df.shape[1] <= COL_CC_GLOB:
            df[f"Extra_{df.shape[1]+1}"] = np.nan
        # init headers
        df.columns.values[COL_BEST_LAG] = "Best Lag (t)"
        df.columns.values[COL_CC_BEST]  = "CC(t)"
        df.columns.values[COL_CC_GLOB]  = "CC(global)"
        cat_frames[key] = df

    # Build per-index p/r speeds for each category
    cat_results = {k: [] for k in SHEET_IDX.keys()}  # list of (row_idx, idx5, p_speeds, r_speeds) per category

    for row_idx, (idx5, crisis_time) in enumerate(zip(indices, crisis_times)):
        if not np.isfinite(crisis_time):
            print(f"{idx5}: SKIP (crisis time NaN)")
            continue

        csv_path, folder = find_csv_for_index(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP (no CSV in {SEARCH_DIRS})")
            continue

        try:
            dfr = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
        except Exception as e:
            print(f"{idx5}: SKIP (CSV read error: {e}) [{csv_path}]")
            continue

        # Normalize headers
        dfr.columns = [normalize_header_token(c) for c in dfr.columns]

        # Required columns
        tcol  = find_col(dfr, TIME_COL)
        pcols = find_cols(dfr, PLAYER_COLS)
        rcols = find_cols(dfr, ROBOT_COLS)
        if tcol is None or any(c is None for c in pcols) or any(c is None for c in rcols):
            miss = []
            if tcol is None: miss.append("time")
            miss += [PLAYER_COLS[i] for i,c in enumerate(pcols) if c is None]
            miss += [ROBOT_COLS[i]  for i,c in enumerate(rcols) if c is None]
            print(f"{idx5}: SKIP (missing cols: {', '.join(miss)})")
            continue

        # Parse numerics
        try:
            times = to_float_array(dfr[tcol])
            pxyz  = dfr[pcols].astype(float).to_numpy()
            rxyz  = dfr[rcols].astype(float).to_numpy()
        except Exception as e:
            print(f"{idx5}: SKIP (parse error positions/time): {e}")
            continue

        n = len(times)
        if n < 2:
            print(f"{idx5}: SKIP (not enough rows)")
            continue

        # Exclusions & states
        in_survey = build_survey_mask(dfr, n)
        in_med    = build_meditation_state(dfr, n)

        # Mask survey rows & non-finite positions
        pxyz_masked = pxyz.copy()
        rxyz_masked = rxyz.copy()
        pxyz_masked[in_survey, :] = -1
        rxyz_masked[in_survey, :] = -1
        nonfinite_rows = ~(np.isfinite(pxyz).all(axis=1) & np.isfinite(rxyz).all(axis=1))
        if np.any(nonfinite_rows):
            pxyz_masked[nonfinite_rows, :] = -1
            rxyz_masked[nonfinite_rows, :] = -1

        # Speeds
        speed_pairs = build_speed_pairs(times, pxyz_masked, rxyz_masked)
        if not speed_pairs:
            print(f"{idx5}: SKIP (no valid speed pairs)")
            continue

        # Split into categories using *sheet 1 crisis time*
        cat_speeds = categorize_pairs(times, in_med, float(crisis_time), speed_pairs)

        # Store arrays
        for cat in SHEET_IDX.keys():
            p_arr, r_arr = cat_speeds[cat]
            cat_results[cat].append((row_idx, idx5, p_arr, r_arr))

    # Per-category CC processing & write
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for cat, df_cat in cat_frames.items():
            rows = cat_results[cat]

            # Lags window based on shortest usable series
            lengths = [len(p) for (_,_,p,_) in rows if len(p) > 1]
            if not lengths:
                print(f"[{cat}] No indices with sufficient data.")
                df_cat.to_excel(writer, sheet_name=sheet_names[SHEET_IDX[cat]], index=False)
                continue

            L = max(1, min(lengths) // 4)
            lags = list(range(-L, L + 1))

            cc_list = []
            per_rows = []  # (row_idx, idx5, best_lag, best_cc, cc_raw, cc_clean)

            for (row_idx, idx5, p, r) in rows:
                if len(p) < 2 or len(r) < 2:
                    per_rows.append((row_idx, idx5, None, None, None, None))
                    continue
                cc_raw = np.array([cc_at_lag(p, r, Lg) for Lg in lags], dtype=float)
                finite = np.isfinite(cc_raw)
                cc_clean = np.where(finite, cc_raw, 0.0)
                if np.any(cc_clean != 0.0):
                    bi = int(np.argmax(cc_clean))
                    best_lag = lags[bi]
                    best_cc  = float(cc_clean[bi]) if np.isfinite(cc_raw[bi]) else 0.0
                else:
                    bi = lags.index(0) if 0 in lags else 0
                    best_lag = lags[bi]
                    best_cc  = 0.0
                cc_list.append(cc_clean)
                per_rows.append((row_idx, idx5, best_lag, best_cc, cc_raw, cc_clean))
                print(f"[{cat}] {idx5}: best lag={best_lag}, CC(best)={best_cc:.4f}")

            if not cc_list:
                df_cat.to_excel(writer, sheet_name=sheet_names[SHEET_IDX[cat]], index=False)
                continue

            cc_matrix = np.vstack(cc_list)
            sum_abs   = np.sum(np.abs(cc_matrix), axis=0)
            glob_idx  = int(np.argmax(sum_abs))
            glob_lag  = lags[glob_idx]
            print(f"[{cat}] Global best lag (Σ|CC|): {glob_lag}")

            # Write O,P,Q
            for (row_idx, idx5, best_lag, best_cc, cc_raw, cc_clean) in per_rows:
                if best_lag is None:
                    continue
                df_cat.iat[row_idx, COL_BEST_LAG] = best_lag         # O
                df_cat.iat[row_idx, COL_CC_BEST]  = best_cc          # P
                df_cat.iat[row_idx, COL_CC_GLOB]  = float(cc_clean[glob_idx])  # Q

            df_cat.to_excel(writer, sheet_name=sheet_names[SHEET_IDX[cat]], index=False)

    print("\n✅ Done. Wrote Best Lag / CC(best) / CC(global) to columns O–Q on sheets 2–5 of",
          OUTPUT_XLSX)

if __name__ == "__main__":
    main()