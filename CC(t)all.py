#!/usr/bin/env python3
import os
import re
import shutil
import numpy as np
import pandas as pd

# ========= Config =========
SOURCE_XLSX  = "Shooter 2 Data.xlsx"
OUTPUT_XLSX  = "Shooter 2 Proceed.xlsx"
ID_COL       = 0  # index column in Excel (0-based)
SEARCH_DIRS  = ["shook", "noshookmodified"]  # order matters; no baseline folders

# Output columns (1-based 18–20 → 0-based 17–19)
COL_BEST_LAG = 17
COL_CC_BEST  = 18
COL_CC_GLOB  = 19

# CSV column names to find (case-insensitive, normalized)
TIME_COL    = "time"
ROOM_EVENT  = "roomevent"
PLAYER_COLS = ["playervr.x", "playervr.y", "playervr.z"]
ROBOT_COLS  = ["robot.x", "robot.y", "robot.z"]

# --------- Helpers ----------
def normalize_header_token(s: str) -> str:
    """Lowercase, strip, remove spaces/._- for robust header matching."""
    return re.sub(r"[ \t._-]+", "", str(s).strip().lower())

def find_cols(df: pd.DataFrame, names):
    """Return actual column names for requested normalized names; None if missing."""
    norm2real = {normalize_header_token(c): c for c in df.columns}
    return [norm2real.get(normalize_header_token(name)) for name in names]

def find_col(df: pd.DataFrame, name):
    return find_cols(df, [name])[0]

def index5(val) -> str:
    """First 5 characters from the Excel cell; no zero-padding per your spec."""
    return str(val).strip()[:5]

def find_csv_for_index(idx5: str):
    """Return (path, folder) for CSV whose name starts with idx5 (check SEARCH_DIRS in order)."""
    for folder in SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

def coerce_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()

def build_speed_pairs(times, p_xyz, r_xyz):
    """
    Compute speeds (distance / dt) between strictly consecutive rows (i-1 -> i),
    skipping any segment with non-finite times, non-positive dt,
    OR any non-finite coord OR any -1 coord at either end.
    Returns list of (i, player_speed, robot_speed).
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
        # Drop any segment with non-finite coords OR -1 sentinels
        if (not np.isfinite(p0).all()) or (not np.isfinite(p1).all()) \
           or (not np.isfinite(r0).all()) or (not np.isfinite(r1).all()):
            prev = i; continue
        if (p0 == -1).any() or (p1 == -1).any() or (r0 == -1).any() or (r1 == -1).any():
            prev = i; continue
        dp = np.linalg.norm(p1 - p0)
        dr = np.linalg.norm(r1 - r0)
        pairs.append((i, dp / (t1 - t0), dr / (t1 - t0)))
        prev = i
    return pairs

def pearson_corr(x, y):
    if len(x) == 0 or len(y) == 0:
        return 0.0
    sx, sy = np.std(x), np.std(y)
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def cc_at_lag(x, y, lag):
    n = len(x)
    if lag > 0:
        xs, ys = x[:n - lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[:n + lag]
    else:
        xs, ys = x, y
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    return pearson_corr(xs, ys)

# --------- Main ----------
def main():
    # Duplicate workbook (so we never touch the original)
    shutil.copyfile(SOURCE_XLSX, OUTPUT_XLSX)
    df_idx = pd.read_excel(SOURCE_XLSX, engine="openpyxl")

    # Prepare output workbook frame
    df_out = pd.read_excel(OUTPUT_XLSX, engine="openpyxl")
    while df_out.shape[1] <= COL_CC_GLOB:
        df_out[f"Extra_{df_out.shape[1] + 1}"] = np.nan
    df_out.iloc[:, COL_BEST_LAG] = np.nan
    df_out.iloc[:, COL_CC_BEST]  = np.nan
    df_out.iloc[:, COL_CC_GLOB]  = np.nan
    df_out.columns.values[COL_BEST_LAG] = "Best Lag (t)"
    df_out.columns.values[COL_CC_BEST]  = "CC(t)"
    df_out.columns.values[COL_CC_GLOB]  = "CC(global)"

    # Pass 1: build speeds per index, excluding Survey Room intervals if present
    per_index_speeds = []  # (row_idx, idx5, p_speeds, r_speeds)
    skipped = {"no_csv": 0, "parse_err": 0, "missing_cols": 0, "no_speeds": 0}

    for row_idx, val in enumerate(df_idx.iloc[:, ID_COL].tolist()):
        idx5 = index5(val)
        csv_path, folder = find_csv_for_index(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP (no CSV found in {SEARCH_DIRS})")
            skipped["no_csv"] += 1
            continue

        try:
            dfr = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
        except Exception as e:
            print(f"{idx5}: SKIP (CSV read error: {e}) [{csv_path}]")
            skipped["parse_err"] += 1
            continue

        # Normalize headers for robust matching
        dfr.columns = [normalize_header_token(c) for c in dfr.columns]

        # Required columns
        tcol = find_col(dfr, TIME_COL)
        pcols = find_cols(dfr, PLAYER_COLS)
        rcols = find_cols(dfr, ROBOT_COLS)
        rroom = find_col(dfr, ROOM_EVENT)  # may be None

        missing = []
        if tcol is None: missing.append("time")
        if any(c is None for c in pcols):
            missing += [PLAYER_COLS[i] for i, c in enumerate(pcols) if c is None]
        if any(c is None for c in rcols):
            missing += [ROBOT_COLS[i] for i, c in enumerate(rcols) if c is None]
        if missing:
            print(f"{idx5}: SKIP (missing cols: {', '.join(missing)})")
            skipped["missing_cols"] += 1
            continue

        # Parse numerics
        try:
            times = coerce_float_series(dfr[tcol])
            pxyz  = dfr[pcols].astype(float).to_numpy()
            rxyz  = dfr[rcols].astype(float).to_numpy()
        except Exception as e:
            print(f"{idx5}: SKIP (parse error for positions/time): {e}")
            skipped["parse_err"] += 1
            continue

        n = len(times)
        if n < 2:
            print(f"{idx5}: SKIP (not enough rows)")
            skipped["no_speeds"] += 1
            continue

        # Exclude Survey Room intervals (robust to dtype) using pandas str.contains
        in_survey = np.zeros(n, dtype=bool)
        if rroom is not None:
            ser = dfr[rroom].astype(str).str.lower()
            ent_mask  = ser.str.contains("robot entered survey room", regex=False, na=False)
            exit_mask = ser.str.contains("robot exited survey room",  regex=False, na=False)
            enters = np.flatnonzero(ent_mask.to_numpy())
            exits  = np.flatnonzero(exit_mask.to_numpy())
            for s, e in zip(enters, exits):
                if s < e:
                    in_survey[s:e + 1] = True

        # Mask survey rows by setting coords to -1 so they get dropped in speed builder
        pxyz_masked = pxyz.copy()
        rxyz_masked = rxyz.copy()
        pxyz_masked[in_survey, :] = -1
        rxyz_masked[in_survey, :] = -1

        # Also mask rows with any non-finite position values → drop them like survey rows
        nonfinite_rows = ~(np.isfinite(pxyz).all(axis=1) & np.isfinite(rxyz).all(axis=1))
        if np.any(nonfinite_rows):
            pxyz_masked[nonfinite_rows, :] = -1
            rxyz_masked[nonfinite_rows, :] = -1
            print(f"{idx5}: masked {int(nonfinite_rows.sum())} row(s) with non-finite positions")

        speed_pairs = build_speed_pairs(times, pxyz_masked, rxyz_masked)
        if not speed_pairs:
            print(f"{idx5}: SKIP (no valid speed pairs after filtering)")
            skipped["no_speeds"] += 1
            continue

        p_speeds = np.array([ps for (_, ps, _) in speed_pairs], dtype=float)
        r_speeds = np.array([rs for (_, _, rs) in speed_pairs], dtype=float)

        # Drop any non-finite speed pairs before CC
        finite_mask = np.isfinite(p_speeds) & np.isfinite(r_speeds)
        dropped = int((~finite_mask).sum())
        if dropped:
            print(f"{idx5}: dropped {dropped} non-finite speed pair(s) before CC")
        p_speeds = p_speeds[finite_mask]
        r_speeds = r_speeds[finite_mask]

        if len(p_speeds) < 2:
            print(f"{idx5}: SKIP (too few finite speed pairs after filtering)")
            skipped["no_speeds"] += 1
            continue

        per_index_speeds.append((row_idx, idx5, p_speeds, r_speeds))
        print(f"{idx5}: built {len(p_speeds)} finite speed pairs")

    if not per_index_speeds:
        print("No valid indices; nothing to write.")
        return

    # ----- Pass 2: per-index best lags & CCs, and robust global best lag with NaN handling -----
    # Common lag window based on the shortest usable series; cap to [-1000, 1000]
    lengths = [len(p) for (_, _, p, _) in per_index_speeds]
    L_data = max(1, min(lengths) // 4)
    L_cap  = 1000
    global_L = min(L_data, L_cap)
    lags = list(range(-global_L, global_L + 1))
    print(f"Lag window capped: L_data={L_data}, L_cap={L_cap} → using [-{global_L}, +{global_L}] (K={len(lags)})")

    # Build raw CC vectors across all lags for each index
    cc_records = []  # (row_idx, idx5, p, r, cc_raw, cc_clean, finite_mask)
    for row_idx, idx5, p, r in per_index_speeds:
        cc_raw = np.array([cc_at_lag(p, r, L) for L in lags], dtype=float)
        finite_mask = np.isfinite(cc_raw)                 # remember where NaNs/±Inf were
        cc_clean = np.where(finite_mask, cc_raw, 0.0)     # sanitize for argmax/global sum
        cc_records.append((row_idx, idx5, p, r, cc_raw, cc_clean, finite_mask))

    # Per-index best lag & CC(best), with explicit NaN flagging
    per_index_out = []  # (row_idx, idx5, best_lag, best_cc_written, cc_raw, cc_clean, finite_mask, flags)
    nan_any_count = 0
    for row_idx, idx5, p, r, cc_raw, cc_clean, finite_mask in cc_records:
        if np.any(cc_clean != 0.0):
            best_idx = int(np.argmax(cc_clean))   # maximize (sanitized) CC
            best_lag = lags[best_idx]
            best_cc  = float(cc_clean[best_idx])
            # Was the original value at best lag NaN? (mark & write 0.0 anyway)
            nan_at_best = not np.isfinite(cc_raw[best_idx])
            best_cc_written = 0.0 if nan_at_best else best_cc
        else:
            # All zeros after sanitize => no reliable structure (all NaN or all flat)
            best_idx = lags.index(0) if 0 in lags else 0
            best_lag = lags[best_idx]
            best_cc_written = 0.0
            nan_at_best = np.any(~finite_mask)
        if np.any(~finite_mask):
            nan_any_count += 1
        per_index_out.append(
            (row_idx, idx5, best_lag, best_cc_written, cc_raw, cc_clean, finite_mask,
             {"nan_any": bool(np.any(~finite_mask)), "nan_at_best": bool(nan_at_best), "best_idx": best_idx})
        )
        note = ""
        if nan_at_best:
            note += " [ERR_NAN_AT_BEST→wrote 0]"
        elif np.any(~finite_mask):
            note += " [NaN_in_CC_vec_excluded]"
        print(f"{idx5}: best lag={best_lag}, CC(best)={best_cc_written:.4f}{note}")

    # Global best lag (maximize Σ|CC(t)|) using **sanitized** matrix (so NaNs contribute 0)
    cc_matrix_clean = np.vstack([rec[5] for rec in per_index_out])    # [M x K]
    sum_abs_by_lag  = np.sum(np.abs(cc_matrix_clean), axis=0)
    global_best_idx = int(np.argmax(sum_abs_by_lag))
    global_best_lag = lags[global_best_idx]
    print(f"\nGlobal best lag (samples) maximizing Σ|CC|: {global_best_lag}")

    # Write outputs:
    # - col 18: per-index best lag
    # - col 19: CC at per-index best lag (0.0 if original NaN at that lag)
    # - col 20: CC at the **global best lag**
    for row_idx, idx5, best_lag, best_cc_written, cc_raw, cc_clean, finite_mask, flags in per_index_out:
        # Global CC for this index at the chosen global lag; write 0.0 if original was NaN
        cc_global_raw   = float(cc_raw[global_best_idx]) if np.isfinite(cc_raw[global_best_idx]) else np.nan
        cc_global_write = float(cc_clean[global_best_idx])   # already 0 if NaN originally
        note_g = ""
        if not np.isfinite(cc_global_raw):
            note_g = " [ERR_NAN_AT_GLOBAL→wrote 0]"
        print(f"    {idx5}: CC(global)={cc_global_write:.4f}{note_g}")

        df_out.iat[row_idx, COL_BEST_LAG] = best_lag
        df_out.iat[row_idx, COL_CC_BEST]  = best_cc_written
        df_out.iat[row_idx, COL_CC_GLOB]  = cc_global_write

    df_out.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")

    # ----- Summary -----
    total = len(df_idx.iloc[:, ID_COL])
    used  = len(per_index_out)
    print(f"\n===== SUMMARY =====")
    print(f"Indices in sheet           : {total}")
    print(f"Processed (written)        : {used}")
    print(f"Skipped (no_csv)           : {skipped['no_csv']}")
    print(f"Skipped (parse_err)        : {skipped['parse_err']}")
    print(f"Skipped (missing)          : {skipped['missing_cols']}")
    print(f"Skipped (no_speeds)        : {skipped['no_speeds']}")
    print(f"Indices with any NaN in CC : {nan_any_count}")
    print(f"Global best lag            : {global_best_lag} (samples)")

if __name__ == "__main__":
    main()