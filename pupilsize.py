#!/usr/bin/env python3
import os
import shutil
import numpy as np
import pandas as pd

# ========= Config =========
INPUT_EXCEL  = "Shooter 2 Data.xlsx"
OUTPUT_EXCEL = "Shooter 2 Data Proceed.xlsx"

# CSV search locations (in order)
SEARCH_DIRS = [
    "shook",
    "noshook",
    os.path.join("shook", "baseline"),
    os.path.join("noshook", "baseline"),
]

# Sheet 1 layout (0-based)
ID_COL     = 0   # index column
CRISIS_COL = 3   # crisis time column (4th Excel column)

# Output columns on Sheet 1 (0-based indices)
# 5-second windows: AE..AT → 30..45
OUT_5S = {
    "preL":  [30, 31, 32, 33],  # AE, AF, AG, AH  → mean, SD, max, min
    "preR":  [34, 35, 36, 37],  # AI, AJ, AK, AL
    "postL": [38, 39, 40, 41],  # AM, AN, AO, AP
    "postR": [42, 43, 44, 45],  # AQ, AR, AS, AT
}
# Full windows: AU..BJ → 46..61
OUT_FULL = {
    "preL":  [46, 47, 48, 49],  # AU, AV, AW, AX
    "preR":  [50, 51, 52, 53],  # AY, AZ, BA, BB
    "postL": [54, 55, 56, 57],  # BC, BD, BE, BF
    "postR": [58, 59, 60, 61],  # BG, BH, BI, BJ
}

# ========= Helpers =========
def index_key(val) -> str:
    """
    5-char key for CSV:
    - If the cell is an int-like number, zero-pad to 5 (123 → '00123').
    - Else, take the first 5 visible characters (trimmed).
    """
    if isinstance(val, (int, np.integer)):
        return str(int(val)).zfill(5)
    if isinstance(val, float) and float(val).is_integer():
        return str(int(val)).zfill(5)
    s = str(val).strip()
    if s.isdigit():
        return s.zfill(5) if len(s) < 5 else s[:5]
    return s[:5]

def find_matching_csv(idx5: str):
    """Return the first CSV path (by SEARCH_DIRS order) whose name starts with idx5, else None."""
    for folder in SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name)
    return None

def norm_token(s: str) -> str:
    """Normalize a header token: lowercase, remove spaces/._-."""
    return "".join(ch for ch in str(s).lower().strip() if ch not in " ._-")

def find_time_col(header) -> int:
    toks = [norm_token(h) for h in header]
    for i, tok in enumerate(toks):
        if tok == "time":
            return i
    return -1

def find_pupil_cols(header) -> tuple[int, int]:
    """Return (left_idx, right_idx) by relaxed matching; (-1, -1) if missing."""
    left = right = -1
    toks = [norm_token(h) for h in header]
    for i, tok in enumerate(toks):
        if left  == -1 and "leftpupil"  in tok: left  = i
        if right == -1 and "rightpupil" in tok: right = i
    return left, right

def compute_stats(values: list[float]) -> list[float]:
    """
    Return [mean, SD(ddof=1), max, min] after:
      - excluding all -1 values,
      - SD requires ≥2 usable values else NaN,
      - min across only values ≥ 0 (NaN if none).
    """
    vals = [v for v in values if v != -1]
    if len(vals) == 0:
        return [np.nan, np.nan, np.nan, np.nan]
    arr = np.asarray(vals, dtype=float)
    mean_v = float(np.mean(arr))
    sd_v   = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    max_v  = float(np.max(arr))
    nonneg = arr[arr >= 0]
    min_v  = float(np.min(nonneg)) if nonneg.size > 0 else float("nan")
    return [mean_v, sd_v, max_v, min_v]

def ensure_width(df: pd.DataFrame, upto_0based: int) -> pd.DataFrame:
    """Ensure df has columns up to index `upto_0based` (0-based)."""
    while df.shape[1] <= upto_0based:
        df[f"Extra_{df.shape[1]+1}"] = np.nan
    return df

# ========= Main =========
def main():
    # Duplicate workbook
    shutil.copyfile(INPUT_EXCEL, OUTPUT_EXCEL)

    # Load sheet 1 (indices + crisis time)
    try:
        df = pd.read_excel(OUTPUT_EXCEL, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_EXCEL}': {e}")
        return

    # Ensure width up to BJ (0-based 61)
    df = ensure_width(df, 61)

    # Pre-clear target columns (5s + full)
    for idxs in list(OUT_5S.values()) + list(OUT_FULL.values()):
        for c in idxs:
            df.iloc[:, c] = np.nan

    # Process each row
    for i in range(len(df)):
        idx5 = index_key(df.iat[i, ID_COL])

        # Crisis time from sheet 1 col 4
        try:
            crisis_time = float(df.iat[i, CRISIS_COL])
        except Exception:
            crisis_time = np.nan

        if not np.isfinite(crisis_time):
            print(f"{idx5}: SKIP — crisis time NaN on sheet 1")
            continue

        csv_path = find_matching_csv(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP — no matching CSV in {SEARCH_DIRS}")
            continue

        # Load CSV
        try:
            raw = pd.read_csv(csv_path, engine="python", on_bad_lines="skip", dtype=str)
        except Exception as e:
            print(f"{idx5}: SKIP — CSV read error: {e}")
            continue
        if raw.shape[0] == 0:
            print(f"{idx5}: SKIP — empty CSV")
            continue

        header = list(raw.columns)
        t_idx = find_time_col(header)
        l_idx, r_idx = find_pupil_cols(header)

        if t_idx == -1 or l_idx == -1 or r_idx == -1:
            missing = []
            if t_idx == -1: missing.append("Time")
            if l_idx == -1: missing.append("leftPupil*")
            if r_idx == -1: missing.append("rightPupil*")
            print(f"{idx5}: SKIP — missing columns: {', '.join(missing)}")
            continue

        # Extract numeric series
        t_ser = pd.to_numeric(raw.iloc[:, t_idx], errors="coerce").to_numpy()
        L_ser = pd.to_numeric(raw.iloc[:, l_idx], errors="coerce").to_numpy()
        R_ser = pd.to_numeric(raw.iloc[:, r_idx], errors="coerce").to_numpy()

        # Masks
        isfinite_t = np.isfinite(t_ser)

        # ---- 5-second windows ----
        pre5_mask  = isfinite_t & (t_ser >= crisis_time - 5.0) & (t_ser <  crisis_time)      # [crisis-5, crisis)
        post5_mask = isfinite_t & (t_ser >= crisis_time)        & (t_ser <= crisis_time + 5) # [crisis, crisis+5]

        pre5_left   = [float(v) for v in L_ser[pre5_mask]   if np.isfinite(v)]
        pre5_right  = [float(v) for v in R_ser[pre5_mask]   if np.isfinite(v)]
        post5_left  = [float(v) for v in L_ser[post5_mask]  if np.isfinite(v)]
        post5_right = [float(v) for v in R_ser[post5_mask]  if np.isfinite(v)]

        pre5L_stats  = compute_stats(pre5_left)
        pre5R_stats  = compute_stats(pre5_right)
        post5L_stats = compute_stats(post5_left)
        post5R_stats = compute_stats(post5_right)

        # ---- Full intervals ----
        pre_mask   = isfinite_t & (t_ser <  crisis_time)
        post_mask  = isfinite_t & (t_ser >= crisis_time)

        pre_left   = [float(v) for v in L_ser[pre_mask]   if np.isfinite(v)]
        pre_right  = [float(v) for v in R_ser[pre_mask]   if np.isfinite(v)]
        post_left  = [float(v) for v in L_ser[post_mask]  if np.isfinite(v)]
        post_right = [float(v) for v in R_ser[post_mask]  if np.isfinite(v)]

        full_preL_stats  = compute_stats(pre_left)
        full_preR_stats  = compute_stats(pre_right)
        full_postL_stats = compute_stats(post_left)
        full_postR_stats = compute_stats(post_right)

        # ---- Write to AE..AT (5s) and AU..BJ (full) ----
        for col_idx, val in zip(OUT_5S["preL"],  pre5L_stats):   df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_5S["preR"],  pre5R_stats):   df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_5S["postL"], post5L_stats):  df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_5S["postR"], post5R_stats):  df.iat[i, col_idx] = val

        for col_idx, val in zip(OUT_FULL["preL"],  full_preL_stats):   df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_FULL["preR"],  full_preR_stats):   df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_FULL["postL"], full_postL_stats):  df.iat[i, col_idx] = val
        for col_idx, val in zip(OUT_FULL["postR"], full_postR_stats):  df.iat[i, col_idx] = val

        print(f"{idx5}: OK — "
              f"5s pre(nL={len(pre5_left)}, nR={len(pre5_right)}), "
              f"5s post(nL={len(post5_left)}, nR={len(post5_right)}), "
              f"full pre(nL={len(pre_left)}, nR={len(pre_right)}), "
              f"full post(nL={len(post_left)}, nR={len(post_right)})")

    # Save
    try:
        df.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
        print(f"\n✅ Done! Wrote pupil stats (5s & full) to AE..BJ in '{OUTPUT_EXCEL}'.")
    except Exception as e:
        print(f"❌ Failed to save '{OUTPUT_EXCEL}': {e}")

if __name__ == "__main__":
    main()