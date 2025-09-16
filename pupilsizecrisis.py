#!/usr/bin/env python3
import os
import re
import csv
import numpy as np
import pandas as pd

# ======== Config (Shooter 2, old 5s-window method with new fallback) ========
INPUT_XLSX  = "Shooter 2 Data.xlsx"
OUTPUT_XLSX = "Shooter 2 Data TMP.xlsx"
ID_COL      = 0  # index column (0-based) in sheet 1

FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshookmodified",
    os.path.join("noshookmodified", "baseline"),
]

# Sheet 1 only. 1-based → 0-based columns:
# Pre (before crisis): L:31–34, R:35–38 ; Post (after crisis): L:39–42, R:43–46
PRE_L_MEAN, PRE_L_SD, PRE_L_MAX, PRE_L_MIN = 30, 31, 32, 33
PRE_R_MEAN, PRE_R_SD, PRE_R_MAX, PRE_R_MIN = 34, 35, 36, 37
POST_L_MEAN, POST_L_SD, POST_L_MAX, POST_L_MIN = 38, 39, 40, 41
POST_R_MEAN, POST_R_SD, POST_R_MAX, POST_R_MIN = 42, 43, 44, 45

WINDOW_PRE_S  = 5.0
WINDOW_POST_S = 5.0
OFFSET = 0.229  # fallback offset when “0.2 seconds” is absent

# ======== Helpers ========
def zfill5(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v)).zfill(5)
    return str(v).zfill(5)

def normalize_header_token(s: str) -> str:
    return re.sub(r"[ \t._-]+", "", str(s).strip().lower())

def find_cols_relaxed(header, want_tokens, prefer_last=None):
    """
    header: raw header strings
    want_tokens: list of normalized tokens to find (e.g., 'time','robotevent',...)
    prefer_last: set of keys for which we return the rightmost match
    returns dict {key -> idx or -1}
    """
    prefer_last = prefer_last or set()
    normed = [normalize_header_token(h) for h in header]
    out = {}
    for key in want_tokens:
        k = normalize_header_token(key)
        idx = -1
        if key in prefer_last:
            # scan all, keep last
            for j, tok in enumerate(normed):
                if k in tok:
                    idx = j
        else:
            for j, tok in enumerate(normed):
                if k in tok:
                    idx = j
                    break
        out[key] = idx
    return out

# tolerant "0.2 seconds" regex (spaces/variants)
ZERO2_REGEX = re.compile(r"\b0\s*\.?\s*2+\s*(seconds?|secs?|s)\b", re.IGNORECASE)

def find_anchors(header, rows):
    """
    Return (t0, t1) where:
      t0 = time at first row containing '0.2 seconds'
            OR (if absent) time('shook') - 0.229
      t1 = time at first row containing 'shook'
    If time column or event column missing, return (None, None).
    If neither tag exists, return (None, None).
    """
    # strict case-insensitive equality for “Time”
    def find_caseins(name):
        n = name.strip().lower()
        for i, h in enumerate(header):
            if isinstance(h, str) and h.strip().lower() == n:
                return i
        return -1

    time_idx = find_caseins("time")
    # prefer rightmost 'robotEvent' if multiple, else fallback to 'Event'
    colmap = find_cols_relaxed(header, ["robotevent", "event"], prefer_last={"robotevent"})
    evt_idx = colmap["robotevent"] if colmap["robotevent"] != -1 else colmap["event"]
    if time_idx == -1 or evt_idx == -1:
        return None, None

    t0 = None
    t1 = None

    # Find “0.2 seconds” time (t0)
    for r in rows:
        if len(r) <= max(time_idx, evt_idx):
            continue
        ev = r[evt_idx]
        if isinstance(ev, str) and ZERO2_REGEX.search(ev):
            try:
                t0 = float(r[time_idx])
            except Exception:
                t0 = None
            break

    # Find “shook” time (t1)
    for r in rows:
        if len(r) <= max(time_idx, evt_idx):
            continue
        ev = r[evt_idx]
        if isinstance(ev, str) and ("shook" in ev.strip().lower()):
            try:
                t1 = float(r[time_idx])
            except Exception:
                t1 = None
            break

    # NEW FALLBACK: if t0 missing but t1 exists, fabricate t0 = t1 - OFFSET
    if t0 is None and t1 is not None and np.isfinite(t1):
        t0 = t1 - OFFSET

    # if still missing either, caller will decide to skip
    return t0, t1

def safe_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan

def compute_stats(vec):
    """Return (mean, sd, max, min_with_nonneg) after dropping -1s."""
    vals = [v for v in vec if v != -1]
    if len(vals) < 1:
        return (np.nan, np.nan, np.nan, np.nan)
    mean_v = float(np.mean(vals)) if len(vals) else np.nan
    sd_v   = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
    max_v  = float(np.max(vals)) if len(vals) else np.nan
    nonneg = [v for v in vals if v >= 0]
    min_v  = float(np.min(nonneg)) if nonneg else np.nan
    return (mean_v, sd_v, max_v, min_v)

# ======== Main ========
def main():
    # Duplicate workbook
    if os.path.abspath(INPUT_XLSX) != os.path.abspath(OUTPUT_XLSX):
        import shutil
        shutil.copyfile(INPUT_XLSX, OUTPUT_XLSX)

    # Load Sheet 1
    try:
        xls = pd.ExcelFile(OUTPUT_XLSX, engine="openpyxl")
        sheet_names = xls.sheet_names
        df = xls.parse(sheet_names[0])
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_XLSX}': {e}")
        return

    # Ensure columns exist up to 46 (0-based 45)
    while df.shape[1] <= POST_R_MIN:
        df[f"Extra_{df.shape[1]+1}"] = np.nan

    # Optional headers
    try:
        df.columns.values[PRE_L_MEAN]  = "Mean Left Pupil (pre 5s)"
        df.columns.values[PRE_L_SD]    = "SD Left Pupil (pre 5s)"
        df.columns.values[PRE_L_MAX]   = "Max Left Pupil (pre 5s)"
        df.columns.values[PRE_L_MIN]   = "Min Left Pupil (pre 5s)"
        df.columns.values[PRE_R_MEAN]  = "Mean Right Pupil (pre 5s)"
        df.columns.values[PRE_R_SD]    = "SD Right Pupil (pre 5s)"
        df.columns.values[PRE_R_MAX]   = "Max Right Pupil (pre 5s)"
        df.columns.values[PRE_R_MIN]   = "Min Right Pupil (pre 5s)"

        df.columns.values[POST_L_MEAN] = "Mean Left Pupil (post 5s)"
        df.columns.values[POST_L_SD]   = "SD Left Pupil (post 5s)"
        df.columns.values[POST_L_MAX]  = "Max Left Pupil (post 5s)"
        df.columns.values[POST_L_MIN]  = "Min Left Pupil (post 5s)"
        df.columns.values[POST_R_MEAN] = "Mean Right Pupil (post 5s)"
        df.columns.values[POST_R_SD]   = "SD Right Pupil (post 5s)"
        df.columns.values[POST_R_MAX]  = "Max Right Pupil (post 5s)"
        df.columns.values[POST_R_MIN]  = "Min Right Pupil (post 5s)"
    except Exception:
        pass

    # Process rows
    for i in range(len(df)):
        idx = zfill5(df.iat[i, ID_COL])

        # locate CSV
        csv_path = None
        for folder in FOLDERS:
            if not os.path.isdir(folder):
                continue
            for name in os.listdir(folder):
                if name.lower().endswith(".csv") and name.startswith(idx):
                    csv_path = os.path.join(folder, name)
                    break
            if csv_path:
                break

        if not csv_path:
            print(f"{idx}: SKIP (no CSV found)")
            continue

        # read CSV
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header_raw = next(reader, [])
                if not header_raw:
                    print(f"{idx}: SKIP (empty header)")
                    continue
                rows_raw = list(reader)
                if not rows_raw:
                    print(f"{idx}: SKIP (no data rows)")
                    continue

                # map columns (prefer rightmost 'robotEvent')
                colmap = find_cols_relaxed(
                    header_raw,
                    want_tokens=[
                        "time", "robotevent", "event",
                        "leftpupildiameter", "rightpupildiameter"
                    ],
                    prefer_last={"robotevent"}
                )
                tcol  = colmap["time"]
                lcol  = colmap["leftpupildiameter"]
                rcol  = colmap["rightpupildiameter"]

                if tcol == -1 or lcol == -1 or rcol == -1:
                    print(f"{idx}: SKIP (missing time/left/right columns)")
                    continue

                # find anchors with new fallback (t0 from '0.2s' OR shook-0.229)
                t0, t1 = find_anchors(header_raw, rows_raw)
                if t1 is None:
                    print(f"{idx}: SKIP (no 'shook' found to anchor post window)")
                    continue
                if t0 is None:
                    print(f"{idx}: SKIP (no t0 anchor found or fabricated)")  # should be fabricated from t1 if only 0.2s missing
                    continue

                # collect values in the two 5s windows
                pre_L, pre_R  = [], []
                post_L, post_R = [], []
                for r in rows_raw:
                    if len(r) <= max(tcol, lcol, rcol):
                        continue
                    t = safe_float(r[tcol])
                    if not np.isfinite(t):
                        continue
                    lv = safe_float(r[lcol]) if len(r) > lcol else np.nan
                    rv = safe_float(r[rcol]) if len(r) > rcol else np.nan

                    # pre window: [t0-5, t0]
                    if (t >= (t0 - WINDOW_PRE_S)) and (t <= t0):
                        if np.isfinite(lv): pre_L.append(lv)
                        if np.isfinite(rv): pre_R.append(rv)
                    # post window: [t1, t1+5]
                    if (t >= t1) and (t <= (t1 + WINDOW_POST_S)):
                        if np.isfinite(lv): post_L.append(lv)
                        if np.isfinite(rv): post_R.append(rv)

                # stats (mean, sd, max, min>=0) dropping -1
                preL  = compute_stats(pre_L)
                preR  = compute_stats(pre_R)
                postL = compute_stats(post_L)
                postR = compute_stats(post_R)

                # write
                df.iat[i, PRE_L_MEAN] = preL[0]
                df.iat[i, PRE_L_SD]   = preL[1]
                df.iat[i, PRE_L_MAX]  = preL[2]
                df.iat[i, PRE_L_MIN]  = preL[3]

                df.iat[i, PRE_R_MEAN] = preR[0]
                df.iat[i, PRE_R_SD]   = preR[1]
                df.iat[i, PRE_R_MAX]  = preR[2]
                df.iat[i, PRE_R_MIN]  = preR[3]

                df.iat[i, POST_L_MEAN] = postL[0]
                df.iat[i, POST_L_SD]   = postL[1]
                df.iat[i, POST_L_MAX]  = postL[2]
                df.iat[i, POST_L_MIN]  = postL[3]

                df.iat[i, POST_R_MEAN] = postR[0]
                df.iat[i, POST_R_SD]   = postR[1]
                df.iat[i, POST_R_MAX]  = postR[2]
                df.iat[i, POST_R_MIN]  = postR[3]

                print(f"{idx}: pre(nL={len(pre_L)}, nR={len(pre_R)}) "
                      f"post(nL={len(post_L)}, nR={len(post_R)})")

        except Exception as e:
            print(f"{idx}: SKIP (CSV read error: {e})")
            continue

    # save back
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            # keep sheet order; only sheet 1 is updated here
            df.to_excel(writer, sheet_name=sheet_names[0], index=False)
            for s in sheet_names[1:]:
                xls.parse(s).to_excel(writer, sheet_name=s, index=False)
        print(f"\n✅ Done — wrote pre/post pupil stats (5s windows, 0.2s←→shook fallback) to Sheet1 cols 31–46 in '{OUTPUT_XLSX}'.")
    except Exception as e:
        print(f"❌ Failed to write '{OUTPUT_XLSX}': {e}")

if __name__ == "__main__":
    main()