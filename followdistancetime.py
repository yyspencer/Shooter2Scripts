#!/usr/bin/env python3
import os
import re
import csv
import math
import shutil
import numpy as np
import pandas as pd

# ========= Config (Shooter 2) =========
SOURCE_XLSX = "Shooter 2 Data.xlsx"
OUTPUT_XLSX = "Shooter 2 Follow.xlsx"

# CSV search folders (order matters)
FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshookmodified",
    os.path.join("noshookmodified", "baseline"),
]

# Excel: index column on sheet 1 (0-based)
ID_COL = 0
# Crisis time on sheet 1 (1-based 4th col -> 0-based index 3)
CRISIS_COL = 3

# Sheet/column targets (Excel 1-based → 0-based)
# Sheet 1 (overview): Dist, Time → cols 21–22 → 20..21
S1_COL_DIST, S1_COL_TIME = 20, 21
# Sheets 2–5 (pre_no, pre_med, post_no, post_med): Dist, Time → cols 19–20 → 18..19
S_COL_DIST, S_COL_TIME = 18, 19

# ========= Header utilities =========
def norm_token(s: str) -> str:
    """Lowercase, trim, remove spaces/._- for robust header matching."""
    return re.sub(r"[ \t._-]+", "", str(s).strip().lower())

def find_col_relaxed(header, target):
    """First match by relaxed token equality; return index or -1."""
    t = norm_token(target)
    for i, h in enumerate(header):
        if norm_token(h) == t:
            return i
    return -1

# ========= Index & file utils =========
def idx5_from_cell(v) -> str:
    """Use first 5 characters of the cell value (trimmed)."""
    return str(v).strip()[:5]

def find_matching_csv(idx5: str):
    """Return (path, folder) for CSV whose name starts with idx5 (in FOLDERS order)."""
    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

# ========= Toggles (Survey & Meditation) =========
def build_toggles(header, rows):
    """
    Build boolean arrays:
      in_survey:  based on 'entered/exited survey room'
      in_med:     based on 'entered/exited meditation area'
    Prefer roomEvent; fallback Event.
    """
    evt_idx = find_col_relaxed(header, "roomEvent")
    if evt_idx == -1:
        evt_idx = find_col_relaxed(header, "Event")

    n = len(rows)
    in_survey = np.zeros(n, dtype=bool)
    in_med    = np.zeros(n, dtype=bool)

    if evt_idx == -1:
        return in_survey, in_med

    state_s = False
    state_m = False
    for i in range(n):
        s = (rows[i][evt_idx] if len(rows[i]) > evt_idx else "")
        s = str(s).strip().lower()
        if "entered survey room" in s:
            state_s = True
        elif "exited survey room" in s:
            state_s = False
        if "entered meditation area" in s:
            state_m = True
        elif "exited meditation area" in s:
            state_m = False
        in_survey[i] = state_s
        in_med[i]    = state_m

    return in_survey, in_med

# ========= Follow computation (future-allowed, side-locked) =========
PROXIMITY_THRESHOLD = 2.0   # meters
FOLLOW_WINDOW       = 10.0  # seconds

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def compute_follow_for_mask_side_locked(rows, times, mask, crisis_time, side):
    """
    Original matching (scan from end; future allowed; 10s window), but *side-locked*:
      - side='pre'  : both robot row i and matched player j must have t < crisis_time
      - side='post' : both i and j must have t >= crisis_time
    `mask` selects which robot rows to accumulate (category).
    rows[k] = (t, player_xyz, robot_xyz, room_evt_lower, raw_row)
    """
    assert side in ("pre", "post")
    def on_side(tval):
        return (tval < crisis_time) if side == "pre" else (tval >= crisis_time)

    followed_distance = 0.0
    followed_time     = 0.0
    prev_pos  = None
    prev_time = None

    n = len(rows)
    t_arr = times
    p_arr = np.array([rows[i][1] for i in range(n)], dtype=float)
    r_arr = np.array([rows[i][2] for i in range(n)], dtype=float)
    evt_s = [rows[i][3] for i in range(n)]  # survey event string (lowercased)

    for i in range(n):
        if not mask[i]:
            continue
        t = t_arr[i]
        if not np.isfinite(t) or not on_side(t):
            continue

        # Survey Room exclusion (and reset at boundaries)
        if evt_s[i] == "entered survey room":
            prev_pos = prev_time = None
            continue
        elif evt_s[i] == "exited survey room":
            prev_pos = prev_time = None
            continue
        # If currently inside survey we shouldn't have mask True, but keep the reset logic

        robot_p = r_arr[i]

        # Matching: scan from end; j must be on the same side
        matched = None
        for j in range(n-1, -1, -1):
            tj = t_arr[j]
            if not np.isfinite(tj) or not on_side(tj):
                continue
            dt = t - tj
            if dt > FOLLOW_WINDOW:
                break  # too far back in time (past 10 s)
            # future (dt < 0) allowed
            if euclidean_distance(p_arr[j], robot_p) <= PROXIMITY_THRESHOLD:
                matched = (tj, p_arr[j])
                break

        if matched:
            tj, player_j = matched
            if prev_pos is not None:
                followed_distance += euclidean_distance(prev_pos, player_j)
                followed_time     += (t - prev_time)
            prev_pos  = player_j
            prev_time = t

    return followed_distance, followed_time

# ========= Main =========
def main():
    # Duplicate workbook so we never touch the original
    try:
        shutil.copyfile(SOURCE_XLSX, OUTPUT_XLSX)
    except Exception as e:
        print(f"❌ Failed to copy '{SOURCE_XLSX}' → '{OUTPUT_XLSX}': {e}")
        return

    # Load all five sheets (overview + 4 categories) to preserve names/order
    try:
        xls = pd.ExcelFile(OUTPUT_XLSX, engine="openpyxl")
        sheet_names = xls.sheet_names
        if len(sheet_names) < 5:
            print(f"❌ Expected at least 5 sheets; found {len(sheet_names)}.")
            return
        df_over    = xls.parse(sheet_names[0])  # overview: indices + crisis times
        df_pre_no  = xls.parse(sheet_names[1])
        df_pre_med = xls.parse(sheet_names[2])
        df_post_no = xls.parse(sheet_names[3])
        df_post_md = xls.parse(sheet_names[4])
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_XLSX}': {e}")
        return

    # Ensure destination columns exist
    for df_s, upto in [
        (df_over,   max(S1_COL_DIST, S1_COL_TIME)),
        (df_pre_no, max(S_COL_DIST,  S_COL_TIME)),
        (df_pre_med,max(S_COL_DIST,  S_COL_TIME)),
        (df_post_no,max(S_COL_DIST,  S_COL_TIME)),
        (df_post_md,max(S_COL_DIST,  S_COL_TIME)),
    ]:
        while df_s.shape[1] <= upto:
            df_s[f"Extra_{df_s.shape[1]+1}"] = np.nan

    # Optional headers
    try:
        df_over.columns.values[S1_COL_DIST] = "Followed Distance (m) — Overall"
        df_over.columns.values[S1_COL_TIME] = "Followed Time (s) — Overall"
        df_pre_no.columns.values[S_COL_DIST]  = "Followed Distance (m) — Pre, No Med"
        df_pre_no.columns.values[S_COL_TIME]  = "Followed Time (s) — Pre, No Med"
        df_pre_med.columns.values[S_COL_DIST] = "Followed Distance (m) — Pre, Med"
        df_pre_med.columns.values[S_COL_TIME] = "Followed Time (s) — Pre, Med"
        df_post_no.columns.values[S_COL_DIST] = "Followed Distance (m) — Post, No Med"
        df_post_no.columns.values[S_COL_TIME] = "Followed Time (s) — Post, No Med"
        df_post_md.columns.values[S_COL_DIST] = "Followed Distance (m) — Post, Med"
        df_post_md.columns.values[S_COL_TIME] = "Followed Time (s) — Post, Med"
    except Exception:
        pass

    print("Index | overall(D,T) = sum of 4 categories | pre_no(D,T) | pre_med(D,T) | post_no(D,T) | post_med(D,T)")
    print("-"*120)

    # Process each index from sheet 1
    for i in range(len(df_over)):
        idx5 = idx5_from_cell(df_over.iat[i, ID_COL])

        # Crisis time from sheet 1 col 4
        try:
            crisis_time = float(df_over.iat[i, CRISIS_COL])
        except Exception:
            crisis_time = np.nan

        csv_path, folder = find_matching_csv(idx5)
        if not csv_path or not np.isfinite(crisis_time):
            # Write NaNs across the board
            for df_s, (cD, cT) in [
                (df_over, (S1_COL_DIST, S1_COL_TIME)),
                (df_pre_no, (S_COL_DIST, S_COL_TIME)),
                (df_pre_med,(S_COL_DIST, S_COL_TIME)),
                (df_post_no,(S_COL_DIST, S_COL_TIME)),
                (df_post_md,(S_COL_DIST, S_COL_TIME)),
            ]:
                df_s.iat[i, cD] = np.nan
                df_s.iat[i, cT] = np.nan
            print(f"{idx5}: SKIP — {'no CSV' if not csv_path else 'crisis_time NaN'}")
            continue

        # Read CSV
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header_raw = next(reader, [])
                if not header_raw:
                    raise ValueError("empty header")
                header = [h.strip() if isinstance(h, str) else h for h in header_raw]
                rows_raw = list(reader)
                if not rows_raw:
                    raise ValueError("no data rows")

                # Required columns (relaxed)
                time_idx = find_col_relaxed(header, "Time")
                px_idx   = find_col_relaxed(header, "PlayerVR.x")
                py_idx   = find_col_relaxed(header, "PlayerVR.y")
                pz_idx   = find_col_relaxed(header, "PlayerVR.z")
                rx_idx   = find_col_relaxed(header, "Robot.x")
                ry_idx   = find_col_relaxed(header, "Robot.y")
                rz_idx   = find_col_relaxed(header, "Robot.z")
                if min(time_idx, px_idx, py_idx, pz_idx, rx_idx, ry_idx, rz_idx) < 0:
                    raise ValueError("missing position/time column(s)")

                # Build toggles (Survey + Meditation) from roomEvent (fallback Event)
                in_survey, in_med = build_toggles(header, rows_raw)

                # Parse arrays
                def sf(s):
                    try: return float(s)
                    except Exception: return np.nan

                n = len(rows_raw)
                times = np.array([sf(r[time_idx]) if len(r) > time_idx else np.nan for r in rows_raw], dtype=float)

                p_arr = np.array([
                    [sf(r[px_idx]), sf(r[py_idx]), sf(r[pz_idx])] if len(r) > max(px_idx,py_idx,pz_idx) else [np.nan]*3
                for r in rows_raw], dtype=float)

                r_arr = np.array([
                    [sf(r[rx_idx]), sf(r[ry_idx]), sf(r[rz_idx])] if len(r) > max(rx_idx,ry_idx,rz_idx) else [np.nan]*3
                for r in rows_raw], dtype=float)

                # Normalize survey event string for reset logic in follow fn
                # We'll store "entered survey room"/"exited survey room" literals for rows; otherwise "".
                evt_lower = []
                state_s = False
                for j in range(n):
                    s = ""
                    # We already computed in_survey; reconstruct boundary markers for reset:
                    if j == 0:
                        if in_survey[j]:
                            s = "entered survey room"
                    else:
                        if (not in_survey[j-1]) and in_survey[j]:
                            s = "entered survey room"
                        elif in_survey[j-1] and (not in_survey[j]):
                            s = "exited survey room"
                    evt_lower.append(s)

                # Valid robot rows (finite pos/time and not in survey)
                finite_pos = np.isfinite(p_arr).all(axis=1) & np.isfinite(r_arr).all(axis=1)
                valid_rows = np.isfinite(times) & finite_pos & (~in_survey)

                # Pack rows
                rows = [(times[k], p_arr[k], r_arr[k], evt_lower[k], rows_raw[k]) for k in range(n)]

                # Category masks (disjoint)
                pre_mask      = (times <  crisis_time) & valid_rows
                post_mask     = (times >= crisis_time) & valid_rows
                pre_no_mask   = pre_mask  & (~in_med)
                pre_med_mask  = pre_mask  & ( in_med)
                post_no_mask  = post_mask & (~in_med)
                post_med_mask = post_mask & ( in_med)

                # Compute category distances/times (side-locked)
                D_pn, T_pn = compute_follow_for_mask_side_locked(rows, times, pre_no_mask,  crisis_time, side="pre")
                D_pm, T_pm = compute_follow_for_mask_side_locked(rows, times, pre_med_mask, crisis_time, side="pre")
                D_tn, T_tn = compute_follow_for_mask_side_locked(rows, times, post_no_mask, crisis_time, side="post")
                D_tm, T_tm = compute_follow_for_mask_side_locked(rows, times, post_med_mask,crisis_time, side="post")

                # Overall = sum of 4 categories (guaranteed equality)
                D_over = D_pn + D_pm + D_tn + D_tm
                T_over = T_pn + T_pm + T_tn + T_tm

                # Write outputs
                df_over.iat[i, S1_COL_DIST] = D_over
                df_over.iat[i, S1_COL_TIME] = T_over

                df_pre_no.iat[i,  S_COL_DIST] = D_pn
                df_pre_no.iat[i,  S_COL_TIME] = T_pn
                df_pre_med.iat[i, S_COL_DIST] = D_pm
                df_pre_med.iat[i, S_COL_TIME] = T_pm
                df_post_no.iat[i, S_COL_DIST] = D_tn
                df_post_no.iat[i, S_COL_TIME] = T_tn
                df_post_md.iat[i, S_COL_DIST] = D_tm
                df_post_md.iat[i, S_COL_TIME] = T_tm

                print(f"{idx5}: overall(D={D_over:.4f}, T={T_over:.4f}) | "
                      f"pre_no(D={D_pn:.4f}, T={T_pn:.4f}) | "
                      f"pre_med(D={D_pm:.4f}, T={T_pm:.4f}) | "
                      f"post_no(D={D_tn:.4f}, T={T_tn:.4f}) | "
                      f"post_med(D={D_tm:.4f}, T={T_tm:.4f})")

        except Exception as e:
            # mark NaNs and continue
            for df_s, (cD, cT) in [
                (df_over,   (S1_COL_DIST, S1_COL_TIME)),
                (df_pre_no, (S_COL_DIST,  S_COL_TIME)),
                (df_pre_med,(S_COL_DIST,  S_COL_TIME)),
                (df_post_no,(S_COL_DIST,  S_COL_TIME)),
                (df_post_md,(S_COL_DIST,  S_COL_TIME)),
            ]:
                df_s.iat[i, cD] = np.nan
                df_s.iat[i, cT] = np.nan
            print(f"{idx5}: CSV error ({e})")
            continue

    # Save all five sheets
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            df_over.to_excel(writer,   sheet_name=sheet_names[0], index=False)
            df_pre_no.to_excel(writer, sheet_name=sheet_names[1], index=False)
            df_pre_med.to_excel(writer, sheet_name=sheet_names[2], index=False)
            df_post_no.to_excel(writer, sheet_name=sheet_names[3], index=False)
            df_post_md.to_excel(writer, sheet_name=sheet_names[4], index=False)
    except Exception as e:
        print(f"❌ Failed to save '{OUTPUT_XLSX}': {e}")
        return

    print("\n✅ Done. Wrote: Sheet1 cols 21–22 (overall), Sheets2–5 cols 19–20 (pre/post × med/no-med).")

if __name__ == "__main__":
    main()