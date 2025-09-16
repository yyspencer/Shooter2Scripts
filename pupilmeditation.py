#!/usr/bin/env python3
import os
import re
import unicodedata
import csv
import numpy as np
import pandas as pd

# ========= Config =========
INPUT_XLSX  = "Shooter 2 Data.xlsx"
OUTPUT_XLSX = "Shooter 2 TMP.xlsx"
ID_COL = 0  # indices column (0-based) — same ordering across all sheets

SEARCH_FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshookmodified",
    os.path.join("noshookmodified", "baseline"),
]

# Sheets: 2..5 (1-based) -> indices 1..4 in pandas list order
SHEET_IDX_PRE_NO   = 1  # pre crisis, no meditation
SHEET_IDX_PRE_MED  = 2  # pre crisis, meditation
SHEET_IDX_POST_NO  = 3  # post crisis, no meditation
SHEET_IDX_POST_MED = 4  # post crisis, meditation

# Output columns (Excel 1-based spec → 0-based)
# You set 20..27 (i.e., Excel 21..28)
COL_L_MEAN = 20
COL_L_SD   = 21
COL_L_MAX  = 22
COL_L_MIN  = 23
COL_R_MEAN = 24
COL_R_SD   = 25
COL_R_MAX  = 26
COL_R_MIN  = 27

# ====== Helpers ======
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_header_token(s: str) -> str:
    # lowercase, strip, and remove spaces/underscore/dot/hyphen to be robust
    return re.sub(r"[ _.\-]+", "", normalize_text(s).lower())

def find_columns(header, want_keys, prefer_last_keys=frozenset()):
    """
    Relaxed substring search on normalized tokens (case-insensitive).

    For keys in `prefer_last_keys` (special-case: 'robotevent'), return the
    RIGHTMOST matching column index. For all others, return the FIRST match.

    Return {key -> col_idx or -1}.
    """
    normed = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        idx = -1
        if key in prefer_last_keys:
            for j, tok in enumerate(normed):
                if k in tok:
                    idx = j  # keep last
            out[key] = idx
        else:
            for j, tok in enumerate(normed):
                if k in tok:
                    idx = j
                    break
            out[key] = idx
    return out

def index5(v) -> str:
    """First 5 chars; zero-pad if purely numeric and <5 digits."""
    if isinstance(v, (int, np.integer)):
        s = str(int(v)).zfill(5)
    elif isinstance(v, float) and v.is_integer():
        s = str(int(v)).zfill(5)
    else:
        s = str(v).strip()
        if s.isdigit() and len(s) < 5:
            s = s.zfill(5)
    return s[:5]

def find_matching_csv(idx5: str):
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name[:5] == idx5:
                return os.path.join(folder, name), folder
    return None, None

def crisis_time_from_event_column(times, col_vals):
    """
    Priority:
      1) first row containing 'shook'  -> t[row]
      2) else first row containing '0.2 seconds' -> t[row] + 0.229
      3) else None
    Case-insensitive substring search.
    """
    ser = pd.Series(col_vals, dtype="string").str.lower()
    shook_idx = ser[ser.str.contains("shook", regex=False, na=False)].index
    if len(shook_idx) > 0:
        i = int(shook_idx[0])
        if np.isfinite(times[i]):
            return float(times[i]), "shook"

    zero_idx = ser[ser.str.contains("0.2 seconds", regex=False, na=False)].index
    if len(zero_idx) > 0:
        i = int(zero_idx[0])
        if np.isfinite(times[i]):
            return float(times[i]) + 0.229, "0.2 seconds"

    return None, None

def build_meditation_toggles(col_vals):
    """
    Build row index -> 'enter'/'exit' from col values using:
      'entered meditation area' / 'exited meditation area' (case-insensitive substrings).
    """
    toggles = {}
    ser = pd.Series(col_vals, dtype="string").str.lower()
    ent_idx = ser[ser.str.contains("entered meditation area", regex=False, na=False)].index
    ex_idx  = ser[ser.str_contains if hasattr(ser, "str_contains") else ser.str.contains]("exited meditation area", regex=False, na=False) if False else ser[ser.str.contains("exited meditation area", regex=False, na=False)].index
    for i in ent_idx:
        toggles[int(i)] = "enter"
    for i in ex_idx:
        toggles[int(i)] = "exit"
    return toggles

def compute_stats(values):
    """
    Match your old script:
    - Drop all -1s entirely from all stats.
    - Min uses only values >= 0.
    - Require >=2 values to compute all four numbers; otherwise return NaNs.
    """
    vals = [v for v in values if v != -1]
    if len(vals) < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    mean_v = float(np.mean(vals))
    sd_v   = float(np.std(vals, ddof=1))
    max_v  = float(np.max(vals))
    nonneg = [v for v in vals if v >= 0]
    min_v  = float(np.min(nonneg)) if nonneg else np.nan
    return (mean_v, sd_v, max_v, min_v)

def ensure_width(df: pd.DataFrame, upto_index_inclusive: int) -> pd.DataFrame:
    """
    Grow df so that column index `upto_index_inclusive` is valid (i.e., df has at least upto+1 columns).
    """
    while df.shape[1] <= upto_index_inclusive:
        df[f"Extra_{df.shape[1]+1}"] = np.nan
    return df

# ========= Main =========
def main():
    # Copy workbook and load all sheets
    if os.path.abspath(INPUT_XLSX) != os.path.abspath(OUTPUT_XLSX):
        try:
            import shutil
            shutil.copyfile(INPUT_XLSX, OUTPUT_XLSX)
        except Exception as e:
            print(f"❌ Failed to duplicate workbook: {e}")
            return

    try:
        xls = pd.ExcelFile(OUTPUT_XLSX, engine="openpyxl")
        sheet_names = xls.sheet_names
        if len(sheet_names) < 5:
            print(f"❌ Expected at least 5 sheets; found {len(sheet_names)}.")
            return
        # Load sheets as DataFrames
        dfs = {name: xls.parse(name) for name in sheet_names}
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_XLSX}': {e}")
        return

    # First sheet: source of indices & row order
    sheet1_name = sheet_names[0]
    df_idx = dfs[sheet1_name]
    if df_idx.shape[1] <= ID_COL:
        print("❌ First sheet has no indices column.")
        return

    # *** FIX: ensure width up to the rightmost target column (COL_R_MIN) ***
    target_last_col = COL_R_MIN  # 0-based last column we will write
    for si in [SHEET_IDX_PRE_NO, SHEET_IDX_PRE_MED, SHEET_IDX_POST_NO, SHEET_IDX_POST_MED]:
        sname = sheet_names[si]
        dfs[sname] = ensure_width(dfs[sname], target_last_col)

    # Diagnostics trackers
    missing_pupil_detail = []     # (idx5, folder, ['leftPupilDiameter', ...])
    missing_crisis_detail = []    # (idx5, folder, event_col_used or None)
    missing_time_detail   = []    # (idx5, folder)
    used_event_fallback   = []    # (idx5, folder)
    missing_both_events   = []    # (idx5, folder)

    indices = df_idx.iloc[:, ID_COL].apply(index5)

    for row_i, raw in enumerate(indices):
        idx5 = raw
        csv_path, folder = find_matching_csv(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP (no CSV found in search folders)")
            continue

        # Read CSV
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    print(f"{idx5} [{folder}]: SKIP (empty CSV)")
                    continue

                # prefer RIGHTMOST 'robotevent'
                header_norm = [normalize_text(h) for h in header]
                colmap = find_columns(
                    header_norm,
                    want_keys=[
                        "time",
                        "robotevent",
                        "roomevent",
                        "event",
                        "leftpupildiameter",
                        "rightpupildiameter",
                    ],
                    prefer_last_keys=frozenset({"robotevent"})
                )

                # Time column required
                if colmap["time"] == -1:
                    print(f"{idx5} [{folder}]: SKIP (missing 'Time' column)")
                    missing_time_detail.append((idx5, folder))
                    continue
                t_col = colmap["time"]

                # Crisis detection column: robotevent (rightmost) preferred, else event
                crisis_col_idx = colmap["robotevent"] if colmap["robotevent"] != -1 else colmap["event"]
                crisis_col_name = header[crisis_col_idx] if crisis_col_idx != -1 else None

                # Meditation toggles column: roomEvent preferred, else Event
                med_col_idx = colmap["roomevent"]
                used_fallback = False
                if med_col_idx == -1:
                    med_col_idx = colmap["event"]
                    if med_col_idx != -1:
                        used_fallback = True

                # Pupil columns
                l_idx = colmap["leftpupildiameter"]
                r_idx = colmap["rightpupildiameter"]
                missing_cols = []
                if l_idx == -1:
                    missing_cols.append("leftPupilDiameter")
                if r_idx == -1:
                    missing_cols.append("rightPupilDiameter")
                if missing_cols:
                    print(f"{idx5} [{folder}]: MISSING pupil columns -> {', '.join(missing_cols)}")
                    missing_pupil_detail.append((idx5, folder, missing_cols))
                # Read rows
                rows = list(reader)
                nrows = len(rows)
                if nrows == 0:
                    print(f"{idx5} [{folder}]: SKIP (no data rows)")
                    continue

                def safe_float(s):
                    try:
                        return float(s)
                    except Exception:
                        return np.nan

                times = np.array([safe_float(r[t_col]) if len(r) > t_col else np.nan for r in rows], dtype=float)

                # crisis-time detection
                crisis_time = None
                crisis_src  = None
                if crisis_col_idx != -1:
                    crisis_vals = [rows[i][crisis_col_idx] if len(rows[i]) > crisis_col_idx else "" for i in range(nrows)]
                    crisis_time, crisis_src = crisis_time_from_event_column(times, crisis_vals)
                else:
                    missing_crisis_detail.append((idx5, folder, None))
                    print(f"{idx5} [{folder}]: NO CRISIS TAG search (no 'robotEvent' nor 'Event' column)")
                    continue

                if crisis_time is None:
                    missing_crisis_detail.append((idx5, folder, crisis_col_name if crisis_col_name else "no event col"))
                    print(f"{idx5} [{folder}]: NO CRISIS TAG in '{crisis_col_name or 'N/A'}' (searched 'shook' / '0.2 seconds')")
                    continue

                # Meditation toggles
                toggles = {}
                if med_col_idx != -1:
                    med_vals = [rows[i][med_col_idx] if len(rows[i]) > med_col_idx else "" for i in range(nrows)]
                    toggles = build_meditation_toggles(med_vals)
                    if used_fallback:
                        used_event_fallback.append((idx5, folder))
                else:
                    missing_both_events.append((idx5, folder))

                # Gather values per category
                pre_no_L, pre_no_R   = [], []
                pre_med_L, pre_med_R = [], []
                post_no_L, post_no_R = [], []
                post_med_L, post_med_R = [], []

                in_med = False
                for i in range(nrows):
                    # Apply toggle at row i BEFORE assigning this sample
                    act = toggles.get(i)
                    if act == "enter": in_med = True
                    elif act == "exit": in_med = False

                    ti = times[i]
                    if not np.isfinite(ti):
                        continue

                    # Skip if pupil columns missing
                    lv = rows[i][l_idx] if l_idx != -1 and len(rows[i]) > l_idx else ""
                    rv = rows[i][r_idx] if r_idx != -1 and len(rows[i]) > r_idx else ""

                    try:
                        lval = float(lv) if lv != "" else np.nan
                    except Exception:
                        lval = np.nan
                    try:
                        rval = float(rv) if rv != "" else np.nan
                    except Exception:
                        rval = np.nan

                    if not np.isfinite(lval) and not np.isfinite(rval):
                        continue

                    pre = (ti <= crisis_time)  # tie → pre
                    if pre and not in_med:
                        if np.isfinite(lval): pre_no_L.append(lval)
                        if np.isfinite(rval): pre_no_R.append(rval)
                    elif pre and in_med:
                        if np.isfinite(lval): pre_med_L.append(lval)
                        if np.isfinite(rval): pre_med_R.append(rval)
                    elif (not pre) and not in_med:
                        if np.isfinite(lval): post_no_L.append(lval)
                        if np.isfinite(rval): post_no_R.append(rval)
                    else:
                        if np.isfinite(lval): post_med_L.append(lval)
                        if np.isfinite(rval): post_med_R.append(rval)

                # Compute stats for each category (Left & Right)
                stats_map = {
                    SHEET_IDX_PRE_NO:   (compute_stats(pre_no_L),  compute_stats(pre_no_R)),
                    SHEET_IDX_PRE_MED:  (compute_stats(pre_med_L), compute_stats(pre_med_R)),
                    SHEET_IDX_POST_NO:  (compute_stats(post_no_L), compute_stats(post_no_R)),
                    SHEET_IDX_POST_MED: (compute_stats(post_med_L),compute_stats(post_med_R)),
                }

                # Write into sheets 2–5 (same row index as sheet 1)
                for si, (Lstats, Rstats) in stats_map.items():
                    sname = sheet_names[si]
                    df_s = dfs[sname]
                    # *** FIX: ensure width up to our farthest output col ***
                    df_s = ensure_width(df_s, COL_R_MIN)

                    l_mean, l_sd, l_max, l_min = Lstats
                    r_mean, r_sd, r_max, r_min = Rstats

                    df_s.iat[row_i, COL_L_MEAN] = l_mean
                    df_s.iat[row_i, COL_L_SD]   = l_sd
                    df_s.iat[row_i, COL_L_MAX]  = l_max
                    df_s.iat[row_i, COL_L_MIN]  = l_min

                    df_s.iat[row_i, COL_R_MEAN] = r_mean
                    df_s.iat[row_i, COL_R_SD]   = r_sd
                    df_s.iat[row_i, COL_R_MAX]  = r_max
                    df_s.iat[row_i, COL_R_MIN]  = r_min

                    dfs[sname] = df_s  # store back

                print(f"{idx5} [{folder}] crisis={crisis_time:.3f}s via {crisis_src}; "
                      f"pre_no(L/R) n=({len(pre_no_L)}/{len(pre_no_R)}), "
                      f"pre_med(L/R) n=({len(pre_med_L)}/{len(pre_med_R)}), "
                      f"post_no(L/R) n=({len(post_no_L)}/{len(post_no_R)}), "
                      f"post_med(L/R) n=({len(post_med_L)}/{len(post_med_R)})")

        except Exception as e:
            print(f"{idx5} [{folder}]: ❌ Error reading CSV: {e}")
            continue

    # Write all sheets back
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            for name in sheet_names:
                dfs[name].to_excel(writer, sheet_name=name, index=False)
        print(f"\n✅ Saved results to '{OUTPUT_XLSX}'.")
    except Exception as e:
        print(f"❌ Failed to write workbook: {e}")

    # ---- Detailed diagnostics ----
    if missing_time_detail:
        print("\n--- Missing 'Time' column ---")
        for idx5, folder in missing_time_detail:
            print(f"{idx5} [{folder}]")

    if missing_pupil_detail:
        print("\n--- Missing pupil columns (per index) ---")
        for idx5, folder, cols in missing_pupil_detail:
            print(f"{idx5} [{folder}]: missing -> {', '.join(cols)}")

    if missing_crisis_detail:
        print("\n--- Missing crisis tags (per index) ---")
        for idx5, folder, searched in missing_crisis_detail:
            src = searched if searched else "no event column"
            print(f"{idx5} [{folder}]: no 'shook' / '0.2 seconds' found (searched in {src})")

    if used_event_fallback:
        print("\n--- Used 'Event' fallback for meditation toggles ---")
        for idx5, folder in used_event_fallback:
            print(f"{idx5} [{folder}]")

    if missing_both_events:
        print("\n--- Missing BOTH 'roomEvent' and 'Event' for meditation toggles (treated as no-meditation) ---")
        for idx5, folder in missing_both_events:
            print(f"{idx5} [{folder}]")

if __name__ == "__main__":
    main()