#!/usr/bin/env python3
import os
import re
import shutil
import unicodedata
import numpy as np
import pandas as pd

# ========= Config =========
SOURCE_XLSX = "Shooter 2 Data.xlsx"
OUTPUT_XLSX = "Shooter 2 TMP.xlsx"

# CSV search folders (in order)
SEARCH_FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshookmodified",
    os.path.join("noshookmodified", "baseline"),
]

# Sheets: 0=overview (indices + crisis time), 1=pre_no, 2=pre_med, 3=post_no, 4=post_med
SHEET_PRE_NO   = 1
SHEET_PRE_MED  = 2
SHEET_POST_NO  = 3
SHEET_POST_MED = 4

# Crisis time column on sheet 1 (0-based; Excel col 4)
CRISIS_COL_0 = 3

# Write targets (1-based Excel) -> we’ll convert to 0-based when writing
# F,G,H,I = Mean, SD(sample), Max, Min
OUT_COLS_1B = [6, 7, 8, 9]

# ========= Helpers =========
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", normalize_text(s).lower())

def find_cols(header, want_keys):
    """Relaxed (case-insensitive) substring match over normalized header tokens."""
    tokens = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        idx = -1
        for j, t in enumerate(tokens):
            if k in t:
                idx = j
                break
        out[key] = idx
    return out

def ensure_width(df: pd.DataFrame, upto_1based: int) -> pd.DataFrame:
    upto_0 = upto_1based - 1
    while df.shape[1] <= upto_0:
        df[f"Extra_{df.shape[1]+1}"] = np.nan
    return df

def index5(v) -> str:
    """First 5 chars; if purely numeric <5 digits, zero-pad to width 5."""
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

def find_matching_csv(idx5: str):
    """Return (csv_path, folder_label) for the first match, else (None, None)."""
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

def compute_stats_if_enough(values: list[float]):
    """Return (mean, sd(ddof=1), max, min) if n>=2 else all NaN."""
    n = len(values)
    if n < 2:
        return (np.nan, np.nan, np.nan, np.nan)
    arr = np.asarray(values, dtype=float)
    mean_v = float(np.mean(arr))
    sd_v   = float(np.std(arr, ddof=1))
    max_v  = float(np.max(arr))
    min_v  = float(np.min(arr))
    return (mean_v, sd_v, max_v, min_v)

# -------- Survey-room exclusion and meditation toggles --------
def build_survey_and_meditation_states(df: pd.DataFrame, room_idx: int, event_idx: int, n_rows: int):
    """
    Build two per-row booleans:
      - in_survey: exclude rows inside Survey Room
      - in_med:    true while in Meditation Area
    roomEvent preferred; fallback Event with substring matching.
    """
    in_survey = np.zeros(n_rows, dtype=bool)
    in_med    = np.zeros(n_rows, dtype=bool)

    if room_idx != -1:
        # Exact markers for survey; exact column used
        ser = df.iloc[:, room_idx].astype(str)
        state_s = False
        state_m = False
        for i in range(n_rows):
            cell = ser.iat[i]
            if cell == "Entered Survey Room":
                state_s = True
            elif cell == "Exited Survey Room":
                state_s = False
            # Meditation: use substring even in roomEvent to be robust
            low = normalize_text(cell).lower()
            if "entered meditation area" in low:
                state_m = True
            elif "exited meditation area" in low:
                state_m = False
            in_survey[i] = state_s
            in_med[i] = state_m
    elif event_idx != -1:
        ser = df.iloc[:, event_idx].astype(str).map(lambda s: normalize_text(s).lower())
        state_s = False
        state_m = False
        for i in range(n_rows):
            s = ser.iat[i]
            if "entered survey room" in s:
                state_s = True
            elif "exited survey room" in s:
                state_s = False
            if "entered meditation area" in s:
                state_m = True
            elif "exited meditation area" in s:
                state_m = False
            in_survey[i] = state_s
            in_med[i] = state_m

    return in_survey, in_med

# -------- Velocity builder --------
def build_player_velocities(times: np.ndarray,
                            pxyz: np.ndarray,
                            in_survey: np.ndarray) -> tuple[list[float], list[int]]:
    """
    Compute per-step player speeds between consecutive usable rows (i-1 -> i),
    skipping inside-survey rows, non-finite coords, any -1 sentinel, or dt<=0.
    Returns (speeds, end_row_indices) so each speed is associated with row i.
    """
    n = len(times)
    speeds = []
    end_rows = []
    prev = None
    for i in range(n):
        if in_survey[i]:
            prev = i
            continue
        t1 = times[i]
        if not np.isfinite(t1):
            prev = i
            continue
        p1 = pxyz[i]
        if (not np.isfinite(p1).all()) or (p1 == -1).any():
            prev = i
            continue

        if prev is None:
            prev = i
            continue
        if i - prev != 1:
            prev = i
            continue

        # now we have a consecutive pair (prev -> i)
        if in_survey[prev]:
            prev = i
            continue
        t0 = times[prev]
        p0 = pxyz[prev]
        if (not np.isfinite(t0)) or (not np.isfinite(p0).all()) or (p0 == -1).any():
            prev = i
            continue
        dt = t1 - t0
        if not np.isfinite(dt) or dt <= 0:
            prev = i
            continue

        speed = float(np.linalg.norm(p1 - p0) / dt)
        speeds.append(speed)
        end_rows.append(i)

        prev = i
    return speeds, end_rows

# ========= Main =========
def main():
    # Duplicate workbook
    try:
        shutil.copyfile(SOURCE_XLSX, OUTPUT_XLSX)
    except Exception as e:
        print(f"❌ Failed to copy '{SOURCE_XLSX}' → '{OUTPUT_XLSX}': {e}")
        return

    # Load sheets
    try:
        xls = pd.ExcelFile(OUTPUT_XLSX, engine="openpyxl")
        sheet_names = xls.sheet_names
        if len(sheet_names) < 5:
            print(f"❌ Expected ≥5 sheets (sheet1 + 4 category sheets). Found {len(sheet_names)}.")
            return
        df_over = xls.parse(sheet_names[0])  # indices + crisis time
        df_pre_no   = xls.parse(sheet_names[SHEET_PRE_NO])
        df_pre_med  = xls.parse(sheet_names[SHEET_PRE_MED])
        df_post_no  = xls.parse(sheet_names[SHEET_POST_NO])
        df_post_med = xls.parse(sheet_names[SHEET_POST_MED])
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_XLSX}': {e}")
        return

    # Ensure destination widths for sheets 2–5 up to column I (1-based 9)
    for df_s in [df_pre_no, df_pre_med, df_post_no, df_post_med]:
        df_s = ensure_width(df_s, max(OUT_COLS_1B))
    df_sheets = [df_pre_no, df_pre_med, df_post_no, df_post_med]  # keep refs

    # Optional: set headers on F–I
    headers = ["Mean Velocity", "SD Velocity", "Max Velocity", "Min Velocity"]
    for df_s in df_sheets:
        for c_1b, name in zip(OUT_COLS_1B, headers):
            col0 = c_1b - 1
            if col0 < len(df_s.columns):
                df_s.columns.values[col0] = name

    n_rows = len(df_over)
    for i in range(n_rows):
        idx5 = index5(df_over.iat[i, 0])

        # Crisis time from sheet 1 col 4
        try:
            crisis_time = float(df_over.iat[i, CRISIS_COL_0])
        except Exception:
            crisis_time = np.nan
        if not np.isfinite(crisis_time):
            # write NaNs for all four buckets on this row
            for df_s in df_sheets:
                for c_1b in OUT_COLS_1B:
                    df_s.iat[i, c_1b - 1] = np.nan
            print(f"{idx5}: SKIP (crisis time NaN on sheet 1)")
            continue

        csv_path, folder = find_matching_csv(idx5)
        if not csv_path:
            for df_s in df_sheets:
                for c_1b in OUT_COLS_1B:
                    df_s.iat[i, c_1b - 1] = np.nan
            print(f"{idx5}: SKIP (no CSV found)")
            continue

        # Load CSV
        try:
            dfr = load_csv_lenient(csv_path)
        except Exception as e:
            for df_s in df_sheets:
                for c_1b in OUT_COLS_1B:
                    df_s.iat[i, c_1b - 1] = np.nan
            print(f"{idx5}: SKIP CSV read error: {e} [{folder}]")
            continue
        if dfr.shape[0] == 0:
            for df_s in df_sheets:
                for c_1b in OUT_COLS_1B:
                    df_s.iat[i, c_1b - 1] = np.nan
            print(f"{idx5}: SKIP empty CSV [{folder}]")
            continue

        header = list(dfr.columns)
        colmap = find_cols(header, ["time", "roomevent", "event", "playervr.x", "playervr.y", "playervr.z"])
        # time + player positions required
        need_missing = []
        for key in ["time", "playervr.x", "playervr.y", "playervr.z"]:
            if colmap[key] == -1:
                need_missing.append(key)
        if need_missing:
            for df_s in df_sheets:
                for c_1b in OUT_COLS_1B:
                    df_s.iat[i, c_1b - 1] = np.nan
            print(f"{idx5}: SKIP missing columns: {', '.join(need_missing)} [{folder}]")
            continue

        time_col = header[colmap["time"]]
        px_col   = header[colmap["playervr.x"]]
        py_col   = header[colmap["playervr.y"]]
        pz_col   = header[colmap["playervr.z"]]

        times = coerce_float(dfr[time_col])
        pxyz  = dfr[[px_col, py_col, pz_col]].astype(float).to_numpy()

        n = len(times)
        # survey + meditation states
        in_survey, in_med = build_survey_and_meditation_states(
            dfr,
            room_idx=colmap["roomevent"],
            event_idx=colmap["event"],
            n_rows=n
        )

        # Build velocities and end-row mapping
        speeds, end_rows = build_player_velocities(times, pxyz, in_survey)

        # Split speeds into four buckets by the row they end on
        buckets = {"pre_no": [], "pre_med": [], "post_no": [], "post_med": []}
        for s, r_end in zip(speeds, end_rows):
            ti = times[r_end]
            if not np.isfinite(ti):
                continue
            pre = (ti < crisis_time)
            med = bool(in_med[r_end])
            if pre and not med:   buckets["pre_no"].append(s)
            elif pre and med:     buckets["pre_med"].append(s)
            elif (not pre) and not med: buckets["post_no"].append(s)
            else:                  buckets["post_med"].append(s)

        # Compute stats per bucket and write to sheets 2..5 (F–I)
        order = ["pre_no", "pre_med", "post_no", "post_med"]
        targets = [SHEET_PRE_NO, SHEET_PRE_MED, SHEET_POST_NO, SHEET_POST_MED]
        for cat, sheet_idx in zip(order, targets):
            stats = compute_stats_if_enough(buckets[cat])
            df_target = df_sheets[sheet_idx - 1]  # adjust because list starts at sheet 2
            for c_1b, val in zip(OUT_COLS_1B, stats):
                df_target.iat[i, c_1b - 1] = val

        print(f"{idx5}: OK — pre_no n={len(buckets['pre_no'])}, pre_med n={len(buckets['pre_med'])}, "
              f"post_no n={len(buckets['post_no'])}, post_med n={len(buckets['post_med'])}")

    # Save back all category sheets (sheet 1 unchanged)
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            df_over.to_excel(writer, sheet_name=sheet_names[0], index=False)
            df_sheets[0].to_excel(writer, sheet_name=sheet_names[SHEET_PRE_NO], index=False)
            df_sheets[1].to_excel(writer, sheet_name=sheet_names[SHEET_PRE_MED], index=False)
            df_sheets[2].to_excel(writer, sheet_name=sheet_names[SHEET_POST_NO], index=False)
            df_sheets[3].to_excel(writer, sheet_name=sheet_names[SHEET_POST_MED], index=False)
        print(f"\n✅ Saved results to '{OUTPUT_XLSX}'.")
    except Exception as e:
        print(f"❌ Failed to write '{OUTPUT_XLSX}': {e}")

if __name__ == "__main__":
    main()