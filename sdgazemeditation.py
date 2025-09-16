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

# Write target column (1-based Excel) -> J
OUT_COL_1B = 10

# ========= Helpers =========
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", normalize_text(s).lower())

def find_cols(header, want_keys):
    """Return dict {key -> FIRST match col index or -1} via relaxed matching."""
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
    """First 5 chars; zero-pad if purely numeric and <5 digits."""
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
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

# ---- Meditation toggles from roomEvent (preferred) or Event ----
def build_in_meditation(df: pd.DataFrame, room_idx: int, event_idx: int, n_rows: int):
    """
    Tags (case-insensitive substring):
      'entered meditation area' => enter (True), 'exited meditation area' => exit (False)
    Returns per-row bool array.
    """
    used_idx = -1
    if room_idx != -1:
        used_idx = room_idx
        ser = df.iloc[:, used_idx].astype(str)
        # exact cell values possible; normalize lower for meditation substrings
        ser_low = ser.map(lambda s: normalize_text(s).lower())
        enter_rows = set(np.flatnonzero(ser_low.str.contains("entered meditation area", regex=False, na=False)).tolist())
        exit_rows  = set(np.flatnonzero(ser_low.str.contains("exited meditation area",  regex=False, na=False)).tolist())
    elif event_idx != -1:
        used_idx = event_idx
        ser_low = df.iloc[:, used_idx].astype(str).map(lambda s: normalize_text(s).lower())
        enter_rows = set(np.flatnonzero(ser_low.str.contains("entered meditation area", regex=False, na=False)).tolist())
        exit_rows  = set(np.flatnonzero(ser_low.str.contains("exited meditation area",  regex=False, na=False)).tolist())
    else:
        return np.zeros(n_rows, dtype=bool)

    in_med = np.zeros(n_rows, dtype=bool)
    state = False
    for i in range(n_rows):
        if i in enter_rows:
            state = True
        if i in exit_rows:
            state = False
        in_med[i] = state
    return in_med

# ---- Gaze column finder (robust to slight name variants) ----
def find_gaze_cols(header):
    """
    Returns (gx, gy, gz) column indices or (-1,-1,-1).
    Prefers exact 'Gaze Visualizer.x/y/z'; falls back to relaxed tokens having 'gaze' and ending in x|y|z.
    """
    # exact first
    exact_map = {h.strip(): i for i, h in enumerate(header) if isinstance(h, str)}
    gx = exact_map.get("Gaze Visualizer.x", -1)
    gy = exact_map.get("Gaze Visualizer.y", -1)
    gz = exact_map.get("Gaze Visualizer.z", -1)
    if gx != -1 and gy != -1 and gz != -1:
        return gx, gy, gz

    # relaxed
    toks = [norm_header_token(h) for h in header]
    def find_tok(pred):
        for i, tk in enumerate(toks):
            if pred(tk):
                return i
        return -1
    gx = find_tok(lambda tk: ("gaze" in tk and tk.endswith("x")))
    gy = find_tok(lambda tk: ("gaze" in tk and tk.endswith("y")))
    gz = find_tok(lambda tk: ("gaze" in tk and tk.endswith("z")))
    return gx, gy, gz

def sd3_string(xs: list[float], ys: list[float], zs: list[float]) -> str:
    """Sample SD (ddof=1) of each axis if n>=2; else NaN string."""
    n = min(len(xs), len(ys), len(zs))
    if n < 2:
        return "NaN"
    def sd(a):
        if len(a) < 2: return float('nan')
        arr = np.asarray(a, dtype=float)
        return float(np.std(arr, ddof=1))
    sdx, sdy, sdz = sd(xs), sd(ys), sd(zs)
    # if all NaN (e.g., only 1 usable), return "NaN"
    if not (np.isfinite(sdx) or np.isfinite(sdy) or np.isfinite(sdz)):
        return "NaN"
    # format like earlier scripts
    def fmt(v): return ("NaN" if not np.isfinite(v) else f"{v:.6f}")
    return f"[{fmt(sdx)}, {fmt(sdy)}, {fmt(sdz)}]"

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

    # Ensure destination width up to column J on sheets 2–5
    for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
        ensure_width(df_s, OUT_COL_1B)

    # Optional header on J
    for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
        df_s.columns.values[OUT_COL_1B-1] = "SD Gaze [x,y,z]"

    n_rows = len(df_over)
    for i in range(n_rows):
        idx5 = index5(df_over.iat[i, 0])

        # Crisis time from sheet 1, col 4
        try:
            crisis_time = float(df_over.iat[i, CRISIS_COL_0])
        except Exception:
            crisis_time = np.nan
        if not np.isfinite(crisis_time):
            # write NaN to J for all 4 sheets
            for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
                df_s.iat[i, OUT_COL_1B-1] = np.nan
            print(f"{idx5}: SKIP (crisis time NaN on sheet 1)")
            continue

        # Find CSV
        csv_path, folder = find_matching_csv(idx5)
        if not csv_path:
            for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
                df_s.iat[i, OUT_COL_1B-1] = np.nan
            print(f"{idx5}: SKIP (no CSV found)")
            continue

        # Load CSV
        try:
            dfr = load_csv_lenient(csv_path)
        except Exception as e:
            for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
                df_s.iat[i, OUT_COL_1B-1] = np.nan
            print(f"{idx5}: SKIP CSV read error: {e} [{folder}]")
            continue
        if dfr.shape[0] == 0:
            for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
                df_s.iat[i, OUT_COL_1B-1] = np.nan
            print(f"{idx5}: SKIP empty CSV [{folder}]")
            continue

        header = list(dfr.columns)
        colmap = find_cols(header, ["time", "roomevent", "event"])
        # Find gaze cols
        gx, gy, gz = find_gaze_cols(header)

        # time + gaze required
        if colmap["time"] == -1 or gx == -1 or gy == -1 or gz == -1:
            for df_s in (df_pre_no, df_pre_med, df_post_no, df_post_med):
                df_s.iat[i, OUT_COL_1B-1] = np.nan
            missing = []
            if colmap["time"] == -1: missing.append("Time")
            if gx == -1 or gy == -1 or gz == -1: missing.append("Gaze Visualizer.{x,y,z}")
            print(f"{idx5}: SKIP missing columns: {', '.join(missing)} [{folder}]")
            continue

        times = coerce_float(dfr.iloc[:, colmap["time"]])

        # Build meditation state (no survey exclusion here unless you want it)
        in_med = build_in_meditation(dfr, colmap["roomevent"], colmap["event"], len(dfr))

        # Collect gaze per category
        gaze = dfr.iloc[:, [gx, gy, gz]].astype(str).to_numpy()
        buckets = {
            "pre_no": ([], [], []),
            "pre_med": ([], [], []),
            "post_no": ([], [], []),
            "post_med": ([], [], []),
        }

        for r in range(len(dfr)):
            t = times[r]
            if not np.isfinite(t):
                continue
            # parse gaze row; reject non-finite / -1
            try:
                vx = float(gaze[r, 0]); vy = float(gaze[r, 1]); vz = float(gaze[r, 2])
            except Exception:
                continue
            if any((not np.isfinite(v)) for v in (vx, vy, vz)):
                continue
            if vx == -1 or vy == -1 or vz == -1:
                continue

            pre = (t < crisis_time)
            med = bool(in_med[r])
            key = ("pre_no" if pre and not med else
                   "pre_med" if pre and med else
                   "post_no" if (not pre) and not med else
                   "post_med")
            xs, ys, zs = buckets[key]
            xs.append(vx); ys.append(vy); zs.append(vz)

        # Compute SD triple per category and write to column J
        order = ["pre_no", "pre_med", "post_no", "post_med"]
        sheets = [df_pre_no, df_pre_med, df_post_no, df_post_med]
        for key, df_target in zip(order, sheets):
            xs, ys, zs = buckets[key]
            sd_str = sd3_string(xs, ys, zs)
            df_target.iat[i, OUT_COL_1B-1] = sd_str

        print(f"{idx5}: OK — wrote SD gaze (x,y,z) to J for all 4 categories")

    # Save back category sheets (sheet 1 unchanged)
    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl", mode="w") as writer:
            # keep original sheet order/names
            df_over.to_excel(writer, sheet_name=sheet_names[0], index=False)
            df_pre_no.to_excel(writer,   sheet_name=sheet_names[SHEET_PRE_NO],   index=False)
            df_pre_med.to_excel(writer,  sheet_name=sheet_names[SHEET_PRE_MED],  index=False)
            df_post_no.to_excel(writer,  sheet_name=sheet_names[SHEET_POST_NO],  index=False)
            df_post_med.to_excel(writer, sheet_name=sheet_names[SHEET_POST_MED], index=False)
        print(f"\n✅ Saved SD gaze per category to '{OUTPUT_XLSX}' (column J in sheets 2–5).")
    except Exception as e:
        print(f"❌ Failed to write '{OUTPUT_XLSX}': {e}")

if __name__ == "__main__":
    main()