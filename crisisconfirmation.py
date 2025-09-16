#!/usr/bin/env python3
import os
import shutil
import numpy as np
import pandas as pd

# ====== Config ======
DATA_XLSX   = "Shooter 2 Data.xlsx"                  # first sheet
OUTPUT_XLSX = "Shooter 2 Data Proceed.xlsx"          # duplicate target
CONF_XLSX   = "ShooterStudy2CrisisStartConfirmation.xlsx"  # first sheet

ID_COL      = 0   # index column (0-based) in both workbooks
CRISIS_COL  = 3   # Shooter 2 Data crisis time column (0-based)

SEARCH_DIRS = ["noshookmodified", os.path.join("noshookmodified", "baseline")]  # preferred order

# ====== Helpers ======
def idx5(x):
    return str(x).strip()[:5]

def is_blank_cell(x):
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"

def nearest_time_index(times: np.ndarray, target: float) -> int:
    diffs = np.full_like(times, np.inf, dtype=float)
    mask  = np.isfinite(times)
    if not np.any(mask):
        return -1
    diffs[mask] = np.abs(times[mask] - target)
    return int(np.argmin(diffs))

def load_csv_lenient(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="python")

def append_event(df: pd.DataFrame, row_idx: int, evt_col_name: str, text: str):
    if evt_col_name not in df.columns:
        df[evt_col_name] = ""
    prev = "" if row_idx >= len(df) else str(df.at[row_idx, evt_col_name])
    prev = "" if prev == "nan" else prev
    df.at[row_idx, evt_col_name] = (prev + (" | " if prev else "") + text)

def find_csv_path_for_index(idx5: str):
    for folder in SEARCH_DIRS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name.startswith(idx5):
                return os.path.join(folder, name), folder
    return None, None

def first_column_times(df: pd.DataFrame) -> np.ndarray:
    if df.shape[1] == 0:
        return np.array([], dtype=float)
    return pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()

# ---------- robust parsing & diagnostics ----------
def _codepoint_summary(s: str) -> str:
    bads = [c for c in s if (not c.isprintable()) or (c in "\xa0\u200b\u200c\u200d\ufeff")]
    if not bads:
        return "(no special codepoints)"
    parts = [f"{repr(c)}(U+{ord(c):04X})" for c in bads]
    return " ".join(parts)

def parse_time_with_diagnostics(raw_val):
    """
    Try to parse a confirmation time into float seconds, with detailed logging.
    Returns (value: float or np.nan, log_lines: list[str]).
    """
    logs = []
    # direct numeric?
    if isinstance(raw_val, (int, float, np.floating)) and np.isfinite(raw_val):
        logs.append(f"  parse: numeric type -> {float(raw_val):.6f}")
        return float(raw_val), logs

    s0 = "" if raw_val is None else str(raw_val)
    s_repr = repr(s0)
    logs.append(f"  raw cell: {s_repr}")
    logs.append(f"  raw codepoints: {_codepoint_summary(s0)}")

    s = s0.replace("\xa0", " ").strip()  # replace NBSP, trim
    logs.append(f"  step1 (NBSP->space, strip): {repr(s)}")

    # empty?
    if s == "":
        logs.append("  FAIL: empty after strip")
        return np.nan, logs

    # try plain float
    try:
        v = float(s)
        logs.append(f"  parse: plain float OK -> {v:.6f}")
        return v, logs
    except Exception:
        logs.append("  parse: plain float failed")

    # remove commas, trailing units
    s2 = s.replace(",", "").lower()
    for suf in [" seconds", " second", " sec", " s"]:
        if s2.endswith(suf):
            s2 = s2[: -len(suf)].strip()
    logs.append(f"  step2 (rm commas/units): {repr(s2)}")
    try:
        v = float(s2)
        logs.append(f"  parse: float after cleanup OK -> {v:.6f}")
        return v, logs
    except Exception:
        logs.append("  parse: float after cleanup failed")

    # try mm:ss(.fff) or hh:mm:ss(.fff)
    if ":" in s2:
        parts = s2.split(":")
        try:
            if len(parts) == 2:
                m = float(parts[0]); sec = float(parts[1])
                v = m * 60.0 + sec
                logs.append(f"  parse: mm:ss(.fff) OK -> {v:.6f}")
                return v, logs
            elif len(parts) == 3:
                h = float(parts[0]); m = float(parts[1]); sec = float(parts[2])
                v = h * 3600.0 + m * 60.0 + sec
                logs.append(f"  parse: hh:mm:ss(.fff) OK -> {v:.6f}")
                return v, logs
            else:
                logs.append(f"  parse: colon format with {len(parts)} parts not supported")
        except Exception as e:
            logs.append(f"  parse: mm:ss/hh:mm:ss failed: {e}")

    logs.append("  RESULT: NaN (unparseable)")
    return np.nan, logs

# ====== Main ======
def main():
    # sanity checks
    if not os.path.isfile(CONF_XLSX):
        print(f"❌ Missing confirmation file: {CONF_XLSX}"); return
    if not os.path.isfile(DATA_XLSX):
        print(f"❌ Missing data file: {DATA_XLSX}"); return

    # load workbooks
    conf = pd.read_excel(CONF_XLSX, sheet_name=0, engine="openpyxl")
    data_src = pd.read_excel(DATA_XLSX,  sheet_name=0, engine="openpyxl")

    conf_idx = conf.iloc[:, ID_COL].apply(idx5)
    conf_raw = conf.iloc[:, 1]  # 2nd column in confirmation

    # build map: index -> (first non-NaN candidate, plus raw row logs)
    conf_map = {}
    dup_rows = {}
    for row_i, (ix, raw_val) in enumerate(zip(conf_idx, conf_raw), start=2):  # start=2 to account header=1
        val, logs = parse_time_with_diagnostics(raw_val)
        if ix not in conf_map:
            conf_map[ix] = {"val": val, "raws": [(row_i, raw_val, logs)]}
        else:
            conf_map[ix]["raws"].append((row_i, raw_val, logs))
            dup_rows[ix] = dup_rows.get(ix, 0) + 1
            # prefer the first non-NaN if current stored is NaN
            if (np.isnan(conf_map[ix]["val"]) or conf_map[ix]["val"] is None) and np.isfinite(val):
                conf_map[ix]["val"] = val

    # duplicate workbook
    try:
        shutil.copyfile(DATA_XLSX, OUTPUT_XLSX)
    except Exception as e:
        print(f"❌ Failed to copy '{DATA_XLSX}' → '{OUTPUT_XLSX}': {e}")
        return
    data_out = pd.read_excel(OUTPUT_XLSX, sheet_name=0, engine="openpyxl")

    # iterate indices needing fill
    total_rows = len(data_src)
    need_fill = [i for i in range(total_rows) if is_blank_cell(data_src.iat[i, CRISIS_COL])]
    filled = 0
    csv_found = csv_tagged = 0
    skipped_not_in_conf = skipped_unparseable = 0
    csv_not_found = csv_write_fail = 0

    print("Index | conf_time(s) | chosen_row_time(s) | CSV_written | Notes")
    print("----------------------------------------------------------------")
    for i in need_fill:
        ix = idx5(data_src.iat[i, ID_COL])
        if ix not in conf_map:
            print(f"{ix:5s} | {'-':>12s} | {'-':>18s} |    NO       | not in confirmation")
            skipped_not_in_conf += 1
            continue

        cinfo = conf_map[ix]
        t_conf = cinfo["val"]
        # If unparseable, print full diagnostics for that index
        if not np.isfinite(t_conf):
            print(f"{ix:5s} | {'-':>12s} | {'-':>18s} |    NO       | confirmation time NaN — details:")
            for (rowno, raw, logs) in cinfo["raws"]:
                print(f"   - CONF row {rowno}, raw={repr(str(raw))}")
                for line in logs:
                    print("     " + line)
            skipped_unparseable += 1
            continue

        # write to duplicate excel
        data_out.iat[i, CRISIS_COL] = float(t_conf)
        filled += 1

        # find csv
        csv_path, folder = find_csv_path_for_index(ix)
        if not csv_path:
            print(f"{ix:5s} | {t_conf:12.6f} | {'-':>18s} |    NO       | CSV not found in {SEARCH_DIRS}")
            csv_not_found += 1
            continue
        csv_found += 1

        # load csv and tag nearest row
        try:
            dfr = load_csv_lenient(csv_path)
        except Exception as e:
            print(f"{ix:5s} | {t_conf:12.6f} | {'-':>18s} |    NO       | CSV read error: {e}")
            csv_not_found += 1
            continue
        if dfr.shape[0] == 0:
            print(f"{ix:5s} | {t_conf:12.6f} | {'-':>18s} |    NO       | CSV empty")
            continue

        times = pd.to_numeric(dfr.iloc[:, 0], errors="coerce").to_numpy()
        if len(times) == 0 or not np.any(np.isfinite(times)):
            print(f"{ix:5s} | {t_conf:12.6f} | {'-':>18s} |    NO       | CSV has no valid time values")
            continue

        j = nearest_time_index(times, float(t_conf))
        if j < 0 or not np.isfinite(times[j]):
            print(f"{ix:5s} | {t_conf:12.6f} | {'-':>18s} |    NO       | No nearest time row")
            continue

        try:
            # Ensure robotEvent exists and append estimated shook
            evt_col = "robotEvent"
            if evt_col not in dfr.columns:
                dfr[evt_col] = ""
            append_event(dfr, j, evt_col, "estimated shook")
            dfr.to_csv(csv_path, index=False)
            csv_tagged += 1
            print(f"{ix:5s} | {t_conf:12.6f} | {times[j]:18.6f} |    YES      | wrote 'estimated shook' @ row {j}")
        except Exception as e:
            csv_write_fail += 1
            print(f"{ix:5s} | {t_conf:12.6f} | {times[j]:18.6f} |    NO       | CSV write error: {e}")

    # save duplicate
    try:
        data_out.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to write '{OUTPUT_XLSX}': {e}")
        return

    # Summary
    print("\n===== SUMMARY =====")
    print(f"Rows needing fill                    : {len(need_fill)}")
    print(f"Filled crisis times in Proceed       : {filled}")
    print(f"Skipped (index not in confirmation)  : {skipped_not_in_conf}")
    print(f"Skipped (unparseable conf time)      : {skipped_unparseable}")
    print(f"CSV found                            : {csv_found}")
    print(f"CSV tagged 'estimated shook'         : {csv_tagged}")
    print(f"CSV not found / read error           : {csv_not_found}")
    print(f"CSV write failures                   : {csv_write_fail}")

    if dup_rows:
        total_dups = sum(dup_rows.values())
        print(f"\nNote: duplicates in confirmation for {len(dup_rows)} indices "
              f"({total_dups} extra occurrence(s)). First non-NaN was used.")

if __name__ == "__main__":
    main()