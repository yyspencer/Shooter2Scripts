#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

CONF_XLSX = "ShooterStudy2CrisisStartConfirmation.xlsx"  # first sheet
DATA_XLSX = "Shooter 2 Data.xlsx"                        # first sheet

CONF_IDX_COL = 0  # first column (0-based) -> 5-char index
CONF_T_COL   = 1  # second column -> crisis time (seconds)

DATA_IDX_COL = 0  # first column (0-based)
DATA_T_COL   = 3  # 4th column -> crisis time (seconds)

def idx5(x):
    return str(x).strip()[:5]

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def fmt(x):
    return "-" if pd.isna(x) else f"{float(x):.6f}"

def main():
    # ---- Load both workbooks (first sheets) ----
    if not os.path.isfile(CONF_XLSX):
        print(f"❌ Missing file: {CONF_XLSX}")
        return
    if not os.path.isfile(DATA_XLSX):
        print(f"❌ Missing file: {DATA_XLSX}")
        return

    conf = pd.read_excel(CONF_XLSX, sheet_name=0, engine="openpyxl")
    data = pd.read_excel(DATA_XLSX,  sheet_name=0, engine="openpyxl")

    # ---- Normalize indices and times ----
    conf_idx = conf.iloc[:, CONF_IDX_COL].apply(idx5)
    conf_t   = to_num(conf.iloc[:, CONF_T_COL])

    data_idx = data.iloc[:, DATA_IDX_COL].apply(idx5)
    data_t   = to_num(data.iloc[:, DATA_T_COL])

    # ---- Build index -> crisis_time mapping from Shooter 2 Data ----
    data_map = {}
    dup_counts = {}
    for ix, t in zip(data_idx, data_t):
        if ix in data_map:
            dup_counts[ix] = dup_counts.get(ix, 1) + 1
            # keep the first non-NaN time if the stored one was NaN
            if pd.isna(data_map[ix]) and pd.notna(t):
                data_map[ix] = float(t)
        else:
            data_map[ix] = float(t) if pd.notna(t) else np.nan

    # ---- Compare per index from Confirmation file ----
    diffs = []
    n_processed = 0
    skipped_not_found = 0
    skipped_missing_data_time = 0
    skipped_missing_conf_time = 0

    print("Index |      t_conf(s)      t_data(s)    diff=t_data-t_conf    Note")
    print("--------------------------------------------------------------------------")
    for ix, t_conf in zip(conf_idx, conf_t):
        note = ""
        # find matching data time
        if ix not in data_map:
            # print line anyway, but mark as skipped
            print(f"{ix:5s} | {fmt(t_conf):>14s}  {'-':>14s}   {'-':>18s}    SKIP: not in Shooter 2 Data")
            skipped_not_found += 1
            continue

        t_data = data_map[ix]
        if pd.isna(t_data):
            print(f"{ix:5s} | {fmt(t_conf):>14s}  {'-':>14s}   {'-':>18s}    SKIP: DATA crisis time empty")
            skipped_missing_data_time += 1
            continue

        if pd.isna(t_conf):
            print(f"{ix:5s} | {'-':>14s}  {fmt(t_data):>14s}   {'-':>18s}    SKIP: CONF crisis time empty")
            skipped_missing_conf_time += 1
            continue

        # valid comparison
        t_conf_f = float(t_conf)
        t_data_f = float(t_data)
        diff = t_data_f - t_conf_f
        diffs.append(diff)
        n_processed += 1
        print(f"{ix:5s} | {t_conf_f:14.6f}  {t_data_f:14.6f}   {diff:+18.6f}    ")

    # ---- Summary ----
    print("\n===== SUMMARY (signed differences, seconds) =====")
    print(f"Compared indices             : {n_processed}")
    print(f"Printed (all rows)           : {len(conf_idx)}")
    print(f"Skipped from stats (not found): {skipped_not_found}")
    print(f"Skipped from stats (DATA empty): {skipped_missing_data_time}")
    print(f"Skipped from stats (CONF empty): {skipped_missing_conf_time}")

    if n_processed > 0:
        arr = np.array(diffs, dtype=float)
        mean   = float(np.mean(arr))
        median = float(np.median(arr))
        std    = float(np.std(arr, ddof=1)) if n_processed > 1 else float('nan')
        vmin   = float(np.min(arr))
        vmax   = float(np.max(arr))
        print(f"\nSigned diff stats (DATA − CONF):")
        print(f"  mean   : {mean:+.6f}")
        print(f"  median : {median:+.6f}")
        print(f"  std    : {std:.6f}")
        print(f"  min    : {vmin:+.6f}")
        print(f"  max    : {vmax:+.6f}")
    else:
        print("No overlapping indices with both crisis times present; stats omitted.")

    # Duplicates note
    if dup_counts:
        total_dups = sum(dup_counts.values()) - len(dup_counts)
        print(f"\nNote: duplicate indices in Shooter 2 Data detected for {len(dup_counts)} key(s); "
              f"{total_dups} extra occurrence(s) in total. First non-NaN time was used per index.")

if __name__ == "__main__":
    main()