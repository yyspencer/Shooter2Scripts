#!/usr/bin/env python3
import os
import re
import shutil
import numpy as np
import pandas as pd

# ---------------- Config (Shooter Study 2) ----------------
INPUT_XLSX  = "Shooter 2 Data.xlsx"
OUTPUT_XLSX = "Shooter 2 Data Proceed.xlsx"
ID_COL = 0  # first column in Excel (0-based)

# Search directory (non-recursive)
SEARCH_DIR = "../Data/Sh Study 2 Data"

# Write destinations (0-based, inclusive ranges) — 11 columns each
# (45,55) -> Excel columns 46..56, etc.
RANGES = {
    "evolab5e+tablet":       (44, 54),  # Excel cols 46..56
    "intermediate+tablet":   (55, 65),  # Excel cols 57..67
    "standardoffice+tablet": (66, 76),  # Excel cols 68..78
}

# Category tokens (case-insensitive; ignore spaces/_/- in filenames)
KEYS = {
    "evolab5e+tablet":       ("evolab5e", "tablet"),
    "intermediate+tablet":   ("intermediate", "tablet"),
    "standardoffice+tablet": ("standard School", "tablet"),
}

# -------------- Helpers -----------------
def normalize(s: str) -> str:
    """Lowercase and remove spaces/underscores/hyphens to simplify substring matches."""
    return re.sub(r"[ _-]+", "", s.lower())

def first5_from_cell(val) -> str:
    """Use the first 5 characters of the first-column cell (no zero-padding)."""
    s = str(val).strip()
    return s[:5]

def ensure_columns(df: pd.DataFrame, upto_col_inclusive: int) -> pd.DataFrame:
    """Ensure DataFrame has columns up to given 0-based index inclusive; fill with NaN."""
    while df.shape[1] <= upto_col_inclusive:
        df[f"Extra_{df.shape[1]+1}"] = np.nan
    return df

def read_single_row_11cols(csv_path: str):
    """
    Robustly read the single data row (11 columns) from csv_path.
    - Works whether there is a header or not.
    - Returns a list of 11 values (floats where possible), or None on failure.
    """
    try:
        raw = pd.read_csv(csv_path, header=None, dtype=str, encoding="utf-8", engine="python")
        raw = raw.dropna(how="all")
        if raw.empty:
            return None
        # Use the last non-empty row as the data row
        row = raw.iloc[-1].tolist()
        # Force exactly 11 columns
        row = row[:11] + [np.nan] * max(0, 11 - len(row))
        out = []
        for v in row:
            if pd.isna(v):
                out.append(np.nan); continue
            sv = str(v).strip()
            if sv == "":
                out.append(np.nan); continue
            try:
                out.append(float(sv))
            except Exception:
                out.append(sv)
        return out
    except Exception:
        return None

def find_category_csv_in_dir(idx5: str, cat_key: str):
    """
    Search ONLY SEARCH_DIR (non-recursive) for CSVs that:
      - filename starts with idx5
      - normalized name contains both tokens for cat_key
    Returns (status, path_or_list):
      - ("FOUND", path)
      - ("NOT_FOUND", None)
      - ("DUPLICATES", [paths])
    """
    want_a, want_b = KEYS[cat_key]
    a_norm, b_norm = normalize(want_a), normalize(want_b)

    try:
        names = os.listdir(SEARCH_DIR)
    except Exception:
        # If directory missing/inaccessible, behave as not found
        return ("NOT_FOUND", None)

    matches = []
    for name in names:
        if not name.lower().endswith(".csv"):
            continue
        if not name.startswith(idx5):
            continue
        norm = normalize(name)
        if a_norm in norm and b_norm in norm:
            matches.append(os.path.join(SEARCH_DIR, name))

    if len(matches) == 1:
        return ("FOUND", matches[0])
    elif len(matches) == 0:
        return ("NOT_FOUND", None)
    else:
        return ("DUPLICATES", matches)

# --------------- Main -------------------
def main():
    # Duplicate Excel
    try:
        shutil.copyfile(INPUT_XLSX, OUTPUT_XLSX)
    except Exception as e:
        print(f"❌ Failed to copy '{INPUT_XLSX}' → '{OUTPUT_XLSX}': {e}")
        return

    try:
        df = pd.read_excel(OUTPUT_XLSX, sheet_name=0, engine="openpyxl")
    except Exception as e:
        print(f"❌ Failed to read '{OUTPUT_XLSX}': {e}")
        return

    # Ensure columns exist up to the highest needed index
    max_col_needed = max(end for (_, end) in RANGES.values())
    df = ensure_columns(df, max_col_needed)

    # Pre-cast all destination blocks to 'object' once (avoids dtype warnings)
    all_block_cols = sorted({c for (start, end) in RANGES.values() for c in range(start, end + 1)})
    df[df.columns[all_block_cols]] = df[df.columns[all_block_cols]].astype("object")

    # Process each index row
    for i in range(len(df)):
        idx5 = first5_from_cell(df.iloc[i, ID_COL])

        statuses = []
        for cat_key, (start, end) in RANGES.items():
            status, payload = find_category_csv_in_dir(idx5, cat_key)

            if status == "FOUND":
                row_vals = read_single_row_11cols(payload)
                if row_vals is None:
                    df.iloc[i, start:end+1] = [np.nan] * 11
                    statuses.append(f"{cat_key}=PARSE_FAIL")
                else:
                    df.iloc[i, start:end+1] = row_vals
                    statuses.append(f"{cat_key}=FOUND")
            elif status == "NOT_FOUND":
                df.iloc[i, start:end+1] = [np.nan] * 11
                statuses.append(f"{cat_key}=NOT_FOUND")
            else:  # DUPLICATES
                df.iloc[i, start:end+1] = [np.nan] * 11
                statuses.append(f"{cat_key}=DUPLICATES({len(payload)})")

        print(f"{idx5}: " + ", ".join(statuses))

    # Save
    try:
        df.to_excel(OUTPUT_XLSX, index=False, engine="openpyxl")
        print(f"\n✅ Done! Results saved to '{OUTPUT_XLSX}'")
    except Exception as e:
        print(f"❌ Failed to write '{OUTPUT_XLSX}': {e}")

if __name__ == "__main__":
    main()