import os
import re
import unicodedata
import pandas as pd
import numpy as np

# ----------------- Config -----------------
EXCEL_FILE = "Shooter 2 Data.xlsx"
ID_COL = 0  # first column holds indices
SEARCH_FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
    "noshook",
    os.path.join("noshook", "baseline"),
]

# Target tags (display names)
TAGS = [
    "Robot entered Survey Room",
    "Entered Survey Room",
    "Robot exited Survey Room",
    "Exited Survey Room",
]

# ----------------- Text/Regex Helpers -----------------
def normalize_text(s: str) -> str:
    """Lowercase, Unicode-normalize, replace NBSP, collapse whitespace, strip ends."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower().replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_header_token(s: str) -> str:
    """Normalize header names to be robust to spaces/underscores/dots/hyphens."""
    return re.sub(r"[ _.\-]+", "", str(s).strip().lower())

def find_column_indices_relaxed(header, want_keys):
    """
    Case-insensitive, relaxed substring match on normalized header tokens.
    want_keys are strings like 'roomEvent','PlayerVR.x','Robot.x', etc.
    Returns dict {key -> col_idx or -1}.
    """
    normed = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        idx = -1
        for j, h in enumerate(normed):
            if k in h:  # substring after normalization
                idx = j
                break
        out[key] = idx
    return out

# ----------------- IO/Parsing Helpers -----------------
def load_csv_lenient(path: str) -> pd.DataFrame:
    """Read CSV robustly: engine='python', on_bad_lines='skip', all as strings first."""
    return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

def first5_index(val) -> str:
    """
    Build the 5-char index used in CSV filenames.
    - If the cell is int-like -> zero-pad to 5 (e.g., 123 -> '00123').
    - Otherwise -> take the first 5 characters of the trimmed string.
    """
    if isinstance(val, (int, np.integer)):
        return str(int(val)).zfill(5)
    if isinstance(val, float) and float(val).is_integer():
        return str(int(val)).zfill(5)
    s = str(val).strip()
    return s[:5]

def find_csv_for_index(idx5: str):
    """
    Search the four folders for a .csv whose filename starts with idx5;
    return (path, folder_label) or (None, None).
    folder_label is exactly one of the entries in SEARCH_FOLDERS that matched.
    """
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.lower().endswith(".csv"):
                continue
            if name[:5] == idx5:
                return os.path.join(folder, name), folder  # exact folder label
    return None, None

# ----------------- Tag matching rules (avoid overlap) -----------------
def event_matches_tag(evt_norm: str, tag_display: str) -> bool:
    """
    Return True iff the normalized event string belongs to the intended tag.
    We avoid overlap by excluding robot-* rows from plain 'Entered/Exited' tags.
    """
    if tag_display == "Robot entered Survey Room":
        return "robot entered survey room" in evt_norm
    elif tag_display == "Entered Survey Room":
        return ("entered survey room" in evt_norm) and ("robot entered survey room" not in evt_norm)
    elif tag_display == "Robot exited Survey Room":
        return "robot exited survey room" in evt_norm
    elif tag_display == "Exited Survey Room":
        return ("exited survey room" in evt_norm) and ("robot exited survey room" not in evt_norm)
    else:
        return False

# ----------------- Aggregator -----------------
def empty_agg():
    """Build the aggregator dict for each tag."""
    return {
        "robot_x": [], "robot_y": [], "robot_z": [],
        "player_x": [], "player_y": [], "player_z": [],
    }

def append_agg(agg, tag, rxyz_abs, pxyz_abs):
    agg[tag]["robot_x"].append(rxyz_abs[0])
    agg[tag]["robot_y"].append(rxyz_abs[1])
    agg[tag]["robot_z"].append(rxyz_abs[2])
    agg[tag]["player_x"].append(pxyz_abs[0])
    agg[tag]["player_y"].append(pxyz_abs[1])
    agg[tag]["player_z"].append(pxyz_abs[2])

def stats(arr):
    """Return (count, mean, sample variance) with ddof=1 if count>1 else variance=NaN."""
    n = len(arr)
    if n == 0:
        return 0, float("nan"), float("nan")
    a = np.array(arr, dtype=float)
    mean = float(np.nanmean(a))
    var  = float(np.nanvar(a, ddof=1)) if n > 1 else float("nan")
    return n, mean, var

def classify_folder_group(folder_label: str) -> str:
    """Return 'noshook' if folder contains 'noshook', else 'shook'."""
    return "noshook" if "noshook" in folder_label.replace("\\", "/").lower() else "shook"

# ----------------- Main -----------------
def main():
    # Prepare aggregator per tag
    agg = {tag: empty_agg() for tag in TAGS}
    # Per-tag counts split by folder group
    agg_counts = {tag: {"shook": 0, "noshook": 0} for tag in TAGS}

    # Load Excel and pull indices
    try:
        df_idx = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read Excel '{EXCEL_FILE}': {e}")
        return

    indices = df_idx.iloc[:, ID_COL].apply(first5_index)

    for idx5 in indices:
        if not idx5 or len(idx5) < 5:
            print(f"{idx5 or '<empty>'}: SKIP (invalid index)")
            continue

        csv_path, folder_label = find_csv_for_index(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP (no CSV found) [folder=none]")
            continue

        # Read CSV
        try:
            df = load_csv_lenient(csv_path)
            if df.shape[0] == 0:
                print(f"{idx5}: SKIP (empty CSV) [folder={folder_label}]")
                continue
        except Exception as e:
            print(f"{idx5}: SKIP (CSV read error: {e}) [folder={folder_label}]")
            continue

        # Required columns
        cols_needed = ["roomEvent", "PlayerVR.x", "PlayerVR.y", "PlayerVR.z",
                       "Robot.x", "Robot.y", "Robot.z"]
        colmap = find_column_indices_relaxed(list(df.columns), cols_needed)

        # Check columns
        missing_cols = [k for k in cols_needed if colmap[k] == -1]
        if missing_cols:
            for mc in missing_cols:
                print(f"{idx5}: '{mc}' column missing [folder={folder_label}]")
            print(f"{idx5}: SKIP (missing required column(s)) [folder={folder_label}]")
            continue

        # Resolve real column names
        ev_col = df.columns[colmap["roomEvent"]]
        p_cols = [df.columns[colmap["PlayerVR.x"]],
                  df.columns[colmap["PlayerVR.y"]],
                  df.columns[colmap["PlayerVR.z"]]]
        r_cols = [df.columns[colmap["Robot.x"]],
                  df.columns[colmap["Robot.y"]],
                  df.columns[colmap["Robot.z"]]]

        # Normalize roomEvent cells
        ev_norm = df[ev_col].astype(str).map(normalize_text)

        # Find rows for each tag and validate uniqueness
        tag_rows = {}
        duplicate_tags = []
        missing_tags = []
        for tag in TAGS:
            mask = ev_norm.apply(lambda s: event_matches_tag(s, tag))
            matches = np.flatnonzero(mask.to_numpy())
            if len(matches) == 0:
                missing_tags.append(tag)
            elif len(matches) > 1:
                duplicate_tags.append((tag, len(matches)))
            else:
                tag_rows[tag] = int(matches[0])

        if missing_tags:
            miss_list = "; ".join([f"'{t}'" for t in missing_tags])
            print(f"{idx5}: MISSING TAGS -> {miss_list} [folder={folder_label}]")
            print(f"{idx5}: SKIP (missing required tag(s)) [folder={folder_label}]")
            continue

        if duplicate_tags:
            dlist = ", ".join([f"'{t}'×{c}" for (t, c) in duplicate_tags])
            print(f"{idx5}: duplicate tag occurrences -> {dlist} [folder={folder_label}]")
            print(f"{idx5}: SKIP (duplicate tag(s) found) [folder={folder_label}]")
            continue

        # For each tag, extract positions; enforce -1 check (on raw values), then store ABS values
        all_valid = True
        per_tag_positions_abs = {}
        for tag, row_idx in tag_rows.items():
            try:
                # Coerce to floats (raw)
                px = float(df.at[row_idx, p_cols[0]])
                py = float(df.at[row_idx, p_cols[1]])
                pz = float(df.at[row_idx, p_cols[2]])
                rx = float(df.at[row_idx, r_cols[0]])
                ry = float(df.at[row_idx, r_cols[1]])
                rz = float(df.at[row_idx, r_cols[2]])
            except Exception:
                print(f"{idx5} '{tag}': position invalid (non-numeric) [folder={folder_label}]")
                all_valid = False
                break

            # Enforce -1 exclusion on raw values
            if any(v == -1 for v in (px, py, pz, rx, ry, rz)):
                print(f"{idx5} '{tag}': position invalid (-1 present) [folder={folder_label}]")
                all_valid = False
                break

            # Store ABSOLUTE values for aggregation
            rxyz_abs = (abs(rx), abs(ry), abs(rz))
            pxyz_abs = (abs(px), abs(py), abs(pz))
            per_tag_positions_abs[tag] = (rxyz_abs, pxyz_abs)

        if not all_valid:
            print(f"{idx5}: SKIP (invalid position value) [folder={folder_label}]")
            continue

        # Append to aggregator (absolute values) and count source group
        group = classify_folder_group(folder_label)
        for tag in TAGS:
            rxyz_abs, pxyz_abs = per_tag_positions_abs[tag]
            append_agg(agg, tag, rxyz_abs, pxyz_abs)
            agg_counts[tag][group] += 1

        print(f"{idx5}: OK (all four tags found, positions valid) [folder={folder_label}]")

    # ---------- Aggregate outputs ----------
    print("\n--- Aggregate Results (ABS positions) ---")
    for tag in TAGS:
        Rxs = agg[tag]["robot_x"]; Rys = agg[tag]["robot_y"]; Rzs = agg[tag]["robot_z"]
        Pxs = agg[tag]["player_x"]; Pys = agg[tag]["player_y"]; Pzs = agg[tag]["player_z"]

        nR, mxR, vxR = stats(Rxs); _, myR, vyR = stats(Rys); _, mzR, vzR = stats(Rzs)
        nP, mxP, vxP = stats(Pxs); _, myP, vyP = stats(Pys); _, mzP, vzP = stats(Pzs)

        # Counts should match within a tag; use robot count as N for display
        n_shook   = agg_counts[tag]["shook"]
        n_noshook = agg_counts[tag]["noshook"]
        n_total   = n_shook + n_noshook  # should equal nR

        print(f'\n"{tag}"')
        print(f"  N = {n_total}  (shook+baseline = {n_shook},  noshook+baseline = {n_noshook})")
        print(f"  Robot mean (abs) : ({mxR:.6f}, {myR:.6f}, {mzR:.6f})")
        print(f"  Robot var  (abs) : ({vxR:.6f}, {vyR:.6f}, {vzR:.6f})   # sample variance")
        print(f"  Player mean (abs): ({mxP:.6f}, {myP:.6f}, {mzP:.6f})")
        print(f"  Player var  (abs): ({vxP:.6f}, {vyP:.6f}, {vzP:.6f})   # sample variance")

if __name__ == "__main__":
    main()