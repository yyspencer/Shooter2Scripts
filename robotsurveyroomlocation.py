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

# We handle these robot tags independently
ROBOT_TAGS = [
    "Robot entered Survey Room",
    "Robot exited Survey Room",
]

# Baseline means/variances (ABS positions), treated as fixed
BASELINES = {
    "Robot entered Survey Room": {
        "mean": (24.492535, 0.000000, 45.259259),
        "var":  (0.001239, 0.000000, 0.000274),
    },
    "Robot exited Survey Room": {
        "mean": (33.641574, 0.000000, 45.199716),
        "var":  (0.000803, 0.000000, 0.001289),
    },
}

EPS = 1e-6  # tolerance for zero-variance axes

# ----------------- Text/Regex Helpers -----------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s)).lower().replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", str(s).strip().lower())

def find_column_indices_relaxed(header, want_keys):
    """
    Case-insensitive, relaxed substring match on normalized header tokens.
    want_keys like 'roomEvent','PlayerVR.x','Robot.x', etc. -> {key: col_idx or -1}
    """
    normed = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        idx = -1
        for j, h in enumerate(normed):
            if k in h:
                idx = j
                break
        out[key] = idx
    return out

# ----------------- IO/Parsing Helpers -----------------
def load_csv_lenient(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

def index_5char(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v)).zfill(5)
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v)).zfill(5)
    return str(v).strip().zfill(5)

def find_csv_for_index(idx5: str):
    for folder in SEARCH_FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if name.lower().endswith(".csv") and name[:5] == idx5:
                return os.path.join(folder, name), folder
    return None, None

def classify_folder_group(folder_label: str) -> str:
    return "noshook" if "noshook" in folder_label.replace("\\", "/").lower() else "shook"

# ----------------- Aggregation -----------------
def empty_agg():
    return {
        "robot_x": [], "robot_y": [], "robot_z": [],
        "player_x": [], "player_y": [], "player_z": [],
    }

def agg_stats(arr):
    n = len(arr)
    if n == 0:
        return 0, float("nan"), float("nan")
    a = np.array(arr, dtype=float)
    mean = float(np.nanmean(a))
    var  = float(np.nanvar(a, ddof=1)) if n > 1 else float("nan")
    return n, mean, var

# ----------------- Tag Search Helpers -----------------
def find_tag_rows(df: pd.DataFrame, tag: str, room_event_col_idx: int):
    """
    Return list of row indices that contain the tag.
    - If room_event_col_idx != -1: search only that column (substring, case-insensitive).
    - If room_event_col_idx == -1: FULL-SCAN all cells; a row matches if ANY cell contains the tag.
    """
    key_norm = normalize_text(tag)

    if room_event_col_idx != -1:
        ev_col = df.columns[room_event_col_idx]
        ev_norm = df[ev_col].astype(str).map(normalize_text)
        mask = ev_norm.str.contains(re.escape(key_norm), regex=True, na=False)
        return list(np.flatnonzero(mask.to_numpy())), False  # not full-scan

    # FULL-SCAN
    try:
        df_norm = df.astype(str).map(normalize_text)  # pandas >= 2.2
    except AttributeError:
        df_norm = df.astype(str).applymap(normalize_text)  # fallback

    contains = df_norm.apply(lambda col: col.str.contains(re.escape(key_norm), regex=True, na=False))
    row_any = contains.any(axis=1)
    return list(np.flatnonzero(row_any.to_numpy())), True

def load_abs_positions_at_row(df: pd.DataFrame, p_cols, r_cols, row_idx: int):
    try:
        px = float(df.at[row_idx, p_cols[0]])
        py = float(df.at[row_idx, p_cols[1]])
        pz = float(df.at[row_idx, p_cols[2]])
        rx = float(df.at[row_idx, r_cols[0]])
        ry = float(df.at[row_idx, r_cols[1]])
        rz = float(df.at[row_idx, r_cols[2]])
    except Exception:
        return False, None, None
    # Drop invalid values
    if any(v == -1 for v in (px, py, pz, rx, ry, rz)):
        return False, None, None
    return True, (abs(rx), abs(ry), abs(rz)), (abs(px), abs(py), abs(pz))

def within_two_sigma(abs_pos, tag: str) -> bool:
    mx, my, mz = BASELINES[tag]["mean"]
    vx, vy, vz = BASELINES[tag]["var"]
    sx, sy, sz = (np.sqrt(vx), np.sqrt(vy), np.sqrt(vz))

    x, y, z = abs_pos
    ok_x = abs(x - mx) <= (2.0 * sx if sx > 0 else EPS)
    ok_y = abs(y - my) <= (2.0 * sy if sy > 0 else EPS)
    ok_z = abs(z - mz) <= (2.0 * sz if sz > 0 else EPS)
    return bool(ok_x and ok_y and ok_z)

def process_tag_rows(tag, row_idxs, df, p_cols, r_cols):
    """
    Returns:
      accepted_robot_abs, accepted_player_abs, info_dict
    Logic:
      - Collect VALID rows (positions parse ok and != -1)
      - If exactly 1 VALID row -> accept it automatically
      - If >=2 VALID rows -> accept those within ±2σ (component-wise) vs BASELINES[tag]
    """
    valid = []
    for ridx in row_idxs:
        ok, r_abs, p_abs = load_abs_positions_at_row(df, p_cols, r_cols, int(ridx))
        if ok:
            valid.append((int(ridx), r_abs, p_abs))

    accepted_r, accepted_p = [], []

    if len(valid) == 1:
        # Single occurrence: accept automatically
        _, r_abs, p_abs = valid[0]
        accepted_r.append(r_abs); accepted_p.append(p_abs)
        rule = "singleton -> auto-accept"
    elif len(valid) >= 2:
        # Duplicates: apply 2σ filter
        for _, r_abs, p_abs in valid:
            if within_two_sigma(r_abs, tag):
                accepted_r.append(r_abs); accepted_p.append(p_abs)
        rule = "duplicates -> 2σ filter"
    else:
        rule = "no valid rows"

    info = {
        "found": len(row_idxs),
        "valid": len(valid),
        "accepted": len(accepted_r),
        "rule": rule,
        "accepted_positions": accepted_r[:5],  # preview first few
    }
    return accepted_r, accepted_p, info

# ----------------- Main -----------------
def main():
    # Aggregators separated by group and tag
    agg = {
        "shook":   {tag: empty_agg() for tag in ROBOT_TAGS},
        "noshook": {tag: empty_agg() for tag in ROBOT_TAGS},
    }

    # Load Excel indices
    try:
        df_idx = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    except Exception as e:
        print(f"Failed to read Excel '{EXCEL_FILE}': {e}")
        return

    indices = df_idx.iloc[:, ID_COL].apply(index_5char)

    for idx5 in indices:
        csv_path, folder_label = find_csv_for_index(idx5)
        if not csv_path:
            print(f"{idx5}: SKIP (no CSV found) [folder=none]")
            continue

        group = classify_folder_group(folder_label)

        # Load CSV
        try:
            df = load_csv_lenient(csv_path)
            if df.shape[0] == 0:
                print(f"{idx5}: SKIP (empty CSV) [folder={folder_label}]")
                continue
        except Exception as e:
            print(f"{idx5}: SKIP (CSV read error: {e}) [folder={folder_label}]")
            continue

        # Resolve columns (positions required)
        cols_needed = ["roomEvent", "PlayerVR.x", "PlayerVR.y", "PlayerVR.z",
                       "Robot.x", "Robot.y", "Robot.z"]
        colmap = find_column_indices_relaxed(list(df.columns), cols_needed)

        room_event_idx = colmap["roomEvent"]  # may be -1 → full-scan

        pos_missing = [k for k in ["PlayerVR.x","PlayerVR.y","PlayerVR.z","Robot.x","Robot.y","Robot.z"]
                       if colmap[k] == -1]
        if pos_missing:
            print(f"{idx5}: SKIP (missing position column(s): {', '.join(pos_missing)}) [folder={folder_label}]")
            continue

        p_cols = [df.columns[colmap["PlayerVR.x"]],
                  df.columns[colmap["PlayerVR.y"]],
                  df.columns[colmap["PlayerVR.z"]]]
        r_cols = [df.columns[colmap["Robot.x"]],
                  df.columns[colmap["Robot.y"]],
                  df.columns[colmap["Robot.z"]]]

        per_index_msgs = []
        # Process each tag independently
        for tag in ROBOT_TAGS:
            rows, used_full_scan = find_tag_rows(df, tag, room_event_idx)
            r_acc, p_acc, info = process_tag_rows(tag, rows, df, p_cols, r_cols)

            # Aggregate accepted rows
            for (rx, ry, rz), (px, py, pz) in zip(r_acc, p_acc):
                agg[group][tag]["robot_x"].append(rx)
                agg[group][tag]["robot_y"].append(ry)
                agg[group][tag]["robot_z"].append(rz)
                agg[group][tag]["player_x"].append(px)
                agg[group][tag]["player_y"].append(py)
                agg[group][tag]["player_z"].append(pz)

            src = "full-scan" if used_full_scan else "roomEvent"
            per_index_msgs.append(
                f'{tag}: found={info["found"]} via {src}, valid={info["valid"]}, accepted={info["accepted"]} ({info["rule"]})'
                + ("" if info["accepted"] == 0 else f", sample accepted={info['accepted_positions']}")
            )

        print(f"{idx5} [folder={folder_label}, group={group}] -> " + " | ".join(per_index_msgs))

    # ---------- Aggregate outputs (SEPARATE sections) ----------
    def print_group_stats(group_label: str):
        print(f"\n--- Aggregate Results (ABS positions) — {group_label.upper()} ONLY ---")
        for tag in ROBOT_TAGS:
            Rxs = agg[group_label][tag]["robot_x"]; Rys = agg[group_label][tag]["robot_y"]; Rzs = agg[group_label][tag]["robot_z"]
            Pxs = agg[group_label][tag]["player_x"]; Pys = agg[group_label][tag]["player_y"]; Pzs = agg[group_label][tag]["player_z"]

            nR, mxR, vxR = agg_stats(Rxs); _, myR, vyR = agg_stats(Rys); _, mzR, vzR = agg_stats(Rzs)
            nP, mxP, vxP = agg_stats(Pxs); _, myP, vyP = agg_stats(Pys); _, mzP, vzP = agg_stats(Pzs)

            print(f'\n"{tag}"')
            print(f"  N (rows accepted) = {nR}")
            print(f"  Robot mean (abs) : ({mxR:.6f}, {myR:.6f}, {mzR:.6f})")
            print(f"  Robot var  (abs) : ({vxR:.6f}, {vyR:.6f}, {vzR:.6f})   # sample variance")
            print(f"  Player mean (abs): ({mxP:.6f}, {myP:.6f}, {mzP:.6f})")
            print(f"  Player var  (abs): ({vxP:.6f}, {vyP:.6f}, {vzP:.6f})   # sample variance")

    print_group_stats("shook")
    print_group_stats("noshook")

if __name__ == "__main__":
    main()