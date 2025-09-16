#!/usr/bin/env python3
import os
import re
import unicodedata
import numpy as np
import pandas as pd

# ========= Config =========
CSV_ROOT = "noshookmodified"   # ONLY process CSVs directly in this folder (ignore 'baseline')

# Learned locations (ABS) and radii = 2σ
ENTER_MEAN  = np.array([24.594621, 0.0, 45.237331], dtype=float)
ENTER_VAR   = np.array([ 0.000659, 0.0,  0.000252], dtype=float)
EXIT_MEAN   = np.array([33.631669, 0.0, 45.163783], dtype=float)
EXIT_VAR    = np.array([ 0.001266, 0.0,  0.001222], dtype=float)
CRISIS_MEAN = np.array([27.699638, 0.0, 46.760727], dtype=float)
CRISIS_VAR  = np.array([ 0.137175, 0.0,  0.406943], dtype=float)

R_ENTER  = 2.0 * np.sqrt(np.sum(ENTER_VAR))
R_EXIT   = 2.0 * np.sqrt(np.sum(EXIT_VAR))
R_CRISIS = 2.0 * np.sqrt(np.sum(CRISIS_VAR))

# Learned delta (crisis time relative to move after longest stationary session)
DELTA_MEAN = -4.989027
DELTA_VAR  =  0.003289
DELTA_SIG  = float(np.sqrt(DELTA_VAR))
DELTA_WIN  = 2.0 * DELTA_SIG   # accept within ±2σ

# Thresholds
EPS_STATIONARY = 0.1    # meters: "still" threshold for building the stationary session
MIN_FIRST_MOVE = 1e-6   # meters: minimal first-move distance after stationary (sanity check)

# ========= Helpers =========
def normalize_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFC", str(s)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()

def norm_header_token(s: str) -> str:
    return re.sub(r"[ _.\-]+", "", normalize_text(s).lower())

def find_columns(header, want_keys):
    normed = [norm_header_token(h) for h in header]
    out = {}
    for key in want_keys:
        k = norm_header_token(key)
        j = -1
        for idx, tok in enumerate(normed):
            if k in tok:
                j = idx
                break
        out[key] = j
    return out

def load_csv_lenient(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

def coerce_float(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy()

def abs_pos_triplet(x, y, z):
    try:
        fx, fy, fz = float(x), float(y), float(z)
    except Exception:
        return None
    if fx == -1 or fy == -1 or fz == -1:
        return None
    return abs(fx), abs(fy), abs(fz)

def dist_euclid(p, q):
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(q, float)))

def nearest_time_index(times: np.ndarray, target_time: float) -> int:
    diffs = np.full_like(times, np.inf, dtype=float)
    mask = np.isfinite(times)
    diffs[mask] = np.abs(times[mask] - target_time)
    return int(np.argmin(diffs))

def append_event(df: pd.DataFrame, row_idx: int, evt_col_name: str, text: str):
    if evt_col_name not in df.columns:
        df[evt_col_name] = ""
    prev = str(df.at[row_idx, evt_col_name]) if row_idx < len(df) else ""
    prev = "" if prev == "nan" else prev
    df.at[row_idx, evt_col_name] = (prev + (" | " if prev else "") + text)

def find_first_in_sphere(t, rx, ry, rz, center, radius, start_idx=0):
    n = len(t)
    for i in range(int(start_idx), n):
        pos = abs_pos_triplet(rx[i], ry[i], rz[i])
        if pos is None: continue
        if dist_euclid(pos, center) <= radius:
            return i
    return -1

def longest_stationary_session_in_crisis(t, rx, ry, rz, enter_i, exit_i):
    best = (None, None, -1.0)  # (s, e, duration)
    s = None
    prev_pos = None
    for i in range(enter_i, exit_i + 1):
        pos = abs_pos_triplet(rx[i], ry[i], rz[i])
        if pos is None:
            if s is not None and i - 1 >= s:
                dur = float(t[i - 1]) - float(t[s])
                if dur > best[2]: best = (s, i - 1, dur)
            s = None; prev_pos = None; continue
        inside = (dist_euclid(pos, CRISIS_MEAN) <= R_CRISIS)
        if not inside:
            if s is not None and i - 1 >= s:
                dur = float(t[i - 1]) - float(t[s])
                if dur > best[2]: best = (s, i - 1, dur)
            s = None; prev_pos = None; continue
        if s is None:
            s = i; prev_pos = pos
        else:
            step = dist_euclid(pos, prev_pos)
            if step <= EPS_STATIONARY:
                prev_pos = pos
            else:
                dur = float(t[i - 1]) - float(t[s])
                if dur > best[2]: best = (s, i - 1, dur)
                s = i; prev_pos = pos
    if s is not None and exit_i >= s:
        dur = float(t[exit_i]) - float(t[s])
        if dur > best[2]: best = (s, exit_i, dur)
    return (best[0], best[1]) if best[0] is not None else (None, None)

def find_window_indices(t, rx, ry, rz, df, colmap):
    n = len(t)
    # 1) Proximity
    i_enter = find_first_in_sphere(t, rx, ry, rz, ENTER_MEAN, R_ENTER, start_idx=0)
    if i_enter != -1:
        i_exit = find_first_in_sphere(t, rx, ry, rz, EXIT_MEAN, R_EXIT, start_idx=i_enter + 1)
        if i_exit != -1 and i_exit > i_enter:
            return i_enter, i_exit, "proximity", True, True
    # 2) roomEvent textual tags
    room_idx = colmap.get("roomevent", -1)
    if room_idx != -1:
        room_col = df.columns[room_idx]
        ser = df[room_col].astype(str).map(lambda s: normalize_text(s).lower())
        ent_mask  = ser.str_contains = ser.str.contains("entered survey room", regex=False, na=False)
        exit_mask = ser.str_contains = ser.str.contains("exited survey room",  regex=False, na=False)
        ent_idxs  = np.flatnonzero(ent_mask.to_numpy())
        exit_idxs = np.flatnonzero(exit_mask.to_numpy())
        if ent_idxs.size > 0 and exit_idxs.size > 0:
            ent0 = int(ent_idxs[0]); exits_after = exit_idxs[exit_idxs >= ent0]
            if exits_after.size > 0:
                ex0 = int(exits_after[0])
                if ex0 > ent0:
                    return ent0, ex0, "roomEvent-tags", True, True
        if ent_idxs.size > 0:
            ent0 = int(ent_idxs[0]); return ent0, n - 1, "roomEvent-partial-enter", True, False
        if exit_idxs.size > 0:
            ex0 = int(exit_idxs[0]); return 0, ex0, "roomEvent-partial-exit", False, True
    # 3) Fallback: full file
    return 0, n - 1, "full-range", False, False

def fmt(x):
    return f"{x:.6g}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else "NaN"

# ========= Main =========
def main():
    counts = {
        "processed": 0,
        "csv_read_error": 0,
        "empty_csv": 0,
        "missing_cols": 0,
        "no_stationary": 0,
        "no_valid_move": 0,
        "too_small_move": 0,
        "nonpos_dt": 0,
        "gate_fail": 0,
        "write_fail": 0,
    }

    if not os.path.isdir(CSV_ROOT):
        print(f"❌ Folder not found: {CSV_ROOT}")
        return

    # Process every CSV directly under noshookmodified/
    for name in sorted(os.listdir(CSV_ROOT)):
        if not name.lower().endswith(".csv"):
            continue
        csv_path = os.path.join(CSV_ROOT, name)
        idx5 = name[:5]

        # Initialize per-file metrics for printing even when we skip
        win_src = "N/A"
        stationary_dur = np.nan
        first_move_dist = np.nan
        dt = np.nan
        speed = np.nan

        # Load CSV
        try:
            df = load_csv_lenient(csv_path)
        except Exception as e:
            counts["csv_read_error"] += 1
            print(f"{idx5}: SKIP csv_read_error [file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue
        if df.shape[0] == 0:
            counts["empty_csv"] += 1
            print(f"{idx5}: SKIP empty_csv [file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        # Locate required columns
        need = ["time", "robotevent", "roomevent", "robot.x", "robot.y", "robot.z"]
        colmap = find_columns(list(df.columns), need)
        missing = [k for k in ["time", "robot.x", "robot.y", "robot.z"] if colmap[k] == -1]
        if missing:
            counts["missing_cols"] += 1
            print(f"{idx5}: SKIP missing_cols({', '.join(missing)}) [file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        time_col = df.columns[colmap["time"]]
        evt_col  = df.columns[colmap["robotevent"]] if colmap["robotevent"] != -1 else "robotEvent"

        t  = coerce_float(df[time_col])
        rx = df[df.columns[colmap["robot.x"]]].to_numpy()
        ry = df[df.columns[colmap["robot.y"]]].to_numpy()
        rz = df[df.columns[colmap["robot.z"]]].to_numpy()

        n = len(t)
        if n == 0 or not np.any(np.isfinite(t)):
            counts["missing_cols"] += 1
            print(f"{idx5}: SKIP no_valid_time [file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        # Choose window
        i_enter, i_exit, win_src, enter_found, exit_found = find_window_indices(t, rx, ry, rz, df, colmap)

        # Longest stationary session in crisis sphere
        s_idx, e_idx = longest_stationary_session_in_crisis(t, rx, ry, rz, i_enter, i_exit)
        if s_idx is not None and e_idx is not None and e_idx >= s_idx and np.isfinite(t[e_idx]) and np.isfinite(t[s_idx]):
            stationary_dur = float(t[e_idx]) - float(t[s_idx])

        if s_idx is None or e_idx is None or e_idx < s_idx:
            counts["no_stationary"] += 1
            print(f"{idx5}: SKIP no_stationary [win={win_src}, file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        # First valid move after stationary
        pos_end = abs_pos_triplet(rx[e_idx], ry[e_idx], rz[e_idx])
        j = None
        for k in range(e_idx + 1, n):
            pk = abs_pos_triplet(rx[k], ry[k], rz[k])
            if pk is None or not np.isfinite(t[k]): continue
            first_move_dist = dist_euclid(pk, pos_end)
            j = k
            break

        if j is None:
            counts["no_valid_move"] += 1
            print(f"{idx5}: SKIP no_valid_move [win={win_src}, file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        dt = float(t[j]) - float(t[e_idx]) if np.isfinite(t[j]) and np.isfinite(t[e_idx]) else np.nan
        speed = (first_move_dist / dt) if (np.isfinite(dt) and dt > 0) else np.nan
        if not np.isfinite(dt) or dt <= 0:
            counts["nonpos_dt"] += 1
            print(f"{idx5}: SKIP nonpos_dt [win={win_src}, file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue
        if first_move_dist < MIN_FIRST_MOVE:
            counts["too_small_move"] += 1
            print(f"{idx5}: SKIP too_small_move({fmt(first_move_dist)} m) [win={win_src}, file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        # Crisis estimate using learned delta
        target_time = float(t[j]) + DELTA_MEAN
        j_near = nearest_time_index(t, target_time)
        if not np.isfinite(t[j_near]) or abs(float(t[j_near]) - target_time) > DELTA_WIN:
            counts["gate_fail"] += 1
            err = abs(float(t[j_near]) - target_time)
            print(f"{idx5}: SKIP gate_fail(|err|={fmt(err)} s) [win={win_src}, file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        # Append events — include first-move distance and dt/speed
        if enter_found: append_event(df, i_enter, evt_col, "Robot entered Survey Room")
        if exit_found:  append_event(df, i_exit,  evt_col, "Robot exited Survey Room")
        append_event(df, j,      evt_col,
                     f"Robot starts moving after longest stationary session (dist={fmt(first_move_dist)} m, dt={fmt(dt)} s, speed={fmt(speed)} m/s)")
        append_event(df, j_near, evt_col, "estimated shook")

        # Save CSV
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            counts["write_fail"] += 1
            print(f"{idx5}: SKIP write_fail [file={name}] — "
                  f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m")
            continue

        counts["processed"] += 1

        # Per-file detail (always includes the two metrics)
        print(f"{idx5}: OK [win={win_src}, file={name}] — "
              f"stationary_dur={fmt(stationary_dur)}s, first_move_dist={fmt(first_move_dist)} m, "
              f"Δt={fmt(dt)} s, speed={fmt(speed)} m/s; "
              f"target={t[j]:.3f}+({DELTA_MEAN:.3f}) → nearest={t[j_near]:.3f} (|err|={fmt(abs(t[j_near]-target_time))} s)")

    # ---- Summary ----
    print("\n===== SUMMARY =====")
    for k, v in counts.items():
        print(f"{k:>16}: {v}")

if __name__ == "__main__":
    main()