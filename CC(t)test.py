#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd

# ====== Config for this diagnostic run ======
IDX5         = "76018"          # fixed index
FOLDER       = "shook"          # only look here
TIME_COL     = "time"
ROOM_EVENT   = "roomevent"
PLAYER_COLS  = ["playervr.x", "playervr.y", "playervr.z"]
ROBOT_COLS   = ["robot.x", "robot.y", "robot.z"]
LAG_CAP      = 1000             # cap lag window to [-1000, +1000]

# ---------- Helpers ----------
def normalize_header_token(s: str) -> str:
    """Lowercase, strip, remove spaces/._- for robust header matching."""
    return re.sub(r"[ \t._-]+", "", str(s).strip().lower())

def find_cols(df: pd.DataFrame, names):
    """Return actual column names for requested normalized names; None if missing."""
    norm2real = {normalize_header_token(c): c for c in df.columns}
    return [norm2real.get(normalize_header_token(name)) for name in names]

def find_col(df: pd.DataFrame, name):
    return find_cols(df, [name])[0]

def coerce_float_series(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()

def build_speed_pairs(times, p_xyz, r_xyz):
    """
    Speeds between strictly consecutive rows (i-1 -> i).
    Drops segments with non-finite time, non-positive dt,
    OR any non-finite coord OR any -1 coord at either end.
    Returns (pairs, stats) where pairs = [(i, p_speed, r_speed), ...]
    """
    n = len(times)
    pairs = []
    prev = None
    dropped_nonfinite_time = dropped_nonposdt = dropped_nonfinite_xyz = dropped_minus1 = 0
    for i in range(n):
        if prev is None:
            prev = i
            continue
        if i - prev != 1:
            prev = i
            continue
        t0, t1 = times[prev], times[i]
        if not (np.isfinite(t0) and np.isfinite(t1)):
            dropped_nonfinite_time += 1
            prev = i
            continue
        if t1 <= t0:
            dropped_nonposdt += 1
            prev = i
            continue
        p0, p1 = p_xyz[prev], p_xyz[i]
        r0, r1 = r_xyz[prev], r_xyz[i]
        if (not np.isfinite(p0).all()) or (not np.isfinite(p1).all()) or \
           (not np.isfinite(r0).all()) or (not np.isfinite(r1).all()):
            dropped_nonfinite_xyz += 1
            prev = i
            continue
        if (p0 == -1).any() or (p1 == -1).any() or (r0 == -1).any() or (r1 == -1).any():
            dropped_minus1 += 1
            prev = i
            continue
        dp = np.linalg.norm(p1 - p0)
        dr = np.linalg.norm(r1 - r0)
        pairs.append((i, dp / (t1 - t0), dr / (t1 - t0)))
        prev = i
    stats = {
        "built": len(pairs),
        "dropped_nonfinite_time": dropped_nonfinite_time,
        "dropped_nonpos_dt": dropped_nonposdt,
        "dropped_nonfinite_xyz": dropped_nonfinite_xyz,
        "dropped_minus1": dropped_minus1
    }
    return pairs, stats

def pearson_raw(xs, ys):
    """Return raw np.corrcoef result (may be NaN)."""
    return float(np.corrcoef(xs, ys)[0, 1])

def cc_at_lag_windows(x, y, lag):
    """Return xs, ys (overlapping windows) for a given lag."""
    n = len(x)
    if lag > 0:
        xs, ys = x[:n - lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[:n + lag]
    else:
        xs, ys = x, y
    return xs, ys

def explain_nan_cc(x, y, lags):
    """
    For lags where raw CC is NaN, print diagnostics:
    - overlap length
    - std(xs), std(ys) and whether a series is constant
    - any non-finite in windows
    """
    nan_lags = []
    for lag in lags:
        xs, ys = cc_at_lag_windows(x, y, lag)
        if len(xs) == 0 or len(ys) == 0:
            continue
        r = pearson_raw(xs, ys)
        if not np.isfinite(r):
            nan_lags.append(lag)
    if not nan_lags:
        print("    No NaNs in CC(t) across tested lags.")
        return
    print(f"    NaN CC(t) occurred at {len(nan_lags)} lag(s): {nan_lags[:10]}{' ...' if len(nan_lags)>10 else ''}")
    for lag in nan_lags[:10]:
        xs, ys = cc_at_lag_windows(x, y, lag)
        has_nonfinite_x = not np.isfinite(xs).all()
        has_nonfinite_y = not np.isfinite(ys).all()
        sx = float(np.std(xs)) if len(xs)>0 else np.nan
        sy = float(np.std(ys)) if len(ys)>0 else np.nan
        const_x = np.isfinite(sx) and sx == 0.0
        const_y = np.isfinite(sy) and sy == 0.0
        print(f"      lag={lag:+d}: len={len(xs)}, std_x={sx:.6g} (const={const_x}), "
              f"std_y={sy:.6g} (const={const_y}), "
              f"nonfinite_in_x={has_nonfinite_x}, nonfinite_in_y={has_nonfinite_y}")

# ---------- Main diagnostic ----------
def main():
    # 1) Locate CSV
    csv_path = None
    if os.path.isdir(FOLDER):
        for name in os.listdir(FOLDER):
            if name.lower().endswith(".csv") and name.startswith(IDX5):
                csv_path = os.path.join(FOLDER, name)
                break
    if not csv_path:
        print(f"❌ No CSV starting with {IDX5} in '{FOLDER}/'.")
        return
    print(f"CSV: {csv_path}")

    # 2) Load CSV
    try:
        dfr = pd.read_csv(csv_path, dtype=str, on_bad_lines="skip")
    except Exception as e:
        print(f"❌ CSV read error: {e}")
        return

    # 3) Normalize headers and map columns
    dfr.columns = [normalize_header_token(c) for c in dfr.columns]
    tcol  = find_col(dfr, TIME_COL)
    pcols = find_cols(dfr, PLAYER_COLS)
    rcols = find_cols(dfr, ROBOT_COLS)
    rroom = find_col(dfr, ROOM_EVENT)  # may be None

    print("Column mapping:")
    print(f"  time     -> {tcol}")
    print(f"  player   -> {pcols}")
    print(f"  robot    -> {rcols}")
    print(f"  roomevent-> {rroom}")

    missing = []
    if tcol is None: missing.append("time")
    if any(c is None for c in pcols):
        missing += [PLAYER_COLS[i] for i,c in enumerate(pcols) if c is None]
    if any(c is None for c in rcols):
        missing += [ROBOT_COLS[i]  for i,c in enumerate(rcols) if c is None]
    if missing:
        print(f"❌ Missing required columns: {', '.join(missing)}")
        return

    # 4) Parse numerics
    times = coerce_float_series(dfr[tcol])
    pxyz  = dfr[pcols].astype(float).to_numpy()
    rxyz  = dfr[rcols].astype(float).to_numpy()

    n = len(times)
    nonfinite_time = np.count_nonzero(~np.isfinite(times))
    print(f"Rows: {n}, non-finite times: {nonfinite_time}")

    # 5) Survey room exclusion
    in_survey = np.zeros(n, dtype=bool)
    if rroom is not None:
        ser = dfr[rroom].astype(str).str.lower()
        ent_mask  = ser.str.contains("robot entered survey room", regex=False, na=False)
        exit_mask = ser.str.contains("robot exited survey room",  regex=False, na=False)
        enters = np.flatnonzero(ent_mask.to_numpy())
        exits  = np.flatnonzero(exit_mask.to_numpy())
        total_survey_rows = 0
        for s, e in zip(enters, exits):
            if s < e:
                in_survey[s:e+1] = True
                total_survey_rows += (e - s + 1)
        print(f"Survey intervals: {len(enters)} enter(s)/{len(exits)} exit(s), "
              f"rows excluded by survey mask: {total_survey_rows}")
    else:
        print("Survey room column missing; no survey exclusion applied.")

    # Mask survey rows by marking coords -1 (so speed builder drops them)
    pxyz_masked = pxyz.copy()
    rxyz_masked = rxyz.copy()
    pxyz_masked[in_survey, :] = -1
    rxyz_masked[in_survey, :] = -1

    # NEW: also mask rows with any non-finite position values → drop them like survey rows
    nonfinite_rows = ~(np.isfinite(pxyz).all(axis=1) & np.isfinite(rxyz).all(axis=1))
    if np.any(nonfinite_rows):
        pxyz_masked[nonfinite_rows, :] = -1
        rxyz_masked[nonfinite_rows, :] = -1
        print(f"Non-finite position rows masked: {int(nonfinite_rows.sum())}")

    # 6) Build speeds
    speed_pairs, drop_stats = build_speed_pairs(times, pxyz_masked, rxyz_masked)
    print("Speed pair stats:", drop_stats)
    if not speed_pairs:
        print("❌ No valid speed pairs after filtering. Cannot compute CC.")
        return

    p_speeds = np.array([ps for (_, ps, _) in speed_pairs], dtype=float)
    r_speeds = np.array([rs for (_, _, rs) in speed_pairs], dtype=float)

    # Drop any non-finite speed pairs (belt-and-suspenders)
    finite_mask = np.isfinite(p_speeds) & np.isfinite(r_speeds)
    dropped = int((~finite_mask).sum())
    if dropped:
        print(f"Dropped {dropped} non-finite speed pair(s) before CC")
    p_speeds = p_speeds[finite_mask]
    r_speeds = r_speeds[finite_mask]

    if len(p_speeds) < 2:
        print("❌ Too few finite speed pairs after filtering. Cannot compute CC.")
        return

    print(f"Built {len(p_speeds)} finite speed pairs.")
    print(f"  p_speeds: finite={np.isfinite(p_speeds).all()}, "
          f"mean={np.nanmean(p_speeds):.6g}, std={np.nanstd(p_speeds):.6g}")
    print(f"  r_speeds: finite={np.isfinite(r_speeds).all()}, "
          f"mean={np.nanmean(r_speeds):.6g}, std={np.nanstd(r_speeds):.6g}")

    # 7) Lag window (capped)
    L_data = max(1, len(p_speeds) // 4)
    L = min(L_data, LAG_CAP)
    lags = list(range(-L, L + 1))
    print(f"Lag window: L_data={L_data}, cap={LAG_CAP} → using [{-L}..{L}] (K={len(lags)})")

    # 8) Compute raw CC(t) over lags; detect NaNs; find best lag (ignoring NaN)
    cc_raw = []
    for lag in lags:
        xs, ys = cc_at_lag_windows(p_speeds, r_speeds, lag)
        if len(xs) == 0:
            cc_raw.append(np.nan)
            continue
        try:
            r = pearson_raw(xs, ys)
        except Exception:
            r = np.nan
        cc_raw.append(r)
    cc_raw = np.array(cc_raw, dtype=float)

    # Report NaNs with diagnostics
    if np.any(~np.isfinite(cc_raw)):
        print(f"Found NaN/Inf in CC(t): count={np.count_nonzero(~np.isfinite(cc_raw))}")
        explain_nan_cc(p_speeds, r_speeds, lags)
    else:
        print("No NaNs in CC(t) — all finite.")

    # Best lag & CC(best): ignore NaN by treating them as -inf in argmax
    cc_arg = np.where(np.isfinite(cc_raw), cc_raw, -np.inf)
    if np.all(~np.isfinite(cc_arg)):
        print("❌ All CC(t) values are NaN. Cannot determine best lag.")
        return

    best_idx = int(np.argmax(cc_arg))
    best_lag = lags[best_idx]
    best_cc_raw = cc_raw[best_idx]
    best_cc_write = 0.0 if not np.isfinite(best_cc_raw) else float(best_cc_raw)

    note = ""
    if not np.isfinite(best_cc_raw):
        note = " [ERR_NAN_AT_BEST→using 0.0]"
    print(f"\nRESULT for {IDX5}: best lag={best_lag}, CC(best)={best_cc_write:.4f}{note}")

if __name__ == "__main__":
    main()