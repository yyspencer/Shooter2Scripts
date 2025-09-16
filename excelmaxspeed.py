import os
import numpy as np
import pandas as pd

def load_speed_data(txt_path):
    player, robot = [], []
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        line = line.strip()
        if line == '-1':
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            p, r = float(parts[0]), float(parts[1])
            player.append(p)
            robot.append(r)
        except ValueError:
            continue
    return np.array(player), np.array(robot)

def pearson_corr(x, y):
    return 0.0 if len(x) == 0 else np.corrcoef(x, y)[0, 1]

def cc_at_lag(x, y, lag):
    n = len(x)
    if lag > 0:
        xs, ys = x[:n - lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[:n + lag]
    else:
        xs, ys = x, y
    return 0.0 if len(xs) == 0 else pearson_corr(xs, ys)

def find_best_lag(x, y):
    n = len(x)
    if n == 0:
        return 0, 0.0
    L = n // 4
    best_lag = 0
    best_cc = 0.0
    for lag in range(-L, L + 1):
        cc = cc_at_lag(x, y, lag)
        if abs(cc) > abs(best_cc):
            best_lag, best_cc = lag, cc
    return best_lag, best_cc

def main():
    xlsx_path = "Shooter Study 2 Exit Choices .xlsx"
    speed_dir = "surveyspeed"

    if not os.path.exists(xlsx_path):
        print(f"Error: Excel file '{xlsx_path}' not found.")
        return
    if not os.path.isdir(speed_dir):
        print(f"Error: Folder '{speed_dir}' not found.")
        return

    df = pd.read_excel(xlsx_path, sheet_name=0)
    indices = df.iloc[:, 0].dropna().astype(str).str[:5].unique()

    print("=== Max CC(t) and Best Lag per Index ===")
    for idx in indices:
        txt_path = os.path.join(speed_dir, f"{idx}.txt")
        if not os.path.isfile(txt_path):
            print(f"Index {idx}: ❌ Not Found")
            continue

        player, robot = load_speed_data(txt_path)
        if len(player) == 0 or len(robot) == 0:
            print(f"Index {idx}: ⚠️ No valid data")
            continue

        best_lag, best_cc = find_best_lag(player, robot)
        #print(f"Index {idx}: bestLag = {best_lag}, CC(t) = {best_cc:.4f}")
        print(f"{best_cc:.4f}")

if __name__ == "__main__":
    main()