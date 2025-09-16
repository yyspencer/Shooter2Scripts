import os
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ---------------------------------------------------------
def load_surveyspeed_data(txt_path):
    """Return two NumPy arrays (player, robot) after skipping rows that contain -1."""
    player, robot = [], []
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        line = line.strip()
        if line == "-1":
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
    """Pearson correlation when y is shifted by `lag` samples."""
    n = len(x)
    if lag > 0:
        xs, ys = x[: n - lag], y[lag:]
    elif lag < 0:
        xs, ys = x[-lag:], y[: n + lag]
    else:
        xs, ys = x, y
    return 0.0 if len(xs) == 0 else pearson_corr(xs, ys)


# ---------- main ------------------------------------------------------------
def main():
    surveyspeed_dir = "surveyspeed"
    if not os.path.isdir(surveyspeed_dir):
        print("Error: 'surveyspeed' folder not found.")
        return

    files = [f for f in os.listdir(surveyspeed_dir) if f.endswith(".txt")]
    if not files:
        print("No .txt files in 'surveyspeed'.")
        return

    data = {}          # index -> (player, robot)
    cc0_list = []      # CC(0) for each index
    lengths = []       # valid length of each index

    # 讀取所有檔案
    for name in files:
        idx = os.path.splitext(name)[0]
        p, r = load_surveyspeed_data(os.path.join(surveyspeed_dir, name))
        if len(p) == 0:
            print(f"{idx} => No valid data")
            continue
        data[idx] = (p, r)
        cc0_list.append(pearson_corr(p, r))
        lengths.append(len(p))

    if not data:
        print("No indices with valid data.")
        return

    # 全域 lag 範圍
    global_L = min(l // 4 for l in lengths)
    lags = list(range(-global_L, global_L + 1))

    mean_cc_by_lag = []
    bestLag, bestSumAbs = 0, -1.0

    # 計算每個 lag 的平均 CC 以及 |CC| 總和
    for lag in lags:
        ccs = [cc_at_lag(p, r, lag) for p, r in data.values()]
        mean_cc_by_lag.append(np.mean(ccs))            # 簽名平均
        total_abs = np.sum(np.abs(ccs))                # 用於找最佳 lag
        if total_abs > bestSumAbs:
            bestSumAbs, bestLag = total_abs, lag

    # 依最佳 lag 取各 index 的 CC
    best_cc_list = [cc_at_lag(p, r, bestLag) for p, r in data.values()]

    # ----------- 輸出結果 -----------------
    print(f"\n=== Results per index (global bestLag = {bestLag}) ===")
    for idx, (p, r) in data.items():
        cc0 = pearson_corr(p, r)
        ccb = cc_at_lag(p, r, bestLag)
        print(f"{idx}  CC(0)={cc0:.4f},  CC({bestLag})={ccb:.4f}")

    mean_cc0  = np.mean(cc0_list)
    var_cc0   = np.var(cc0_list)
    mean_best = np.mean(best_cc_list)
    var_best  = np.var(best_cc_list)

    print("\n===== Summary across all indices =====")
    print(f"Count indices            : {len(cc0_list)}")
    print(f"Number of lag points     : {len(lags)}")
    print(f"Mean  CC(0)              : {mean_cc0:.4f}")
    print(f"Var   CC(0)              : {var_cc0:.6f}")
    print(f"Global bestLag (samples) : {bestLag}")
    print(f"Mean  CC(bestLag)        : {mean_best:.4f}")
    print(f"Var   CC(bestLag)        : {var_best:.6f}")

    # ----------- 繪圖 -----------------
    plt.figure()
    plt.plot(lags, mean_cc_by_lag, marker="o")
    plt.axvline(bestLag, linestyle="--", label=f"bestLag = {bestLag}")
    plt.xlabel("Lag (t)")
    plt.ylabel("Average CC(t) across indices")
    plt.title("Shooter Survey Average Cross-Correlation vs Lag")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()