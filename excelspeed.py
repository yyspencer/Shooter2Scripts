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

# Step 1: Load indices from Excel
xlsx_path = "Shooter Study 2 Exit Choices .xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0)
indices = df.iloc[:, 0].dropna().astype(str).str[:5].unique()

# Step 2: Check each index in surveyspeed folder
speed_dir = "surveyspeed"
if not os.path.isdir(speed_dir):
    print("Error: 'surveyspeed' folder not found.")
else:
    print("=== Cross Correlation Results (CC(0)) ===")
    for idx in indices:
        txt_file = os.path.join(speed_dir, f"{idx}.txt")
        if not os.path.isfile(txt_file):
            print(f"❌ Not Found")
            continue

        player, robot = load_speed_data(txt_file)
        if len(player) == 0 or len(robot) == 0:
            print(f"Index {idx}: ⚠️ No valid data")
            continue

        cc0 = pearson_corr(player, robot)
        print(f"{cc0:.4f}")