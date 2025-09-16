import os
import csv
import math

# Constants
PROXIMITY_THRESHOLD = 2.0
FOLLOW_WINDOW = 10.0  # seconds
OFFSET = 0.229  # Time added after "0.2" tag to estimate "shook" time

# Folder containing noshook CSVs
noshook_folder = os.path.join("./", "noshook")

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

displacements = []

for fname in os.listdir(noshook_folder):
    if not fname.endswith(".csv"):
        continue

    index = fname[:5]
    fpath = os.path.join(noshook_folder, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        header = [h.strip().lower() for h in raw_header]

        try:
            time_idx = header.index("time")
            px, py, pz = header.index("playervr.x"), header.index("playervr.y"), header.index("playervr.z")
            rx, ry, rz = header.index("robot.x"), header.index("robot.y"), header.index("robot.z")
        except ValueError as e:
            print(f"❌ Skipping {fname} due to missing column: {e}")
            continue

        rows = []
        est_shook_time = None
        for row in reader:
            if len(row) <= max(time_idx, px, py, pz, rx, ry, rz):
                continue
            try:
                t = float(row[time_idx])
                player = [float(row[px]), float(row[py]), float(row[pz])]
                robot = [float(row[rx]), float(row[ry]), float(row[rz])]
            except ValueError:
                continue
            rows.append((t, player, robot, row))

            # Find "0.2" marker
            if est_shook_time is None:
                for cell in row:
                    if "0.2" in cell:
                        est_shook_time = t + OFFSET
                        break

        if est_shook_time is None:
            print(f"❌ {index}: No '0.2' marker found.")
            continue

        displacement = 0.0
        prev_player_pos = None

        for i in range(len(rows)):
            curr_time, _, robot_pos, _ = rows[i]
            if curr_time < est_shook_time:
                continue

            matched_player = None
            for j in range(i, -1, -1):
                t_j, player_pos_j, _, _ = rows[j]
                if curr_time - t_j > FOLLOW_WINDOW:
                    break
                if euclidean_distance(player_pos_j, robot_pos) <= PROXIMITY_THRESHOLD:
                    matched_player = player_pos_j
                    break

            if matched_player:
                if prev_player_pos is not None:
                    displacement += euclidean_distance(prev_player_pos, matched_player)
                prev_player_pos = matched_player

        print(f"✅ Index {index}: Total following displacement = {displacement:.4f} meters")
        displacements.append(displacement)

# Summary stats
if displacements:
    n = len(displacements)
    mean_disp = sum(displacements) / n
    var_disp = sum((d - mean_disp) ** 2 for d in displacements) / n
    print("\n📊 Summary Statistics:")
    print(f"👉 Number of files processed: {n}")
    print(f"👉 Mean following displacement: {mean_disp:.4f} meters")
    print(f"👉 Variance: {var_disp:.4f}")
else:
    print("\n⚠️ No valid displacement data to compute summary statistics.")