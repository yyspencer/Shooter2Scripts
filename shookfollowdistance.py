import os
import csv
import math

# Constants
PROXIMITY_THRESHOLD = 2.0
FOLLOW_WINDOW = 10.0  # seconds

# Path to the folder containing .csv files with "shook" events
shook_folder = os.path.join("./", "shook")

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

displacements = []

for fname in os.listdir(shook_folder):
    if not fname.endswith(".csv"):
        continue

    index = fname[:5]
    fpath = os.path.join(shook_folder, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        header = [h.strip().lower() for h in raw_header]

        try:
            time_idx = header.index("time")
            px, py, pz = header.index("playervr.x"), header.index("playervr.y"), header.index("playervr.z")
            rx, ry, rz = header.index("robot.x"), header.index("robot.y"), header.index("robot.z")
            event_idx = header.index("robotevent")
        except ValueError as e:
            print(f"❌ Skipping {fname} due to missing column: {e}")
            continue

        rows = []
        shook_idx = -1
        for i, row in enumerate(reader):
            if len(row) <= max(time_idx, px, py, pz, rx, ry, rz, event_idx):
                continue
            try:
                t = float(row[time_idx])
                player = [float(row[px]), float(row[py]), float(row[pz])]
                robot = [float(row[rx]), float(row[ry]), float(row[rz])]
                event = row[event_idx].strip().lower()
            except ValueError:
                continue
            rows.append((t, player, robot, event))
            if shook_idx == -1 and "shook" in event:
                shook_idx = i

        if shook_idx == -1:
            print(f"❌ {index}: No 'shook' substring found in robotevent.")
            continue

        shook_time = rows[shook_idx][0]
        displacement = 0.0
        prev_player_pos = None

        for i in range(shook_idx + 1, len(rows)):
            curr_time, _, robot_pos, _ = rows[i]

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