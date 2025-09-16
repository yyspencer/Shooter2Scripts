import os
import csv
import math

# Constants
PROXIMITY_THRESHOLD = 2.0
FOLLOW_WINDOW = 10.0  # seconds
OFFSET = 0.229  # Estimated offset for noshook files

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def process_shook_folder(folder_path):
    displacements = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return displacements

    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue

        index = fname[:5]
        fpath = os.path.join(folder_path, fname)

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]

            try:
                time_idx = header.index("time")
                px, py, pz = header.index("playervr.x"), header.index("playervr.y"), header.index("playervr.z")
                rx, ry, rz = header.index("robot.x"), header.index("robot.y"), header.index("robot.z")
                event_idx = header.index("robotevent")
            except ValueError as e:
                print(f"❌ Skipping {fname}: {e}")
                continue

            rows, shook_idx = [], -1
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
                print(f"❌ {index}: No 'shook' in robotevent.")
                continue

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

            print(f"✅ (shook) Index {index}: Displacement = {displacement:.4f} m")
            displacements.append(displacement)
    return displacements

def process_noshook_folder(folder_path):
    displacements = []
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return displacements

    for fname in os.listdir(folder_path):
        if not fname.endswith(".csv"):
            continue

        index = fname[:5]
        fpath = os.path.join(folder_path, fname)

        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]

            try:
                time_idx = header.index("time")
                px, py, pz = header.index("playervr.x"), header.index("playervr.y"), header.index("playervr.z")
                rx, ry, rz = header.index("robot.x"), header.index("robot.y"), header.index("robot.z")
            except ValueError as e:
                print(f"❌ Skipping {fname}: {e}")
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

                if est_shook_time is None:
                    for cell in row:
                        if "0.2" in cell:
                            est_shook_time = t + OFFSET
                            break

            if est_shook_time is None:
                print(f"❌ {index}: No '0.2' found.")
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

            print(f"✅ (noshook) Index {index}: Displacement = {displacement:.4f} m")
            displacements.append(displacement)
    return displacements

# Run processing
shook_disps = process_shook_folder("./shook")
noshook_disps = process_noshook_folder("./noshook")
all_disps = shook_disps + noshook_disps

# Output summary
if all_disps:
    n = len(all_disps)
    mean_disp = sum(all_disps) / n
    var_disp = sum((d - mean_disp) ** 2 for d in all_disps) / n
    print("\n📊 Combined Summary Statistics:")
    print(f"👉 Total files processed: {n}")
    print(f"👉 Mean Displacement: {mean_disp:.4f} m")
    print(f"👉 Variance: {var_disp:.4f}")
else:
    print("\n⚠️ No displacement data found.")