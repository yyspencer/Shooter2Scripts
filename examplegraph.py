import os
import csv
import matplotlib.pyplot as plt
import math

# Folder and target index
folder = "survey"
target_index = "81235"

# Step 1: Find the file starting with the index
target_file = None
for file in os.listdir(folder):
    if file.endswith(".csv") and file.startswith(target_index):
        target_file = os.path.join(folder, file)
        break

if not target_file:
    print(f"No .csv file starting with index {target_index} found in '{folder}/'")
    exit()

# Step 2: Read and parse file
with open(target_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    raw_header = next(reader)
    header = [col.strip().lower() for col in raw_header]

    try:
        time_idx = header.index("time")
        px = header.index("playervr.x")
        py = header.index("playervr.y")
        pz = header.index("playervr.z")
        rx = header.index("robot.x")
        ry = header.index("robot.y")
        rz = header.index("robot.z")
    except ValueError:
        print("Missing required columns in header.")
        exit()

    last_time = None
    player_prev = None
    robot_prev = None

    times = []
    player_speeds = []
    robot_speeds = []

    for row in reader:
        try:
            t = float(row[time_idx])
            p_pos = [float(row[px]), float(row[py]), float(row[pz])]
            r_pos = [float(row[rx]), float(row[ry]), float(row[rz])]
        except ValueError:
            continue

        if last_time is None:
            last_time = t
            player_prev = p_pos
            robot_prev = r_pos
            times.append(t)
            player_speeds.append(0.0)
            robot_speeds.append(0.0)
            continue

        if t - last_time >= 0.1:
            player_speed = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_pos, player_prev)))
            robot_speed = math.sqrt(sum((a - b) ** 2 for a, b in zip(r_pos, robot_prev)))

            times.append(t)
            player_speeds.append(player_speed)
            robot_speeds.append(robot_speed)

            last_time = t
            player_prev = p_pos
            robot_prev = r_pos

# Step 3: Plot
plt.figure(figsize=(10, 5))
plt.plot(times, player_speeds, label="Player Speed")
plt.plot(times, robot_speeds, label="Robot Speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed")
plt.title(f"Velocity of Player and Robot (Index {target_index})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()