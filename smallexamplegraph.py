 import os
import csv
import math
import matplotlib.pyplot as plt

# Folder and target file index
folder = "survey"
target_index = "83254"

# Find the correct CSV file
target_file = None
for file in os.listdir(folder):
    if file.endswith(".csv") and file.startswith(target_index):
        target_file = os.path.join(folder, file)
        break

if not target_file:
    print(f"No .csv file starting with index {target_index} found in '{folder}/'")
    exit()

# Parse and collect relevant data
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
        ev = header.index("roomevent")
    except ValueError:
        print("Missing required columns in header.")
        exit()

    data_rows = []
    event_time = None

    for row in reader:
        try:
            t = float(row[time_idx])
            p_pos = [float(row[px]), float(row[py]), float(row[pz])]
            r_pos = [float(row[rx]), float(row[ry]), float(row[rz])]
            event = row[ev].strip().lower()
        except ValueError:
            continue

        data_rows.append((t, p_pos, r_pos))

        if "robot entered survey room" in event and event_time is None:
            event_time = t

    if event_time is None:
        print("Event 'robot entered survey room' not found.")
        exit()

# Extract the 20-second window before the event
window_data = [row for row in data_rows if event_time - 20 <= row[0] < event_time]

# Compute speeds
times = []
player_speeds = []
robot_speeds = []

prev_p = None
prev_r = None

for t, p, r in window_data:
    if prev_p is None:
        times.append(t)
        player_speeds.append(0.0)
        robot_speeds.append(0.0)
    else:
        p_speed = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, prev_p)))
        r_speed = math.sqrt(sum((a - b) ** 2 for a, b in zip(r, prev_r)))
        times.append(t)
        player_speeds.append(p_speed)
        robot_speeds.append(r_speed)

    prev_p = p
    prev_r = r

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(times, player_speeds, label="Player Speed")
plt.plot(times, robot_speeds, label="Robot Speed")
plt.xlabel("Time (s)")
plt.ylabel("Speed")
plt.title(f"Speed (20s before event) — Index {target_index}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()