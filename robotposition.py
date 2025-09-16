import os
import csv
import numpy as np

survey_dir = "survey"
target_event = "robot entered survey room"

robot_x, robot_y, robot_z = [], [], []

for filename in os.listdir(survey_dir):
    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(survey_dir, filename)

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        raw_header = next(reader)

        # Normalize header: trim and lowercase
        header = [h.strip().lower() for h in raw_header]

        try:
            ix = header.index("robot.x")
            iy = header.index("robot.y")
            iz = header.index("robot.z")
            ie = header.index("roomevent")
        except ValueError:
            print(f"Skipping {filename}: missing required column.")
            continue

        for row in reader:
            if len(row) <= max(ix, iy, iz, ie):
                continue
            if row[ie].strip().lower() == target_event:
                try:
                    x = float(row[ix])
                    y = float(row[iy])
                    z = float(row[iz])
                    robot_x.append(x)
                    robot_y.append(y)
                    robot_z.append(z)
                except ValueError:
                    print(f"Invalid position value in {filename}")
                break

# Final stats
x_arr = np.array(robot_x)
y_arr = np.array(robot_y)
z_arr = np.array(robot_z)

if len(x_arr) > 0:
    print("Robot.x → Mean: {:.4f}, Variance: {:.6f}".format(np.mean(x_arr), np.var(x_arr)))
    print("Robot.y → Mean: {:.4f}, Variance: {:.6f}".format(np.mean(y_arr), np.var(y_arr)))
    print("Robot.z → Mean: {:.4f}, Variance: {:.6f}".format(np.mean(z_arr), np.var(z_arr)))
else:
    print("No valid data found.")