import os
import csv
import math

# Constants
target_x = -24.6070
target_z = -45.2378
threshold = 0.5 #0.3 is also good
data_dir = "noshook copy"

print("=== Robot Position Match (distance < 0.1) ===")

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".csv"):
        continue

    index = fname[:5]
    fpath = os.path.join(data_dir, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        header = [h.strip().lower() for h in raw_header]

        try:
            ix = header.index("robot.x")
            iz = header.index("robot.z")
        except ValueError:
            print(f"Index {index}: ❌ Missing 'robot.x' or 'robot.z' column")
            continue

        found = False
        row_number = 1  # Starts at 1 since header is line 1

        for row in reader:
            row_number += 1
            if len(row) <= max(ix, iz):
                continue

            try:
                x = float(row[ix])
                z = float(row[iz])
            except ValueError:
                continue

            dist = math.sqrt((x - target_x) ** 2 + (z - target_z) ** 2)
            if dist < threshold:
                print(f"Index {index}: ✅ Match at row {row_number}, x = {x:.4f}, z = {z:.4f}")
                found = True
                break

        if not found:
            print(f"Index {index}: ❌ No match found")