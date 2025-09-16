import os
import csv
import math

# Constants
target_x = -24.6070
target_z = -45.2378
threshold = 0.5
data_dir = "."

print("=== Updating roomEvent for robot position match ===")

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(".csv"):
        continue

    index = fname[:5]
    fpath = os.path.join(data_dir, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        raw_header = reader[0]
        data_rows = reader[1:]
        header = [h.strip().lower() for h in raw_header]

        try:
            ix = header.index("robot.x")
            iz = header.index("robot.z")
            ie = header.index("roomevent")
        except ValueError:
            print(f"Index {index}: ❌ Missing required columns.")
            continue

        match_found = False

        for i, row in enumerate(data_rows):
            if len(row) <= max(ix, iz, ie):
                continue

            try:
                x = float(row[ix])
                z = float(row[iz])
            except ValueError:
                continue

            dist = math.sqrt((x - target_x) ** 2 + (z - target_z) ** 2)
            if dist < threshold:
                row[ie] = "Robot entered survey room"
                print(f"Index {index}: ✅ Updated at row {i + 2}")
                match_found = True
                break

        if not match_found:
            print(f"Index {index}: ❌ No match found")
            continue

    # Write back to the same file
    with open(fpath, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(raw_header)
        writer.writerows(data_rows)

print("✅ All files processed.")