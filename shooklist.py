#!/usr/bin/env python3
import os

FOLDERS = [
    "shook",
    os.path.join("shook", "baseline"),
]

def collect_indices():
    indices = set()
    warnings = []

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            warnings.append(f"⚠️ Folder not found: {folder}")
            continue

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if not name.lower().endswith(".csv"):
                continue
            if len(name) < 5:
                warnings.append(f"⚠️ Filename too short for index (skipped): {os.path.join(folder, name)}")
                continue
            idx = name[:5]
            indices.add(idx)

    return indices, warnings

def sort_key(idx: str):
    # Sort numerically when possible, else lexicographically
    return (0, int(idx)) if idx.isdigit() else (1, idx)

def main():
    indices, warnings = collect_indices()

    # Print any warnings first
    for w in warnings:
        print(w)

    # Print results as a single comma-separated line
    sorted_indices = sorted(indices, key=sort_key)
    print(f"Found {len(sorted_indices)} indices in 'shook' and 'shook/baseline':")
    if sorted_indices:
        print(",".join(sorted_indices))
    else:
        print("")

if __name__ == "__main__":
    main()