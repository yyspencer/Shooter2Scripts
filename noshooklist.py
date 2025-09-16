#!/usr/bin/env python3
import os

FOLDERS = [
    "noshook",
    os.path.join("noshook", "baseline"),
]

def collect_indices():
    indices = set()
    warnings = []

    for folder in FOLDERS:
        if not os.path.isdir(folder):
            warnings.append(f"⚠️ Folder not found: {folder}")
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

    # Print any warnings to stderr-like prefix
    for w in warnings:
        print(w)

    # Print results
    sorted_indices = sorted(indices, key=sort_key)
    print(f"Found {len(sorted_indices)} indices in 'noshook' and 'noshook/baseline':")
    for idx in sorted_indices:
        print(idx)

if __name__ == "__main__":
    main()