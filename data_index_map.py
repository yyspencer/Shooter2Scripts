import os
import pandas as pd

# === Constants ===
EXCEL_FILE = "Shooter 2 Data.xlsx"
SEARCH_DIRS = [
    os.path.join("shook"),                         # ./shook/
    os.path.join("shook", "baseline"),             # ./shook/baseline/
    os.path.join("noshook"),                       # ./noshook/
    os.path.join("noshook", "baseline")            # ./noshook/baseline/
]

# === Load Excel IDs ===
try:
    df = pd.read_excel(EXCEL_FILE)
    indices = df.iloc[:, 0].astype(str).str.zfill(5).tolist()  # Ensure 5-char string
except Exception as e:
    print(f"❌ Failed to load Excel file: {e}")
    exit(1)

# === Search Logic ===
for idx in indices:
    found = False
    for directory in SEARCH_DIRS:
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if fname.startswith(idx) and fname.endswith(".csv"):
                print(f"✅ Found file for index {idx} in {directory}")
                found = True
                break
        if found:
            break
    if not found:
        print(f"❌ Missing file for index {idx}")