import os
import shutil
import pandas as pd

# Constants
EXCEL_FILE = "Shooter Study 2 Exit Choices .xlsx"
SHOOK_DIR = "./shook"
NOSHOOK_DIR = "./noshook"
BASELINE_FOLDER = "baseline"

# Load Excel file
try:
    df = pd.read_excel(EXCEL_FILE)
except Exception as e:
    print(f"❌ Failed to load Excel file: {e}")
    exit(1)

# Iterate through rows
for _, row in df.iterrows():
    try:
        index = str(row[0]).zfill(5)  # First column: 5-character index
        condition = int(row[1])       # Second column: CONDITION value
    except:
        continue

    if condition != 3:
        continue

    found = False
    for folder in [SHOOK_DIR, NOSHOOK_DIR]:
        for fname in os.listdir(folder):
            if fname.startswith(index) and fname.endswith(".csv"):
                src = os.path.join(folder, fname)
                dest_dir = os.path.join(folder, BASELINE_FOLDER)
                os.makedirs(dest_dir, exist_ok=True)
                dst = os.path.join(dest_dir, fname)
                shutil.move(src, dst)
                print(f"✅ Moved {fname} to {dest_dir}")
                found = True
                break
        if found:
            break
    if not found:
        print(f"⚠️ File for index {index} not found in {SHOOK_DIR} or {NOSHOOK_DIR}")