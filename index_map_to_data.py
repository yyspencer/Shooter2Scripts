import os
import pandas as pd

# Constants
EXCEL_FILE = "Shooter 2 Data.xlsx"
FOLDERS = ["shook", os.path.join("shook", "baseline"),
           "noshook", os.path.join("noshook", "baseline")]

def extract_indices_from_folders():
    indices = set()
    for folder in FOLDERS:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.endswith(".csv") and len(fname) >= 5:
                indices.add(fname[:5])
    return indices

def extract_ids_from_excel():
    try:
        df = pd.read_excel(EXCEL_FILE)
        return set(str(x).zfill(5) for x in df.iloc[:, 0] if pd.notna(x))
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return set()

def main():
    excel_ids = extract_ids_from_excel()
    folder_indices = extract_indices_from_folders()

    both = sorted(excel_ids & folder_indices)
    missing_in_folders = sorted(excel_ids - folder_indices)
    missing_in_excel = sorted(folder_indices - excel_ids)

    print(f"\n✅ Found in both ({len(both)}):")
    print(", ".join(both))

    print(f"\n❌ Present in Excel but missing in folders ({len(missing_in_folders)}):")
    print(", ".join(missing_in_folders))

    print(f"\n❌ Present in folders but missing in Excel ({len(missing_in_excel)}):")
    print(", ".join(missing_in_excel))

if __name__ == "__main__":
    main()