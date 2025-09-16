import os
import pandas as pd

def main():
    xlsx_path = "Shooter Study 2 Exit Choices .xlsx"
    survey_dir = "survey"

    if not os.path.exists(xlsx_path):
        print(f"Error: Excel file '{xlsx_path}' not found.")
        return
    if not os.path.isdir(survey_dir):
        print(f"Error: Folder '{survey_dir}' not found.")
        return

    # Step 1: Load indices from Excel
    df = pd.read_excel(xlsx_path, sheet_name=0)
    indices = df.iloc[:, 0].dropna().astype(str).str[:5].unique()

    # Step 2: List all .csv files in the survey folder
    csv_files = [f for f in os.listdir(survey_dir) if f.endswith(".csv")]

    print("=== Matched Files ===")
    for idx in indices:
        matched = [f for f in csv_files if f.startswith(idx)]
        if not matched:
            print(f"Index {idx}: ❌ No matching .csv file")
            continue

        for fname in matched:
            label = ""
            if "Emergi" in fname:
                label = "Emergi"
            elif "Andi" in fname:
                label = "Andi"
            else:
                label = "None"

            last_char = fname[-5] if fname[-4:] == ".csv" else fname[-1]  # e.g., "12345Andi1.csv" → "1"
            #print(f"Index {idx}: {fname} → Label = {label}, LastChar = {last_char}")
            print(f"{last_char}")
            

if __name__ == "__main__":
    main()