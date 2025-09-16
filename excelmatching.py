import pandas as pd
import os

# Step 1: Load Excel file and extract the first column
xlsx_path = "Shooter Study 2 Exit Choices .xlsx"
df = pd.read_excel(xlsx_path, sheet_name=0)

# Step 2: Extract first column (excluding header), convert to string and take first 5 chars
indices = df.iloc[:, 0].dropna().astype(str).str[:5].unique()

# Step 3: Collect all .csv files in "survey/" folder
survey_dir = "survey"
csv_files = [f for f in os.listdir(survey_dir) if f.endswith(".csv")]
csv_prefixes = {f[:5] for f in csv_files}

# Step 4: Check and print results
print("=== Index Check ===")
for idx in indices:
    status = "✅ Found" if idx in csv_prefixes else "❌ Not Found"
    print(f"Index {idx}: {status}")