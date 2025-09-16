import pandas as pd
import shutil

# File names
source_file = "Shooter 2 Data.xlsx"
exit_choices_file = "Shooter Study 2 Exit Choices.xlsx"
proceed_file = "Shooter 2 Data Proceed.xlsx"
keyword = "followed robot"

# 1. Duplicate the source Excel file
shutil.copyfile(source_file, proceed_file)

# 2. Load both Excel files
df_proceed = pd.read_excel(proceed_file, engine='openpyxl')
df_exit = pd.read_excel(exit_choices_file, engine='openpyxl')

# 3. Find the column in exit_choices_file whose header contains the keyword
exit_headers = [str(col).strip().lower() for col in df_exit.columns]
matching_cols = [i for i, col in enumerate(exit_headers) if keyword in col]

print(exit_headers)

if not matching_cols:
    raise ValueError(f"No column header contains the keyword '{keyword}'")
exit_col_idx = matching_cols[0]  # use the first/only match
exit_col_name = df_exit.columns[exit_col_idx]

# 4. Build a dictionary for fast value lookup by ID
exit_val_dict = dict(zip(df_exit.iloc[:, 0], df_exit.iloc[:, exit_col_idx]))

missing_ids = []

# 5. Fill in the "Exit Choice" column (column 3, index 2) for matching IDs
for idx, row in df_proceed.iterrows():
    participant_id = row.iloc[0]
    if participant_id in exit_val_dict:
        df_proceed.iat[idx, 2] = exit_val_dict[participant_id]
    else:
        df_proceed.iat[idx, 2] = ""  # Leave cell blank
        missing_ids.append(participant_id)

# 6. Save the updated DataFrame to Excel
df_proceed.to_excel(proceed_file, index=False, engine='openpyxl')

# 7. Print any missing IDs
if missing_ids:
    print("These participant IDs were not found in the Exit Choices file:")
    for mid in missing_ids:
        print(mid)
else:
    print("All IDs matched successfully!")