import pandas as pd
import shutil
import os

# Filenames
source_file = "Shooter 2 Data.xlsx"
exit_choices_file = "Shooter Study 2 Exit Choices .xlsx"
proceed_file = "Shooter 2 Data Proceed.xlsx"

# Step 1: Duplicate the source Excel file
shutil.copyfile(source_file, proceed_file)

# Step 2: Load both Excel files
df_proceed = pd.read_excel(proceed_file, engine='openpyxl')
df_exit = pd.read_excel(exit_choices_file, engine='openpyxl')

# Build a dictionary for fast CONDITION lookup by ID
exit_dict = dict(zip(df_exit.iloc[:, 0], df_exit.iloc[:, 1]))

missing_ids = []

# Step 3: Fill in the CONDITION column for matching IDs
for idx, row in df_proceed.iterrows():
    participant_id = row.iloc[0]
    if participant_id in exit_dict:
        df_proceed.iat[idx, 1] = exit_dict[participant_id]
    else:
        df_proceed.iat[idx, 1] = ""  # Leave cell blank
        missing_ids.append(participant_id)

# Step 4: Save the updated DataFrame to Excel
df_proceed.to_excel(proceed_file, index=False, engine='openpyxl')

# Step 5: Print any missing IDs
if missing_ids:
    print("These participant IDs were not found in the Exit Choices file:")
    for mid in missing_ids:
        print(mid)
else:
    print("All IDs matched successfully!")