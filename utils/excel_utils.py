import pandas as pd
import os

def append_to_excel(file_path, data_dict):
    """
    Append a dictionary of data as a new row to an Excel file.
    If the file doesn't exist, create it.
    """
    df_new = pd.DataFrame([data_dict])
    if not os.path.exists(file_path):
        df_new.to_excel(file_path, index=False)
    else:
        df_existing = pd.read_excel(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file_path, index=False)
