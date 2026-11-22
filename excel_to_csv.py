#!/usr/bin/env python3
"""
Excel to CSV converter script
Convert an Excel file to CSV, intelligently detecting compound-library data layout
"""

import pandas as pd
import os
import sys
from pathlib import Path

def find_data_start(df):
    """
    Find the row where the actual data begins
    """
    for idx, row in df.iterrows():
        # Look for the row containing column numbers (e.g. 1,2,3,4...)
        if pd.notna(row.iloc[1]) and str(row.iloc[1]).strip() == '1':
            return idx
    return 0

def excel_to_csv(excel_path, csv_path=None):
    """
    Convert the Excel file to CSV
    
    Args:
        excel_path (str): Path to the Excel file
        csv_path (str, optional): CSV output path; defaults to a .csv file with the same name in the same directory
    """
    try:
        # Check whether the Excel file exists
        if not os.path.exists(excel_path):
            print(f"Error: file {excel_path} does not exist")
            return False
        
        # Read the Excel file and inspect its sheets
        print(f"Reading file: {excel_path}")
        xl_file = pd.ExcelFile(excel_path)
        print(f"Found sheets: {xl_file.sheet_names}")

        # Prefer the 'Compound Information' sheet; otherwise read the first sheet
        if 'Compound Information' in xl_file.sheet_names:
            sheet_name = 'Compound Information'
        else:
            sheet_name = xl_file.sheet_names[0]

        print(f"Reading sheet: {sheet_name}")
        df_raw = pd.read_excel(excel_path, header=None, sheet_name=sheet_name)
        
        print(f"Raw data shape: {df_raw.shape}")
        
        # Find where the data starts
        data_start_row = find_data_start(df_raw)
        print(f"Data start row: {data_start_row}")
        
        if data_start_row > 0:
            # Extract the compound layout data
            data_rows = df_raw.iloc[data_start_row+1:].copy()

            # Set column names
            columns = ['Row'] + [f'Col_{i}' for i in range(1, df_raw.shape[1])]
            data_rows.columns = columns

            # Clean the data
            data_rows = data_rows.dropna(how='all')

            print(f"Cleaned data shape: {data_rows.shape}")
            print(f"Columns: {list(data_rows.columns)}")
            print("\nFirst 5 rows preview:")
            print(data_rows.head())

            df_to_save = data_rows
        else:
            # Fall back to the raw data if no clear data-start row is found
            df_to_save = df_raw
        
        # Determine the output path
        if csv_path is None:
            excel_file = Path(excel_path)
            csv_path = excel_file.parent / f"{excel_file.stem}.csv"
        
        # Save as CSV
        print(f"\nSaving to: {csv_path}")
        df_to_save.to_csv(csv_path, index=False, encoding='utf-8')
        
        print("Conversion complete!")
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    # Target file path
    excel_file_path = "data/xy01/mols.xlsx"
    
    # Run the conversion
    success = excel_to_csv(excel_file_path)
    
    if success:
        print("\n✓ Excel file successfully converted to CSV")
    else:
        print("\n✗ Conversion failed")
        sys.exit(1)