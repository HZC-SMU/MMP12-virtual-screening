#!/usr/bin/env python3
"""
Compound information extraction script
Extracts the CAS Number, Product Name, and SMILES columns
"""

import pandas as pd
import os
import sys
from pathlib import Path

def extract_compound_info(excel_path, csv_path=None):
    """
    Extract the specified compound-information columns from an Excel file

    Args:
        excel_path (str): Path to the Excel file
        csv_path (str, optional): CSV output path
    """
    try:
        # Columns to extract
        target_columns = ['CAS Number', 'Product Name', 'SMILES']

        # Check whether the Excel file exists
        if not os.path.exists(excel_path):
            print(f"Error: file {excel_path} does not exist")
            return False

        # Read the Excel file and inspect its sheets
        print(f"Reading file: {excel_path}")
        xl_file = pd.ExcelFile(excel_path)
        print(f"Found sheets: {xl_file.sheet_names}")

        # Prefer the 'Compound Information' sheet
        if 'Compound Information' in xl_file.sheet_names:
            sheet_name = 'Compound Information'
        else:
            sheet_name = xl_file.sheet_names[0]

        print(f"Reading sheet: {sheet_name}")
        # Use the first row as the header
        df_raw = pd.read_excel(excel_path, header=0, sheet_name=sheet_name)

        print(f"Raw data shape: {df_raw.shape}")
        print(f"All columns: {list(df_raw.columns)}")

        # Check whether the target columns exist
        missing_columns = [col for col in target_columns if col not in df_raw.columns]
        if missing_columns:
            print(f"Error: the following required columns are missing: {missing_columns}")
            print("Available columns:")
            for i, col in enumerate(df_raw.columns):
                print(f"  {i}: {col}")
            return False

        print(f"Extracting target columns: {target_columns}")

        # Extract the specified columns
        df_extracted = df_raw[target_columns].copy()

        # Clean the data - drop rows where every column is empty
        df_clean = df_extracted.dropna(how='all')

        # Drop rows where both CAS Number and SMILES are empty (likely invalid compounds)
        df_clean = df_clean.dropna(subset=['CAS Number', 'SMILES'], how='all')

        print(f"Rows before cleaning: {len(df_extracted)}")
        print(f"Rows after cleaning: {len(df_clean)}")
        print(f"Final columns: {list(df_clean.columns)}")

        # Show statistics
        print(f"\nData statistics:")
        print(f"- Records with CAS Number: {df_clean['CAS Number'].notna().sum()}")
        print(f"- Records with Product Name: {df_clean['Product Name'].notna().sum()}")
        print(f"- Records with SMILES: {df_clean['SMILES'].notna().sum()}")

        print("\nFirst 5 rows preview:")
        print(df_clean.head())

        # Determine the output path
        if csv_path is None:
            excel_file = Path(excel_path)
            csv_path = excel_file.parent / f"{excel_file.stem}_compound_info.csv"

        # Save as CSV
        print(f"\nSaving to: {csv_path}")
        df_clean.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"Conversion complete! Saved {len(df_clean)} compound records")
        return True

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

if __name__ == "__main__":
    # Target file path
    excel_file_path = "data/xy01/mols.xlsx"

    # Run the conversion
    success = extract_compound_info(excel_file_path)

    if success:
        print("\n✓ Compound information successfully extracted to CSV")
    else:
        print("\n✗ Extraction failed")
        sys.exit(1)