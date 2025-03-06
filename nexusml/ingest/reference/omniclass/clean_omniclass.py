#!/usr/bin/env python
"""
Clean OmniClass Data

This script cleans the OmniClass CSV file by filling in null values.
"""

from pathlib import Path

import pandas as pd


def clean_omniclass_data():
    """Clean the OmniClass CSV file."""
    print("Cleaning OmniClass data...")

    # Path to the OmniClass CSV file (in the same directory as this script)
    file_path = Path(__file__).parent / "omniclass.csv"

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Print original stats
        print(f"\nOriginal data stats:")
        print(f"Total rows: {len(df)}")
        print(f"Null values in OmniClass_Code: {df['OmniClass_Code'].isna().sum()}")
        print(f"Null values in OmniClass_Title: {df['OmniClass_Title'].isna().sum()}")
        print(f"Null values in Description: {df['Description'].isna().sum()}")

        # Fill null values
        df["OmniClass_Title"] = df["OmniClass_Title"].fillna("Untitled")
        df["Description"] = df["Description"].fillna("No description available")

        # Print cleaned stats
        print(f"\nCleaned data stats:")
        print(f"Total rows: {len(df)}")
        print(f"Null values in OmniClass_Code: {df['OmniClass_Code'].isna().sum()}")
        print(f"Null values in OmniClass_Title: {df['OmniClass_Title'].isna().sum()}")
        print(f"Null values in Description: {df['Description'].isna().sum()}")

        # Save the cleaned data
        df.to_csv(file_path, index=False)
        print(f"\nCleaned data saved to {file_path}")

    except Exception as e:
        print(f"Error cleaning OmniClass data: {e}")


if __name__ == "__main__":
    clean_omniclass_data()
