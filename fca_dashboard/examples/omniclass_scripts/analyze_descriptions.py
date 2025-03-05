#!/usr/bin/env python
"""
Script to analyze OmniClass description files.

This script analyzes OmniClass description files to check how many descriptions
were generated and how many NaN values remain.
"""
import os
import sys
import pandas as pd
import glob
from pathlib import Path

def analyze_file(file_path):
    """Analyze a single file."""
    print(f"\nAnalyzing {file_path}...")
    
    # Load the file
    df = pd.read_csv(file_path)
    
    # Get total rows
    total_rows = len(df)
    print(f"Total rows: {total_rows}")
    
    # Check if Description column exists
    if 'Description' not in df.columns:
        print("No Description column found!")
        return
    
    # Count NaN values
    nan_count = df['Description'].isna().sum()
    empty_count = (df['Description'] == '').sum()
    
    # Count non-empty descriptions
    filled_count = total_rows - nan_count - empty_count
    
    # Print results
    print(f"Descriptions generated: {filled_count} ({filled_count/total_rows:.1%})")
    print(f"Empty descriptions: {empty_count} ({empty_count/total_rows:.1%})")
    print(f"NaN values: {nan_count} ({nan_count/total_rows:.1%})")
    
    # Print sample descriptions
    if filled_count > 0:
        print("\nSample descriptions:")
        samples = df[df['Description'].notna() & (df['Description'] != '')].sample(min(3, filled_count))
        for _, row in samples.iterrows():
            print(f"- {row['OmniClass_Code']}: {row['OmniClass_Title']}")
            print(f"  Description: {row['Description']}")

def main():
    """Main function."""
    # Define the output directory
    output_dir = "fca_dashboard/generator/ingest/output"
    
    # Check if directory exists
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist!")
        return 1
    
    # Get all CSV files in the output directory
    csv_files = glob.glob(f"{output_dir}/*.csv")
    
    if not csv_files:
        print(f"No CSV files found in {output_dir}!")
        return 1
    
    print(f"Found {len(csv_files)} CSV files in {output_dir}:")
    for file in csv_files:
        print(f"- {os.path.basename(file)}")
    
    # Analyze each file
    for file in csv_files:
        analyze_file(file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())