"""
Simple example script for extracting OmniClass data from Excel files.

This script demonstrates how to use the OmniClass generator with hardcoded paths
to extract data from OmniClass Excel files and create a unified CSV file.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.generator.omniclass import extract_omniclass_data


def main():
    """Main function."""
    print("OmniClass Simple Example")
    print("=======================")
    
    # Define input and output paths
    input_dir = "files/omniclass_tables"
    output_file = "fca_dashboard/generator/ingest/omniclass.csv"
    
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    
    # Extract OmniClass data
    try:
        print("\nExtracting OmniClass data...")
        combined_df = extract_omniclass_data(
            input_dir=input_dir,
            output_file=output_file,
            file_pattern="*.xlsx"
        )
        
        # Show summary of the extracted data
        print(f"\nExtracted {len(combined_df)} total rows with {len(combined_df.columns)} columns")
        print(f"Columns: {combined_df.columns.tolist()}")
        
        # Show table distribution
        if 'table_number' in combined_df.columns:
            table_counts = combined_df['table_number'].value_counts()
            print("\nDistribution by OmniClass table:")
            for table, count in table_counts.items():
                print(f"  - Table {table}: {count} rows")
        
        # Show a sample of the data
        print("\nSample data:")
        print(combined_df.sample(min(5, len(combined_df))).to_string())
        
        print(f"\nOutput file saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())