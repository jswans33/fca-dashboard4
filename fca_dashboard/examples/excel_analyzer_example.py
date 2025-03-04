"""
Example script for analyzing Excel files using the robust Excel utilities.

This script demonstrates how to use the advanced Excel utilities to extract data from Excel files
with complex structures, such as headers in non-standard positions, multiple sheets, etc.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    extract_excel_with_config,
    normalize_sheet_names,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def analyze_and_extract_medtronics_file(file_path):
    """
    Analyze and extract data from the Medtronics Asset Log Uploader Excel file using the robust Excel utilities.
    
    Args:
        file_path: Path to the Excel file to analyze.
        
    Returns:
        A dictionary containing the extracted data for each sheet.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Analyzing Medtronics Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get normalized sheet names
    sheet_name_mapping = normalize_sheet_names(file_path)
    print(f"Normalized sheet names: {sheet_name_mapping}")
    
    # Create a configuration for the Medtronics Excel file
    # This configuration is based on our analysis of the file structure
    config = {
        "default": {
            "header_row": None,  # Auto-detect for most sheets
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Asset Data": {
            "header_row": 6,  # We know the header starts at row 7 (index 6)
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "EQ IDs": {
            "header_row": 0,  # Header is in the first row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Cobie": {
            "header_row": 1,  # Header is in the second row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        },
        "Dropdowns": {
            "header_row": 0,  # Header is in the first row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        }
    }
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Print information about each extracted sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nProcessed sheet: {sheet_name}")
        print(f"Extracted {len(df)} rows with {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        print(f"First few rows:")
        print(df.head(3))
    
    return extracted_data


def save_config_example():
    """
    Example of how to save a configuration to a JSON file.
    """
    # Create a sample configuration
    config = {
        "default": {
            "header_row": None,  # Auto-detect
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
            "convert_dtypes": True,
            "date_columns": ["Date", "Scheduled Delivery Date", "Actual On-Site Date"],
            "numeric_columns": ["Motor HP", "Size"],
            "boolean_columns": ["O&M Received", "Attic Stock"],
        },
        "Asset Data": {
            "header_row": 6,  # We know the header starts at row 7 (index 6)
        },
        "Equipment Log": {
            "required_columns": ["Equipment Name", "Equipment Tag ID"],
        }
    }
    
    # Save the configuration to a JSON file
    config_path = Path("excel_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved configuration to {config_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Analyze and extract data from the Medtronics Excel file
    extracted_data = analyze_and_extract_medtronics_file(file_path)
    
    # Print a summary of the extracted data
    print("\nExtraction Summary:")
    for sheet_name, df in extracted_data.items():
        print(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
    
    # Uncomment to save a sample configuration
    # save_config_example()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())