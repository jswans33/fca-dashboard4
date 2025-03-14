"""
Example script for validating Excel data.

This script demonstrates how to use the Excel validation utilities to validate data
from Excel files, including checking for missing values, duplicate rows, value ranges,
and data types.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    extract_excel_with_config,
    validate_dataframe,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_validate_data(file_path, output_dir):
    """
    Extract data from an Excel file and validate it.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the validation reports.
        
    Returns:
        A dictionary containing the validation results.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting and validating data from Excel file: {file_path}")
    
    # First, analyze the Excel file structure to understand its contents
    analysis = analyze_excel_structure(file_path)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Get extraction configuration from settings
    extraction_config = settings.get("excel_utils.extraction", {})
    
    # Convert sheet names to match the format in the settings
    # The settings use lowercase with underscores, but the Excel file uses spaces and title case
    config = {}
    
    # Add default settings
    if "default" in extraction_config:
        config["default"] = extraction_config["default"]
    
    # Add sheet-specific settings
    for sheet_name in analysis['sheet_names']:
        # Convert sheet name to the format used in settings (lowercase with underscores)
        settings_key = sheet_name.lower().replace(" ", "_")
        
        # If there are settings for this sheet, add them to the config
        if settings_key in extraction_config:
            config[sheet_name] = extraction_config[settings_key]
    
    print(f"Using extraction configuration from settings: {config}")
    
    # Extract data from the Excel file using our configuration
    extracted_data = extract_excel_with_config(file_path, config)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a dictionary to store the validation results
    validation_results = {}
    
    # Validate each sheet
    for sheet_name, df in extracted_data.items():
        print(f"\nValidating sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # Create a validation configuration based on the sheet
        validation_config = create_validation_config(sheet_name, df)
        
        # Validate the DataFrame
        results = validate_dataframe(df, validation_config)
        validation_results[sheet_name] = results
        
        # Save the validation report
        save_validation_report(sheet_name, df, results, output_dir)
    
    return validation_results


def create_validation_config(sheet_name, df):
    """
    Create a validation configuration based on the sheet name and DataFrame.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame to validate.
        
    Returns:
        A dictionary containing the validation configuration.
    """
    # Get validation configuration from settings
    validation_settings = settings.get("excel_utils.validation", {})
    
    # Initialize the validation configuration with default settings
    validation_config = {}
    
    # Add default settings
    if "default" in validation_settings:
        default_settings = validation_settings["default"]
        
        # Add missing values check
        if "missing_values" in default_settings:
            missing_values_settings = default_settings["missing_values"]
            validation_config["missing_values"] = {
                "columns": missing_values_settings.get("columns") or list(df.columns),
                "threshold": missing_values_settings.get("threshold", 0.5)
            }
        
        # Add duplicate rows check
        if "duplicate_rows" in default_settings:
            duplicate_rows_settings = default_settings["duplicate_rows"]
            validation_config["duplicate_rows"] = {
                "subset": duplicate_rows_settings.get("subset")
            }
        
        # Add data types check
        if "data_types" in default_settings:
            data_types_settings = default_settings["data_types"]
            
            # Create type specifications based on settings
            type_specs = {}
            
            # Add date columns
            for col in data_types_settings.get("date_columns", []):
                if col in df.columns:
                    type_specs[col] = "date"
            
            # Add numeric columns
            for col in data_types_settings.get("numeric_columns", []):
                if col in df.columns:
                    type_specs[col] = "float"
            
            # Add string columns
            for col in data_types_settings.get("string_columns", []):
                if col in df.columns:
                    type_specs[col] = "str"
            
            # Add boolean columns
            for col in data_types_settings.get("boolean_columns", []):
                if col in df.columns:
                    type_specs[col] = "bool"
            
            if type_specs:
                validation_config["data_types"] = type_specs
    
    # Add sheet-specific validation
    # Convert sheet name to the format used in settings (lowercase with underscores)
    settings_key = sheet_name.lower()
    
    if settings_key in validation_settings:
        sheet_settings = validation_settings[settings_key]
        
        # Add value ranges check
        if "value_ranges" in sheet_settings:
            validation_config["value_ranges"] = sheet_settings["value_ranges"]
        
        # Add required columns check
        if "required_columns" in sheet_settings:
            validation_config["required_columns"] = sheet_settings["required_columns"]
    
    # If no validation config was created from settings, create a basic one
    if not validation_config:
        # Add missing values check for all sheets
        validation_config["missing_values"] = {
            "columns": list(df.columns),
            "threshold": 0.5  # Allow up to 50% missing values
        }
        
        # Add duplicate rows check for all sheets
        validation_config["duplicate_rows"] = {
            "subset": None  # Check all columns for duplicates
        }
    
    return validation_config


def save_validation_report(sheet_name, df, results, output_dir):
    """
    Save a validation report for a sheet.
    
    Args:
        sheet_name: Name of the sheet.
        df: DataFrame that was validated.
        results: Validation results.
        output_dir: Directory to save the report.
    """
    # Create a report file
    report_path = os.path.join(output_dir, f"{sheet_name}_validation_report.txt")
    
    with open(report_path, "w") as f:
        f.write(f"Validation Report for Sheet: {sheet_name}\n")
        f.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")
        
        # Write missing values report
        if "missing_values" in results:
            f.write("Missing Values Report:\n")
            f.write("-" * 50 + "\n")
            
            missing_values = results["missing_values"]
            for col, pct in missing_values.items():
                f.write(f"  {col}: {pct * 100:.2f}% missing\n")
            
            f.write("\n")
        
        # Write duplicate rows report
        if "duplicate_rows" in results:
            f.write("Duplicate Rows Report:\n")
            f.write("-" * 50 + "\n")
            
            duplicate_rows = results["duplicate_rows"]
            f.write(f"  Duplicate rows: {duplicate_rows['duplicate_count']}\n")
            
            if duplicate_rows["duplicate_count"] > 0:
                f.write(f"  Duplicate indices: {duplicate_rows['duplicate_indices'][:10]}")
                if len(duplicate_rows["duplicate_indices"]) > 10:
                    f.write(f" ... and {len(duplicate_rows['duplicate_indices']) - 10} more")
                f.write("\n")
            
            f.write("\n")
        
        # Write value ranges report
        if "value_ranges" in results:
            f.write("Value Ranges Report:\n")
            f.write("-" * 50 + "\n")
            
            value_ranges = results["value_ranges"]
            for col, res in value_ranges.items():
                f.write(f"  {col}:\n")
                f.write(f"    Below minimum: {res['below_min']}\n")
                f.write(f"    Above maximum: {res['above_max']}\n")
                f.write(f"    Total outside range: {res['total_outside_range']}\n")
            
            f.write("\n")
        
        # Write data types report
        if "data_types" in results:
            f.write("Data Types Report:\n")
            f.write("-" * 50 + "\n")
            
            data_types = results["data_types"]
            for col, res in data_types.items():
                f.write(f"  {col}:\n")
                f.write(f"    Expected type: {res['expected_type']}\n")
                f.write(f"    Current type: {res['current_type']}\n")
                f.write(f"    Error count: {res['error_count']}\n")
            
            f.write("\n")
    
    print(f"  Saved validation report to: {report_path}")


def main():
    """Main function."""
    # Get the file path from command line arguments or use a default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use the example file
        file_path = "C:/Repos/fca-dashboard4/uploads/Medtronics - Asset Log Uploader.xlsx"
    
    # Get the output directory
    output_dir = os.path.join(get_root_dir(), "examples", "output")
    
    # Extract and validate data
    validation_results = extract_and_validate_data(file_path, output_dir)
    
    print("\nValidation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())