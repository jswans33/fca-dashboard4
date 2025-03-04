"""
Example script for exporting Excel data to different formats.

This script demonstrates how to use the Excel utilities to extract data from Excel files
and save it in different formats (CSV, Excel, SQLite database).
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
    convert_excel_to_csv,
    extract_excel_with_config,
    save_excel_to_database,
)
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def extract_and_export_data(file_path, output_dir):
    """
    Extract data from an Excel file and export it to different formats.
    
    Args:
        file_path: Path to the Excel file to analyze.
        output_dir: Directory to save the output files.
        
    Returns:
        A dictionary containing the paths to the exported files.
    """
    # Resolve the file path
    file_path = resolve_path(file_path)
    
    print(f"Extracting data from Excel file: {file_path}")
    
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
    
    # Initialize a dictionary to store the paths to the exported files
    exported_files = {}
    
    # Export each sheet to different formats
    for sheet_name, df in extracted_data.items():
        print(f"\nExporting sheet: {sheet_name}")
        
        # Skip empty sheets
        if len(df) == 0:
            print(f"  Skipping empty sheet: {sheet_name}")
            continue
        
        # 1. Export to CSV
        csv_path = os.path.join(output_dir, f"{sheet_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Exported to CSV: {csv_path}")
        exported_files[f"{sheet_name}_csv"] = csv_path
        
        # 2. Export to Excel
        excel_path = os.path.join(output_dir, f"{sheet_name}.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"  Exported to Excel: {excel_path}")
        exported_files[f"{sheet_name}_excel"] = excel_path
        
        # 3. Export to SQLite database
        db_path = os.path.join(output_dir, "excel_data.db")
        connection_string = f"sqlite:///{db_path}"
        save_excel_to_database(
            df=df,
            table_name=sheet_name,
            connection_string=connection_string,
            if_exists="replace"
        )
        print(f"  Exported to SQLite database: {db_path}, table: {sheet_name}")
        exported_files["sqlite_db"] = db_path
    
    return exported_files


def verify_exports(exported_files):
    """
    Verify that the exported files were created correctly.
    
    Args:
        exported_files: Dictionary containing the paths to the exported files.
    """
    print("\nVerifying exported files:")
    
    # Verify CSV files
    for key, path in exported_files.items():
        if key.endswith("_csv"):
            if os.path.exists(path):
                # Read the CSV file to verify it contains data
                df = pd.read_csv(path)
                print(f"  CSV file {path}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"  Error: CSV file {path} not found")
        
        elif key.endswith("_excel"):
            if os.path.exists(path):
                # Read the Excel file to verify it contains data
                df = pd.read_excel(path)
                print(f"  Excel file {path}: {len(df)} rows, {len(df.columns)} columns")
            else:
                print(f"  Error: Excel file {path} not found")
    
    # Verify SQLite database
    if "sqlite_db" in exported_files:
        db_path = exported_files["sqlite_db"]
        if os.path.exists(db_path):
            # Connect to the database and list tables
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"  SQLite database {db_path} contains tables: {[table[0] for table in tables]}")
            
            # Query each table to verify it contains data
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                print(f"    Table {table_name}: {row_count} rows")
            
            conn.close()
        else:
            print(f"  Error: SQLite database {db_path} not found")


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
    
    # Extract and export data
    exported_files = extract_and_export_data(file_path, output_dir)
    
    # Verify the exported files
    verify_exports(exported_files)
    
    print("\nExport completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())