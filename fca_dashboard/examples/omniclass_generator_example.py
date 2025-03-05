"""
Example script for extracting OmniClass data from Excel files.

This script demonstrates how to use the OmniClass generator to extract data
from OmniClass Excel files and create a unified CSV file for classifier training.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.generator.omniclass import extract_omniclass_data
from fca_dashboard.utils.env_utils import get_env_var, ENV_VAR_NAME
from fca_dashboard.utils.path_util import get_root_dir, resolve_path
from fca_dashboard.utils.logging_config import get_logger


def analyze_omniclass_sample():
    """
    Extract data from a single OmniClass file as a sample.
    
    This function demonstrates how to extract data from a single OmniClass file
    and analyze its structure before processing all files.
    """
    logger = get_logger("omniclass_example")
    
    # Get the path to a sample OmniClass file
    input_dir = settings.get("generator.omniclass.input_dir", "files/omniclass_tables")
    omniclass_dir = resolve_path(input_dir)
    file_pattern = settings.get("generator.omniclass.file_pattern", "*.xlsx")
    
    logger.info(f"Looking for OmniClass files in: {omniclass_dir}")
    sample_files = list(omniclass_dir.glob(file_pattern))
    
    if not sample_files:
        logger.warning(f"No OmniClass files found in {omniclass_dir}")
        return
    
    # Use the first file as a sample
    sample_file = sample_files[0]
    logger.info(f"Analyzing sample OmniClass file: {sample_file.name}")
    print(f"Analyzing sample OmniClass file: {sample_file.name}")
    
    # Import here to avoid circular imports
    from fca_dashboard.utils.excel import (
        analyze_excel_structure,
        get_sheet_names,
        normalize_sheet_names,
    )
    
    # Analyze the Excel file structure
    analysis = analyze_excel_structure(sample_file)
    
    print(f"File type: {analysis['file_type']}")
    print(f"Sheet names: {analysis['sheet_names']}")
    
    # Find the FLAT sheet
    flat_sheet = None
    for sheet in analysis['sheet_names']:
        if 'FLAT' in sheet.upper():
            flat_sheet = sheet
            break
    
    if flat_sheet is None:
        print(f"No FLAT sheet found in {sample_file.name}")
        return
    
    print(f"Found FLAT sheet: {flat_sheet}")
    
    # Get information about the FLAT sheet
    sheet_info = analysis['sheets_info'][flat_sheet]
    print(f"Sheet shape: {sheet_info['shape']}")
    print(f"Columns: {sheet_info['columns'][:10]}...")  # Show first 10 columns
    print(f"Header row: {sheet_info['header_row']}")
    
    # Extract data from the FLAT sheet
    from fca_dashboard.utils.excel import extract_excel_with_config
    
    # The sheet name might be normalized in the extraction process
    # So we need to use the normalized sheet name
    config = {
        flat_sheet: {
            "header_row": 0,  # Assume header is in the first row
            "drop_empty_rows": True,
            "clean_column_names": True,
            "strip_whitespace": True,
        }
    }
    
    extracted_data = extract_excel_with_config(sample_file, config)
    
    # Find the sheet in the extracted data
    # The sheet name might have been normalized
    if flat_sheet in extracted_data:
        df = extracted_data[flat_sheet]
    else:
        # Try to find a sheet with a similar name
        normalized_sheet_names = normalize_sheet_names(sample_file)
        normalized_flat_sheet = None
        
        for original, normalized in normalized_sheet_names.items():
            if original == flat_sheet:
                normalized_flat_sheet = normalized
                break
        
        if normalized_flat_sheet and normalized_flat_sheet in extracted_data:
            df = extracted_data[normalized_flat_sheet]
            logger.info(f"Using normalized sheet name: {normalized_flat_sheet}")
        else:
            # Just use the first sheet as a fallback
            sheet_name = list(extracted_data.keys())[0]
            df = extracted_data[sheet_name]
            logger.warning(f"Could not find sheet '{flat_sheet}', using '{sheet_name}' instead")
    
    print(f"\nExtracted {len(df)} rows with {len(df.columns)} columns")
    print(f"First few rows:")
    print(df.head(3))
    
    # Show column statistics
    print("\nColumn statistics:")
    for col in df.columns[:5]:  # Show stats for first 5 columns
        print(f"Column '{col}':")
        print(f"  - Unique values: {df[col].nunique()}")
        print(f"  - Null values: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")
        if df[col].dtype == 'object':
            print(f"  - Sample values: {df[col].dropna().sample(min(3, df[col].nunique())).tolist()}")
        else:
            print(f"  - Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean()}")
    
    return df


def extract_all_omniclass_data():
    """
    Extract data from all OmniClass files and save to a CSV file.
    
    This function demonstrates how to use the OmniClass generator to extract data
    from all OmniClass Excel files and create a unified CSV file.
    """
    logger = get_logger("omniclass_example")
    
    logger.info("Extracting data from all OmniClass files...")
    print("\nExtracting data from all OmniClass files...")
    
    # Get environment-specific settings
    env = get_env_var(ENV_VAR_NAME, "development")
    logger.info(f"Current environment: {env}")
    
    # Extract OmniClass data using the generator
    input_dir = settings.get("generator.omniclass.input_dir", "files/omniclass_tables")
    output_file = settings.get("generator.omniclass.output_file", "fca_dashboard/generator/ingest/omniclass.csv")
    file_pattern = settings.get("generator.omniclass.file_pattern", "*.xlsx")
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File pattern: {file_pattern}")
    
    combined_df = extract_omniclass_data(
        input_dir=input_dir,
        output_file=output_file,
        file_pattern=file_pattern
    )
    
    # Show summary of the extracted data
    logger.info(f"Extracted {len(combined_df)} total rows with {len(combined_df.columns)} columns")
    print(f"\nExtracted {len(combined_df)} total rows with {len(combined_df.columns)} columns")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    # Show table distribution
    table_counts = combined_df['table_number'].value_counts()
    print("\nDistribution by OmniClass table:")
    for table, count in table_counts.items():
        print(f"  - Table {table}: {count} rows")
        logger.info(f"Table {table}: {count} rows")
    
    # Show a sample of the data
    print("\nSample data:")
    print(combined_df.sample(min(5, len(combined_df))).to_string())
    
    return combined_df


def main():
    """Main function."""
    logger = get_logger("omniclass_example")
    
    # Get environment information
    env = get_env_var(ENV_VAR_NAME, "development")
    root_dir = get_root_dir()
    
    logger.info(f"Starting OmniClass Generator Example in {env} environment")
    logger.info(f"Project root directory: {root_dir}")
    
    print("OmniClass Generator Example")
    print("==========================")
    print(f"Environment: {env}")
    print(f"Project root: {root_dir}")
    
    # Analyze a sample OmniClass file
    print("\nStep 1: Analyze a sample OmniClass file")
    logger.info("Step 1: Analyzing a sample OmniClass file")
    sample_df = analyze_omniclass_sample()
    
    # Extract data from all OmniClass files
    print("\nStep 2: Extract data from all OmniClass files")
    logger.info("Step 2: Extracting data from all OmniClass files")
    combined_df = extract_all_omniclass_data()
    
    # Show the path to the output file
    output_path = settings.get("generator.omniclass.output_file", "fca_dashboard/generator/ingest/omniclass.csv")
    output_file = resolve_path(output_path)
    logger.info(f"Output file saved to: {output_file}")
    print(f"\nOutput file: {output_file}")
    
    logger.info("OmniClass Generator Example completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())