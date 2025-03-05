"""
Example script for extracting OmniClass data from Excel files.

This script demonstrates how to use the OmniClass generator to extract data
from OmniClass Excel files and create a unified CSV file for classifier training
that is compliant with the unified training data plan.
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
    It also shows how to map the headers to the unified training data plan format.
    """
    logger = get_logger("omniclass_example")
    
    # Define the unified headers according to the training data plan
    unified_headers = {
        'OmniClass_Code': 'OmniClass code (e.g., 23-33 10 00)',
        'OmniClass_Title': 'OmniClass title (e.g., Air Distribution Systems)',
        'Description': 'Plain-English description of equipment/system'
    }
    
    print("Unified Training Data Plan Headers:")
    for header, description in unified_headers.items():
        print(f"  - {header}: {description}")
    print()
    
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
    
    # Get column mapping from settings
    column_mapping = settings.get("generator.omniclass.column_mapping", {
        'Number': 'OmniClass_Code',
        'Title': 'OmniClass_Title',
        'Definition': 'Description'
    })
    
    print("\nColumn Mapping for Unified Training Data Plan:")
    for original, mapped in column_mapping.items():
        print(f"  - {original} -> {mapped}")
    
    # Apply the column mapping to standardize column names
    from fca_dashboard.utils.data_cleaning_utils import standardize_column_names
    standardized_df = standardize_column_names(df, column_mapping=column_mapping)
    
    # Show the standardized columns
    print("\nStandardized Columns:")
    print(standardized_df.columns.tolist())
    
    # Show a sample of the standardized data
    print("\nSample of Standardized Data:")
    print(standardized_df.head(3))
    
    return standardized_df


def extract_all_omniclass_data():
    """
    Extract data from all OmniClass files and save to a CSV file.
    
    This function demonstrates how to use the OmniClass generator to extract data
    from all OmniClass Excel files and create a unified CSV file that is compliant
    with the unified training data plan.
    """
    logger = get_logger("omniclass_example")
    
    # Define the unified headers according to the training data plan
    unified_headers = {
        'OmniClass_Code': 'OmniClass code (e.g., 23-33 10 00)',
        'OmniClass_Title': 'OmniClass title (e.g., Air Distribution Systems)',
        'Description': 'Plain-English description of equipment/system'
    }
    
    print("\nUnified Training Data Plan Headers:")
    for header, description in unified_headers.items():
        print(f"  - {header}: {description}")
    
    logger.info("Extracting data from all OmniClass files...")
    print("\nExtracting data from all OmniClass files...")
    
    # Get environment-specific settings
    env = get_env_var(ENV_VAR_NAME, "development")
    logger.info(f"Current environment: {env}")
    
    # Extract OmniClass data using the generator
    input_dir = settings.get("generator.omniclass.input_dir", "files/omniclass_tables")
    output_file = settings.get("generator.omniclass.output_file", "fca_dashboard/generator/ingest/unified_training_data.csv")
    file_pattern = settings.get("generator.omniclass.file_pattern", "*.xlsx")
    
    # Get column mapping from settings
    column_mapping = settings.get("generator.omniclass.column_mapping", {
        'Number': 'OmniClass_Code',
        'Title': 'OmniClass_Title',
        'Definition': 'Description'
    })
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"File pattern: {file_pattern}")
    logger.info(f"Column mapping: {column_mapping}")
    
    print(f"Column Mapping for Unified Training Data Plan:")
    for original, mapped in column_mapping.items():
        print(f"  - {original} -> {mapped}")
    
    # Extract data but don't save to the original output file
    combined_df = extract_omniclass_data(
        input_dir=input_dir,
        output_file=None,  # Don't save the original output
        file_pattern=file_pattern
    )
    
    # Show summary of the extracted data
    logger.info(f"Extracted {len(combined_df)} total rows with {len(combined_df.columns)} columns")
    print(f"\nExtracted {len(combined_df)} total rows with {len(combined_df.columns)} columns")
    print(f"Columns: {combined_df.columns.tolist()}")
    
    # Verify that the required columns from the unified training data plan are present
    missing_columns = [col for col in unified_headers.keys() if col not in combined_df.columns]
    if missing_columns:
        logger.warning(f"Missing required columns from unified training data plan: {missing_columns}")
        print(f"\nWARNING: Missing required columns from unified training data plan: {missing_columns}")
    else:
        logger.info("All required columns from unified training data plan are present")
        print("\nAll required columns from unified training data plan are present")
    
    # Show table distribution
    table_counts = combined_df['table_number'].value_counts()
    print("\nDistribution by OmniClass table:")
    for table, count in table_counts.items():
        print(f"  - Table {table}: {count} rows")
        logger.info(f"Table {table}: {count} rows")
    
    # Show a sample of the data
    print("\nSample data:")
    print(combined_df.sample(min(5, len(combined_df))).to_string())
    
    # Save the data in the unified training data format
    unified_output_path = resolve_path(output_file)
    
    # Create a new DataFrame with the unified structure
    # First, identify which columns might contain OmniClass codes, titles, and descriptions
    # Look for columns that might contain OmniClass codes (typically formatted like XX-XX XX XX)
    code_columns = [col for col in combined_df.columns if isinstance(col, str) and
                   (col.count('-') > 0 or col.count(' ') > 0) and
                   any(c.isdigit() for c in col)]
    
    # Look for columns that might contain titles (typically text without numbers)
    title_columns = [col for col in combined_df.columns if isinstance(col, str) and
                    col not in code_columns and
                    col not in ['source_file', 'table_number'] and
                    not all(c.isdigit() for c in col if c.isalnum())]
    
    # Create a new DataFrame with the unified structure
    unified_df = pd.DataFrame(columns=list(unified_headers.keys()))
    
    # Process each table separately
    for table in combined_df['table_number'].unique():
        table_df = combined_df[combined_df['table_number'] == table]
        
        # Find the code and title columns for this table
        table_code_col = None
        table_title_col = None
        
        # First, try to find columns that start with the table number
        for col in code_columns:
            if isinstance(col, str) and col.startswith(f"{table}-"):
                table_code_col = col
                break
        
        # If we found a code column, look for the corresponding title column
        if table_code_col is not None:
            # The title column is typically the next column after the code column
            col_idx = list(combined_df.columns).index(table_code_col)
            if col_idx + 1 < len(combined_df.columns):
                table_title_col = combined_df.columns[col_idx + 1]
        
        # If we couldn't find the columns, skip this table
        if table_code_col is None or table_title_col is None:
            logger.warning(f"Could not find code and title columns for table {table}")
            continue
        
        # Create a new DataFrame for this table with the unified structure
        table_unified = pd.DataFrame({
            'OmniClass_Code': table_df[table_code_col],
            'OmniClass_Title': table_df[table_title_col],
            'Description': ''  # No description available in the original data
        })
        
        # Add the table number as a prefix to the OmniClass_Code if it's not already there
        # Also clean up the OmniClass_Code format to ensure it follows the pattern XX-XX XX XX
        def clean_omniclass_code(code, table_num):
            if not isinstance(code, str):
                return f"{table_num}-00 00 00"  # Default code if not a string
            
            # If code doesn't start with table number, add it
            if not code.startswith(f"{table_num}-"):
                code = f"{table_num}-{code}"
            
            # Extract all digits and dashes
            digits_and_dashes = ''.join(c for c in code if c.isdigit() or c == '-')
            
            # Ensure there's at least one dash
            if '-' not in digits_and_dashes:
                digits_and_dashes = f"{table_num}-{digits_and_dashes}"
            
            # Split by dash to get the parts
            parts = digits_and_dashes.split('-')
            if len(parts) < 2:
                return f"{table_num}-00 00 00"  # Default if can't parse
            
            # Get the table number and the rest of the code
            table_part = parts[0]
            code_part = ''.join(parts[1:])
            
            # Ensure code_part has at least 6 digits, pad with zeros if needed
            code_part = code_part.ljust(6, '0')
            
            # Format as XX-XX XX XX
            formatted_code = f"{table_part}-{code_part[:2]} {code_part[2:4]} {code_part[4:6]}"
            
            return formatted_code
        
        table_unified['OmniClass_Code'] = table_unified['OmniClass_Code'].apply(
            lambda x: clean_omniclass_code(x, table)
        )
        
        # Append to the unified DataFrame
        unified_df = pd.concat([unified_df, table_unified], ignore_index=True)
    
    # Handle duplicate OmniClass codes by adding a suffix
    duplicate_mask = unified_df.duplicated(subset=['OmniClass_Code'], keep=False)
    if duplicate_mask.any():
        logger.info(f"Found {duplicate_mask.sum()} rows with duplicate OmniClass_Code values")
        print(f"Found {duplicate_mask.sum()} rows with duplicate OmniClass_Code values")
        
        # Add a counter to make duplicate codes unique
        duplicate_codes = unified_df.loc[duplicate_mask, 'OmniClass_Code'].unique()
        for code in duplicate_codes:
            # Get all rows with this code
            code_mask = unified_df['OmniClass_Code'] == code
            # Add a suffix to make them unique
            for i, idx in enumerate(unified_df[code_mask].index):
                if i > 0:  # Skip the first occurrence
                    unified_df.loc[idx, 'OmniClass_Code'] = f"{unified_df.loc[idx, 'OmniClass_Code']}-{i}"
        
        logger.info("Duplicate OmniClass codes have been made unique")
        print("Duplicate OmniClass codes have been made unique")
    
    # Save the unified data
    unified_df.to_csv(unified_output_path, index=False)
    logger.info(f"Saved unified training data to {unified_output_path}")
    print(f"\nSaved unified training data to {unified_output_path}")
    print(f"Unified data contains {len(unified_df)} rows with {len(unified_df.columns)} columns")
    
    # Show a sample of the unified data
    print("\nSample of Unified Data:")
    print(unified_df.head(5).to_string())
    
    # Validate the unified data
    from fca_dashboard.utils.validation_utils import is_valid_omniclass_data
    
    validation_result = is_valid_omniclass_data(unified_df)
    
    print("\nValidation Results:")
    print(f"Valid: {validation_result['valid']}")
    
    if not validation_result['valid']:
        print("Errors:")
        for error in validation_result['errors']:
            print(f"  - {error}")
            logger.warning(f"Validation error: {error}")
    
    print("\nData Statistics:")
    for key, value in validation_result['stats'].items():
        if key not in ['columns', 'invalid_code_examples', 'duplicate_code_examples']:
            print(f"  - {key}: {value}")
    
    return combined_df


def main():
    """Main function."""
    logger = get_logger("omniclass_example")
    
    # Get environment information
    env = get_env_var(ENV_VAR_NAME, "development")
    root_dir = get_root_dir()
    
    logger.info(f"Starting OmniClass Generator Example in {env} environment")
    logger.info(f"Project root directory: {root_dir}")
    
    print("OmniClass Generator Example - Unified Training Data Plan")
    print("======================================================")
    print(f"Environment: {env}")
    print(f"Project root: {root_dir}")
    
    # Analyze a sample OmniClass file
    print("\nStep 1: Analyze a sample OmniClass file and map to unified headers")
    logger.info("Step 1: Analyzing a sample OmniClass file and mapping to unified headers")
    sample_df = analyze_omniclass_sample()
    
    # Extract data from all OmniClass files
    print("\nStep 2: Extract data from all OmniClass files and create unified training data")
    logger.info("Step 2: Extracting data from all OmniClass files and creating unified training data")
    combined_df = extract_all_omniclass_data()
    
    # Show the path to the output file
    output_path = settings.get("generator.omniclass.output_file", "fca_dashboard/generator/ingest/unified_training_data.csv")
    output_file = resolve_path(output_path)
    
    logger.info(f"Unified training data saved to: {output_file}")
    
    print(f"\nOutput file:")
    print(f"  - Unified training data: {output_file}")
    
    logger.info("OmniClass Generator Example completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())