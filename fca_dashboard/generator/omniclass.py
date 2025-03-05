"""
OmniClass data extraction module for the FCA Dashboard application.

This module provides utilities for extracting OmniClass data from Excel files
and generating a unified CSV file for classifier training.
"""

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.data_cleaning_utils import DataCleaningError, clean_dataframe
from fca_dashboard.utils.excel import (
    analyze_excel_structure,
    extract_excel_with_config,
    get_sheet_names,
)
from fca_dashboard.utils.excel.sheet_utils import normalize_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


def find_flat_sheet(sheet_names: List[str]) -> Optional[str]:
    """
    Find the sheet name that contains 'FLAT' in it.
    
    Args:
        sheet_names: List of sheet names to search through.
        
    Returns:
        The name of the sheet containing 'FLAT', or None if not found.
    """
    for sheet in sheet_names:
        if 'FLAT' in sheet.upper():
            return sheet
    return None


def extract_omniclass_data(
    input_dir: Optional[Union[str, Path]] = None,
    output_file: Optional[Union[str, Path]] = None,
    file_pattern: str = "*.xlsx"
) -> pd.DataFrame:
    """
    Extract OmniClass data from Excel files and save to a CSV file.
    
    Args:
        input_dir: Directory containing OmniClass Excel files.
            If None, uses the directory from settings.
        output_file: Path to save the output CSV file.
            If None, uses the path from settings.
        file_pattern: Pattern to match Excel files (default: "*.xlsx").
        
    Returns:
        DataFrame containing the combined OmniClass data.
        
    Raises:
        FileNotFoundError: If the input directory does not exist.
        ValueError: If no OmniClass files are found or if no FLAT sheets are found.
    """
    logger = get_logger("generator")
    
    # Use settings if parameters are not provided
    if input_dir is None:
        input_dir = settings.get("generator.omniclass.input_dir", "files/omniclass_tables")
    
    if output_file is None:
        output_file = settings.get("generator.omniclass.output_file", "fca_dashboard/generator/ingest/omniclass.csv")
    
    # Resolve paths
    input_dir = resolve_path(input_dir)
    output_file = resolve_path(output_file)
    
    logger.info(f"Extracting OmniClass data from {input_dir}")
    
    # Check if input directory exists
    if not input_dir.is_dir():
        error_msg = f"Input directory does not exist: {input_dir}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Find all Excel files in the input directory
    file_paths = list(input_dir.glob(file_pattern))
    
    if not file_paths:
        error_msg = f"No Excel files found in {input_dir} matching pattern {file_pattern}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Found {len(file_paths)} Excel files")
    
    # Create the output directory if it doesn't exist
    output_dir = output_file.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each Excel file
    all_data = []
    
    for file_path in file_paths:
        logger.info(f"Processing {file_path.name}")
        
        try:
            # Get sheet names
            sheet_names = get_sheet_names(file_path)
            
            # Find the FLAT sheet
            flat_sheet = find_flat_sheet(sheet_names)
            
            if flat_sheet is None:
                logger.warning(f"No FLAT sheet found in {file_path.name}, skipping")
                continue
            
            logger.info(f"Found FLAT sheet: {flat_sheet}")
            
            # Create extraction config
            config = {
                flat_sheet: {
                    "header_row": 0,  # Assume header is in the first row
                    "drop_empty_rows": True,
                    "clean_column_names": True,
                    "strip_whitespace": True,
                }
            }
            
            # Extract data from the FLAT sheet
            extracted_data = extract_excel_with_config(file_path, config)
            
            # Find the sheet in the extracted data
            # The sheet name might have been normalized
            df = None
            if flat_sheet in extracted_data:
                df = extracted_data[flat_sheet]
            else:
                # Try to find a sheet with a similar name
                normalized_sheet_names = normalize_sheet_names(file_path)
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
                    if extracted_data:
                        sheet_name = list(extracted_data.keys())[0]
                        df = extracted_data[sheet_name]
                        logger.warning(f"Could not find sheet '{flat_sheet}', using '{sheet_name}' instead")
                    else:
                        logger.warning(f"Failed to extract data from {flat_sheet} in {file_path.name}")
                        continue
            
            if df is None:
                logger.warning(f"Failed to extract data from {flat_sheet} in {file_path.name}")
                continue
            
            try:
                # Clean the DataFrame using our data cleaning utilities
                # Set is_omniclass=True to enable special handling for OmniClass headers
                cleaned_df = clean_dataframe(df, is_omniclass=True)
                
                # Add file name as a column for tracking
                cleaned_df['source_file'] = file_path.name
                
                # Add table number from filename (e.g., OmniClass_22_2020-08-15_2022.xlsx -> 22)
                table_number = file_path.stem.split('_')[1] if len(file_path.stem.split('_')) > 1 else ''
                cleaned_df['table_number'] = table_number
                
                # Append to the list of dataframes
                all_data.append(cleaned_df)
                
                logger.info(f"Cleaned and extracted {len(cleaned_df)} rows from {file_path.name}")
            except DataCleaningError as e:
                logger.warning(f"Error cleaning data from {file_path.name}: {str(e)}")
                continue
            
            logger.info(f"Extracted {len(df)} rows from {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    if not all_data:
        error_msg = "No data extracted from any OmniClass files"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get column mapping from settings
    column_mapping = settings.get("generator.omniclass.column_mapping", {
        'Number': 'OmniClass_Code',
        'Title': 'OmniClass_Title',
        'Definition': 'Description'
    })
    
    # No need to apply column mapping here as it's already done in clean_dataframe
    # But we can standardize any remaining columns if needed
    from fca_dashboard.utils.data_cleaning_utils import standardize_column_names
    combined_df = standardize_column_names(combined_df, column_mapping=column_mapping)
    
    # Save to CSV if output_file is not None
    if output_file is not None:
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(combined_df)} rows to {output_file}")
    else:
        logger.info(f"Skipping saving to CSV as output_file is None")
    
    return combined_df


def main():
    """
    Main function to run the OmniClass data extraction as a standalone script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract OmniClass data from Excel files')
    parser.add_argument('--input-dir', type=str, help='Directory containing OmniClass Excel files')
    parser.add_argument('--output-file', type=str, help='Path to save the output CSV file')
    parser.add_argument('--file-pattern', type=str, default="*.xlsx", help='Pattern to match Excel files')
    
    args = parser.parse_args()
    
    # Extract OmniClass data
    extract_omniclass_data(
        input_dir=args.input_dir,
        output_file=args.output_file,
        file_pattern=args.file_pattern
    )


if __name__ == "__main__":
    main()