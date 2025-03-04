"""
Extraction utility module for Excel operations.

This module provides utilities for extracting data from Excel files,
including header detection, configuration-based extraction, and more.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.analysis_utils import detect_header_row
from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type, is_excel_file
from fca_dashboard.utils.excel.sheet_utils import clean_sheet_name, get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def read_excel_with_header_detection(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    header_row: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Read an Excel file with automatic header detection.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        header_row: Row index to use as the header (0-based). If None, will attempt to detect the header row.
        **kwargs: Additional arguments to pass to pandas.read_excel or pandas.read_csv.
        
    Returns:
        A pandas DataFrame with the detected headers.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # If header_row is not provided, detect it
        if header_row is None:
            # Read a sample of the file to detect the header row
            if path.suffix.lower() == ".csv":
                sample_df = pd.read_csv(path, nrows=20, header=None)
            else:
                sample_df = pd.read_excel(path, sheet_name=sheet_name, nrows=20, header=None)
            
            # Detect the header row
            detected_header_row = detect_header_row(sample_df)
            
            if detected_header_row is not None:
                header_row = detected_header_row
                logger.info(f"Detected header row at index {header_row}")
            else:
                # If no header row is detected, use the first row
                header_row = 0
                logger.info("No header row detected, using the first row")
        
        # Read the file with the detected header row
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, header=header_row, **kwargs)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, **kwargs)
        
        # Clean up column names
        df.columns = [str(col).strip() for col in df.columns]
        
        return df
    except Exception as e:
        error_msg = f"Error reading Excel file {path} with header detection: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def extract_excel_with_config(
    file_path: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    Extract data from an Excel file using a configuration dictionary.
    
    This function provides a robust way to extract data from Excel files with complex structures,
    such as headers in non-standard positions, multiple sheets with different structures, etc.
    
    Args:
        file_path: Path to the Excel file.
        config: Configuration dictionary with the following structure:
            {
                "default": {  # Default settings for all sheets
                    "header_row": int or None,  # Row index to use as header (0-based)
                    "skip_rows": int or list,  # Rows to skip
                    "column_mapping": dict,  # Map original column names to new names
                    "required_columns": list,  # Columns that must exist
                    "drop_empty_rows": bool,  # Whether to drop rows that are all NaN
                    "drop_empty_columns": bool,  # Whether to drop columns that are all NaN
                    "clean_column_names": bool,  # Whether to clean column names
                    "strip_whitespace": bool,  # Whether to strip whitespace from string values
                    "convert_dtypes": bool,  # Whether to convert data types
                    "date_columns": list,  # Columns to convert to dates
                    "numeric_columns": list,  # Columns to convert to numeric
                    "boolean_columns": list,  # Columns to convert to boolean
                    "fillna_values": dict,  # Values to use for filling NaN values
                    "drop_columns": list,  # Columns to drop
                    "rename_columns": dict,  # Columns to rename
                    "sheet_name_mapping": dict,  # Map original sheet names to new names
                },
                "sheet_name1": {  # Settings specific to sheet_name1, overrides defaults
                    # Same structure as default
                },
                "sheet_name2": {
                    # Same structure as default
                }
            }
        **kwargs: Additional arguments to pass to pandas.read_excel or pandas.read_csv.
        
    Returns:
        A dictionary mapping sheet names to pandas DataFrames.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs during extraction.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's an Excel file
    if not is_excel_file(path):
        error_msg = f"Not an Excel file: {path}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    # Get the file type
    file_type = get_excel_file_type(path)
    
    # Get the sheet names
    sheet_names = get_sheet_names(path)
    
    # Initialize the default configuration
    default_config = {
        "header_row": None,  # Auto-detect
        "skip_rows": None,
        "column_mapping": {},
        "required_columns": [],
        "drop_empty_rows": True,
        "drop_empty_columns": False,
        "clean_column_names": True,
        "strip_whitespace": True,
        "convert_dtypes": True,
        "date_columns": [],
        "numeric_columns": [],
        "boolean_columns": [],
        "fillna_values": {},
        "drop_columns": [],
        "rename_columns": {},
        "sheet_name_mapping": {},
    }
    
    # Use provided config or empty dict
    config = config or {}
    
    # Get the default configuration from the provided config
    default_config.update(config.get("default", {}))
    
    # Initialize the result dictionary
    result = {}
    
    # Process each sheet
    for sheet_name in sheet_names:
        logger.info(f"Processing sheet: {sheet_name}")
        
        try:
            # Get sheet-specific configuration, falling back to default
            sheet_config = default_config.copy()
            sheet_config.update(config.get(sheet_name, {}))
            
            # Determine the header row
            header_row = sheet_config["header_row"]
            
            # If header_row is None, try to detect it
            if header_row is None:
                # Read a sample of the sheet to detect the header row
                if file_type == "csv":
                    sample_df = pd.read_csv(path, nrows=20, header=None)
                else:
                    sample_df = pd.read_excel(path, sheet_name=sheet_name, nrows=20, header=None)
                
                # Detect the header row
                detected_header_row = detect_header_row(sample_df)
                
                if detected_header_row is not None:
                    header_row = detected_header_row
                    logger.info(f"Detected header row at index {header_row}")
                else:
                    # If no header row is detected, use the first row
                    header_row = 0
                    logger.info("No header row detected, using the first row")
            
            # Determine the rows to skip
            skip_rows = sheet_config["skip_rows"]
            
            # Read the sheet
            if file_type == "csv":
                df = pd.read_csv(path, header=header_row, skiprows=skip_rows, **kwargs)
            else:
                df = pd.read_excel(path, sheet_name=sheet_name, header=header_row, skiprows=skip_rows, **kwargs)
            
            # Clean column names if requested
            if sheet_config["clean_column_names"]:
                df.columns = [str(col).strip() for col in df.columns]
            
            # Check for required columns
            required_columns = sheet_config["required_columns"]
            if required_columns:
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    error_msg = f"Required columns not found in sheet {sheet_name}: {missing_columns}"
                    logger.error(error_msg)
                    raise ExcelUtilError(error_msg)
            
            # Drop empty rows if requested
            if sheet_config["drop_empty_rows"]:
                df = df.dropna(how='all')
            
            # Drop empty columns if requested
            if sheet_config["drop_empty_columns"]:
                df = df.dropna(axis=1, how='all')
            
            # Strip whitespace from string values if requested
            if sheet_config["strip_whitespace"]:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype(str).str.strip()
            
            # Convert data types if requested
            if sheet_config["convert_dtypes"]:
                # Convert date columns
                for col in sheet_config["date_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to date: {str(e)}")
                
                # Convert numeric columns
                for col in sheet_config["numeric_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to numeric: {str(e)}")
                
                # Convert boolean columns
                for col in sheet_config["boolean_columns"]:
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(str).str.lower().map({
                                'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                                'false': False, 'no': False, 'n': False, '0': False, 'f': False,
                                'nan': None, 'none': None, 'null': None, '': None
                            })
                        except Exception as e:
                            logger.warning(f"Error converting column {col} to boolean: {str(e)}")
            
            # Fill NaN values if requested
            fillna_values = sheet_config["fillna_values"]
            if fillna_values:
                df = df.fillna(fillna_values)
            
            # Drop columns if requested
            drop_columns = sheet_config["drop_columns"]
            if drop_columns:
                df = df.drop(columns=[col for col in drop_columns if col in df.columns])
            
            # Rename columns if requested
            rename_columns = sheet_config["rename_columns"]
            if rename_columns:
                df = df.rename(columns=rename_columns)
            
            # Apply column mapping if provided
            column_mapping = sheet_config["column_mapping"]
            if column_mapping:
                # Create a new DataFrame with mapped columns
                new_df = pd.DataFrame()
                for new_col, old_col in column_mapping.items():
                    if old_col in df.columns:
                        new_df[new_col] = df[old_col]
                
                # Replace the original DataFrame with the mapped one
                if not new_df.empty:
                    df = new_df
            
            # Get the normalized sheet name
            sheet_name_mapping = sheet_config["sheet_name_mapping"]
            if sheet_name_mapping and sheet_name in sheet_name_mapping:
                normalized_sheet_name = sheet_name_mapping[sheet_name]
            else:
                normalized_sheet_name = clean_sheet_name(sheet_name)
            
            # Store the processed DataFrame
            result[normalized_sheet_name] = df
            
            logger.info(f"Successfully processed sheet {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
        
        except Exception as e:
            error_msg = f"Error processing sheet {sheet_name}: {str(e)}"
            logger.error(error_msg)
            if isinstance(e, ExcelUtilError):
                raise
            else:
                raise ExcelUtilError(error_msg) from e
    
    return result


def load_excel_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load an Excel configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        A dictionary containing the configuration.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while loading the configuration.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(config_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"Configuration file not found: {path}")
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    try:
        # Determine the file type
        if path.suffix.lower() == ".json":
            import json
            with open(path, 'r') as f:
                config = json.load(f)
        elif path.suffix.lower() in [".yml", ".yaml"]:
            import yaml
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            error_msg = f"Unsupported configuration file format: {path.suffix}"
            logger.error(error_msg)
            raise ExcelUtilError(error_msg)
        
        return config
    except Exception as e:
        error_msg = f"Error loading configuration from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e