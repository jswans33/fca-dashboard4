"""
Column utility module for Excel operations.

This module provides utilities for working with Excel columns,
including column name retrieval and validation.
"""

from pathlib import Path
from typing import List, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_column_names(file_path: Union[str, Path], sheet_name: Union[str, int] = 0) -> List[str]:
    """
    Get the column names from an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        
    Returns:
        A list of column names.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If the sheet does not exist or an error occurs while reading the file.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Read the first row of the file to get column names
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=0)
        else:
            # Check if the sheet exists
            sheet_names = get_sheet_names(path)
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {path}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            df = pd.read_excel(path, sheet_name=actual_sheet_name, nrows=0)
        
        return list(df.columns)
    except ExcelUtilError:
        # Re-raise ExcelUtilError
        raise
    except Exception as e:
        error_msg = f"Error getting column names from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def validate_columns_exist(file_path: Union[str, Path], columns: List[str], sheet_name: Union[str, int] = 0) -> bool:
    """
    Validate that the specified columns exist in an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        columns: List of column names to validate.
        sheet_name: Name or index of the sheet to read (default: 0, first sheet).
        
    Returns:
        True if all columns exist, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    # If no columns to validate, return True
    if not columns:
        return True
    
    # Get the column names from the file
    file_columns = get_column_names(file_path, sheet_name)
    
    # Check if all columns exist
    return all(column in file_columns for column in columns)