"""
File utility module for Excel operations.

This module provides utilities for working with Excel files,
including file type detection and validation.
"""

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_excel_file_type(file_path: Union[str, Path]) -> Optional[str]:
    """
    Get the file type of an Excel file.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        The file type as a string (e.g., "xlsx", "csv"), or None if not an Excel file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file path
    path = resolve_path(file_path)
    
    # Check if file exists
    if not path.is_file():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    # Get the file extension
    extension = path.suffix.lower().lstrip(".")
    
    # Check if it's an Excel file
    if extension in ["xlsx", "xls", "xlsm", "xlsb"]:
        return extension
    elif extension == "csv":
        return "csv"
    else:
        return None


def is_excel_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file is an Excel file.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        True if the file is an Excel file, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_type = get_excel_file_type(file_path)
    return file_type is not None


def is_valid_excel_file(file_path: Union[str, Path]) -> bool:
    """
    Check if an Excel file is valid by attempting to read it.
    
    Args:
        file_path: Path to the file to check.
        
    Returns:
        True if the file is a valid Excel file, False otherwise.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    logger = get_logger("excel_utils")
    
    # Check if it's an Excel file
    if not is_excel_file(file_path):
        return False
    
    try:
        # Try to read the file
        if str(file_path).lower().endswith(".csv"):
            pd.read_csv(file_path, nrows=1)
        else:
            pd.read_excel(file_path, nrows=1)
        return True
    except Exception as e:
        logger.warning(f"Invalid Excel file {file_path}: {str(e)}")
        return False