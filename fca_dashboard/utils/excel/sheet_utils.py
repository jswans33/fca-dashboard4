"""
Sheet utility module for Excel operations.

This module provides utilities for working with Excel sheets,
including sheet name retrieval, cleaning, and normalization.
"""

import re
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def get_sheet_names(file_path: Union[str, Path]) -> List[Union[str, int]]:
    """
    Get the sheet names from an Excel file.
    
    Args:
        file_path: Path to the Excel file.
        
    Returns:
        A list of sheet names. For CSV files, returns a list with a single element [0].
        
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
        # Handle CSV files
        if path.suffix.lower() == ".csv":
            return [0]  # CSV files have a single sheet with index 0
        
        # Get sheet names from Excel file
        excel_file = pd.ExcelFile(path)
        return excel_file.sheet_names
    except Exception as e:
        error_msg = f"Error getting sheet names from {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def clean_sheet_name(sheet_name: str) -> str:
    """
    Clean and normalize a sheet name.
    
    Args:
        sheet_name: The sheet name to clean.
        
    Returns:
        A cleaned and normalized sheet name.
    """
    # Convert to string if it's not already
    sheet_name = str(sheet_name)
    
    # Remove leading/trailing whitespace
    sheet_name = sheet_name.strip()
    
    # Replace special characters with underscores
    sheet_name = re.sub(r'[^\w\s]', '_', sheet_name)
    
    # Replace multiple spaces with a single underscore
    sheet_name = re.sub(r'\s+', '_', sheet_name)
    
    # Convert to lowercase
    sheet_name = sheet_name.lower()
    
    return sheet_name


def normalize_sheet_names(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Get a mapping of original sheet names to normalized sheet names.
    
    Args:
        file_path: Path to the Excel file.
        
    Returns:
        A dictionary mapping original sheet names to normalized sheet names.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs while reading the file.
    """
    # Get the sheet names
    sheet_names = get_sheet_names(file_path)
    
    # Create a mapping of original to normalized sheet names
    sheet_name_mapping = {}
    for name in sheet_names:
        normalized_name = clean_sheet_name(name)
        sheet_name_mapping[name] = normalized_name
    
    return sheet_name_mapping