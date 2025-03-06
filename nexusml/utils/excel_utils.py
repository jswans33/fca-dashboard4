"""
Excel utility module for the NexusML application.

This module provides utilities for working with Excel files,
particularly for data extraction and cleaning.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DataCleaningError(Exception):
    """Exception raised for errors in the data cleaning process."""

    pass


def get_logger(name: str):
    """Simple logger function."""
    import logging

    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)


def resolve_path(path: Union[str, Path, None]) -> Path:
    """
    Resolve a path to an absolute path.

    Args:
        path: The path to resolve. If None, returns the current working directory.

    Returns:
        The resolved path as a Path object.
    """
    if path is None:
        return Path.cwd()
    if isinstance(path, str):
        path = Path(path)
    return path.resolve()


def get_sheet_names(file_path: Union[str, Path]) -> List[str]:
    """
    Get sheet names from an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        List of sheet names as strings.
    """
    # Convert all sheet names to strings to ensure type safety
    return [str(name) for name in pd.ExcelFile(file_path).sheet_names]


def extract_excel_with_config(
    file_path: Union[str, Path], config: Dict[str, Dict[str, Any]]
) -> Dict[str, pd.DataFrame]:
    """
    Extract data from Excel file using a configuration.

    Args:
        file_path: Path to the Excel file.
        config: Configuration dictionary with sheet names as keys and sheet configs as values.
            Each sheet config can have the following keys:
            - header_row: Row index to use as header (default: 0)
            - drop_empty_rows: Whether to drop empty rows (default: False)
            - strip_whitespace: Whether to strip whitespace from string columns (default: False)

    Returns:
        Dictionary with sheet names as keys and DataFrames as values.
    """
    result = {}
    for sheet_name, sheet_config in config.items():
        header_row = sheet_config.get("header_row", 0)
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

        if sheet_config.get("drop_empty_rows", False):
            df = df.dropna(how="all")

        if sheet_config.get("strip_whitespace", False):
            for col in df.select_dtypes(include=["object"]).columns:
                df[col] = df[col].str.strip()

        result[sheet_name] = df
    return result


def normalize_sheet_names(file_path: Union[str, Path]) -> Dict[str, str]:
    """
    Normalize sheet names in an Excel file.

    Args:
        file_path: Path to the Excel file.

    Returns:
        Dictionary mapping original sheet names to normalized names.
    """
    sheet_names = get_sheet_names(file_path)
    return {name: name.lower().replace(" ", "_") for name in sheet_names}


def find_flat_sheet(sheet_names: List[str]) -> Optional[str]:
    """
    Find the sheet name that contains 'FLAT' in it.

    Args:
        sheet_names: List of sheet names to search through.

    Returns:
        The name of the sheet containing 'FLAT', or None if not found.
    """
    for sheet in sheet_names:
        if "FLAT" in sheet.upper():
            return sheet
    return None


def clean_dataframe(
    df: pd.DataFrame,
    header_patterns: Optional[List[str]] = None,
    copyright_patterns: Optional[List[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    is_omniclass: bool = False,
) -> pd.DataFrame:
    """
    Clean a DataFrame.

    Args:
        df: The DataFrame to clean.
        header_patterns: List of patterns to identify the header row.
        copyright_patterns: List of patterns to identify copyright rows.
        column_mapping: Dictionary mapping original column names to standardized names.
        is_omniclass: Whether the DataFrame contains OmniClass data, which requires special handling.

    Returns:
        A cleaned DataFrame.
    """
    # Basic cleaning
    df = df.copy()

    # Drop completely empty rows
    df = df.dropna(how="all")

    # Handle OmniClass specific cleaning
    if is_omniclass:
        # Look for common OmniClass column names
        for col in df.columns:
            col_str = str(col).lower()
            if "number" in col_str:
                df.rename(columns={col: "OmniClass_Code"}, inplace=True)
            elif "title" in col_str:
                df.rename(columns={col: "OmniClass_Title"}, inplace=True)
            elif "definition" in col_str:
                df.rename(columns={col: "Description"}, inplace=True)

    return df


def standardize_column_names(
    df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame.

    Args:
        df: The DataFrame to standardize.
        column_mapping: Dictionary mapping original column names to standardized names.
            If None, uses default mapping.

    Returns:
        A new DataFrame with standardized column names.
    """
    if column_mapping:
        df = df.rename(columns={v: k for k, v in column_mapping.items()})
    return df
