"""
Analysis utility module for Excel operations.

This module provides utilities for analyzing Excel files,
including structure analysis, header detection, and column analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.file_utils import get_excel_file_type, is_excel_file
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def analyze_excel_structure(
    file_path: Union[str, Path],
    max_rows: int = 20
) -> Dict[str, Any]:
    """
    Analyze the structure of an Excel file and return information about its sheets, headers, etc.
    
    Args:
        file_path: Path to the Excel file to analyze.
        max_rows: Maximum number of rows to read for analysis (default: 20).
        
    Returns:
        A dictionary containing information about the Excel file structure:
        {
            'file_type': str,
            'sheet_names': List[str],
            'sheets_info': Dict[str, Dict],  # Information about each sheet
        }
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ExcelUtilError: If an error occurs during analysis.
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
    
    try:
        # Get the file type
        file_type = get_excel_file_type(path)
        
        # Get the sheet names
        sheet_names = get_sheet_names(path)
        
        # Initialize the result dictionary
        result = {
            'file_type': file_type,
            'sheet_names': sheet_names,
            'sheets_info': {}
        }
        
        # Analyze each sheet
        for sheet_name in sheet_names:
            # Read a sample of the sheet
            if file_type == "csv":
                df_sample = pd.read_csv(path, nrows=max_rows)
            else:
                df_sample = pd.read_excel(path, sheet_name=sheet_name, nrows=max_rows)
            
            # Analyze the sheet structure
            sheet_info = {
                'shape': df_sample.shape,
                'columns': list(df_sample.columns),
                'empty_rows': detect_empty_rows(df_sample),
                'header_row': detect_header_row(df_sample),
                'duplicate_columns': detect_duplicate_columns(df_sample),
                'unnamed_columns': detect_unnamed_columns(df_sample)
            }
            
            result['sheets_info'][sheet_name] = sheet_info
        
        return result
    except Exception as e:
        error_msg = f"Error analyzing Excel file {path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def detect_empty_rows(df: pd.DataFrame, max_rows: int = 10) -> List[int]:
    """
    Detect empty rows in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        max_rows: Maximum number of rows to check (default: 10).
        
    Returns:
        A list of indices of empty rows.
    """
    empty_rows = []
    for i in range(min(max_rows, len(df))):
        if df.iloc[i].isna().all():
            empty_rows.append(i)
    return empty_rows


def detect_header_row(df: pd.DataFrame, max_rows: int = 10) -> Optional[int]:
    """
    Detect the most likely header row in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        max_rows: Maximum number of rows to check (default: 10).
        
    Returns:
        The index of the most likely header row, or None if no header row is detected.
    """
    potential_header_rows = []
    
    for i in range(min(max_rows, len(df))):
        # Skip empty rows
        if df.iloc[i].isna().all():
            continue
        
        # Check if this row has string values that could be headers
        row = df.iloc[i]
        
        # Count string values that could be headers
        string_count = sum(1 for val in row if isinstance(val, str) and val.strip())
        
        # If most cells in this row are non-empty strings, it might be a header row
        if string_count / len(row) > 0.5:
            potential_header_rows.append((i, string_count))
    
    # Return the row with the most string values, or None if no potential header rows
    if potential_header_rows:
        # Sort by string count in descending order
        potential_header_rows.sort(key=lambda x: x[1], reverse=True)
        return potential_header_rows[0][0]
    
    return None


def detect_duplicate_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect duplicate column names in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        
    Returns:
        A list of duplicate column names.
    """
    return df.columns[df.columns.duplicated()].tolist()


def detect_unnamed_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect unnamed columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        
    Returns:
        A list of unnamed column names.
    """
    return [str(col) for col in df.columns if "Unnamed" in str(col)]


def analyze_unique_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    max_unique_values: int = 20
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze unique values in specified columns of a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all columns.
        max_unique_values: Maximum number of unique values to include in the result.
            If a column has more unique values than this, only the counts will be included.
        
    Returns:
        A dictionary mapping column names to dictionaries containing:
            - 'count': The number of unique values
            - 'values': The unique values (if count <= max_unique_values)
            - 'value_counts': Dictionary mapping values to their counts (if count <= max_unique_values)
            - 'null_count': The number of null values
            - 'null_percentage': The percentage of null values
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, analyze all columns
    if columns is None:
        columns = df.columns
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get unique values
        unique_values = df[col].dropna().unique()
        unique_count = len(unique_values)
        
        # Get null count and percentage
        null_count = df[col].isna().sum()
        null_percentage = null_count / len(df) if len(df) > 0 else 0.0
        
        # Initialize column result
        col_result = {
            'count': unique_count,
            'null_count': null_count,
            'null_percentage': null_percentage
        }
        
        # Include unique values and counts if there aren't too many
        if unique_count <= max_unique_values:
            # Convert values to strings for better display
            unique_values_str = [str(val) for val in unique_values]
            
            # Get value counts
            value_counts = df[col].value_counts().to_dict()
            
            # Convert keys to strings for better display
            value_counts_str = {str(k): v for k, v in value_counts.items() if pd.notna(k)}
            
            col_result['values'] = unique_values_str
            col_result['value_counts'] = value_counts_str
        
        # Add to result
        result[col] = col_result
    
    return result


def analyze_column_statistics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate statistics for numeric columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all numeric columns.
        
    Returns:
        A dictionary mapping column names to dictionaries containing statistics:
            - 'min': Minimum value
            - 'max': Maximum value
            - 'mean': Mean value
            - 'median': Median value
            - 'std': Standard deviation
            - 'q1': First quartile (25th percentile)
            - 'q3': Third quartile (75th percentile)
            - 'iqr': Interquartile range
            - 'outliers_count': Number of outliers (values outside 1.5*IQR from Q1 and Q3)
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns
    
    # Initialize result dictionary
    result = {}
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning(f"Column '{col}' is not numeric, skipping statistics calculation")
            continue
        
        # Get numeric values (drop NaN)
        values = df[col].dropna()
        
        # Skip if no values
        if len(values) == 0:
            logger.warning(f"Column '{col}' has no non-null values, skipping statistics calculation")
            continue
        
        # Calculate statistics
        min_val = values.min()
        max_val = values.max()
        mean_val = values.mean()
        median_val = values.median()
        std_val = values.std()
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = values[(values < lower_bound) | (values > upper_bound)]
        outliers_count = len(outliers)
        
        # Add to result
        result[col] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'outliers_count': outliers_count
        }
    
    return result


def analyze_text_columns(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze text columns in a DataFrame.
    
    Args:
        df: The DataFrame to analyze.
        columns: Optional list of column names to analyze. If None, analyzes all object columns.
        
    Returns:
        A dictionary mapping column names to dictionaries containing:
            - 'min_length': Minimum string length
            - 'max_length': Maximum string length
            - 'avg_length': Average string length
            - 'empty_count': Number of empty strings
            - 'pattern_analysis': Dictionary with counts of different patterns
                (e.g., 'numeric', 'alpha', 'alphanumeric', 'email', 'url', 'date', 'other')
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, use all object columns
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    # Initialize result dictionary
    result = {}
    
    # Compile regex patterns
    import re
    numeric_pattern = re.compile(r'^\d+$')
    alpha_pattern = re.compile(r'^[a-zA-Z]+$')
    alphanumeric_pattern = re.compile(r'^[a-zA-Z0-9]+$')
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    url_pattern = re.compile(r'^(http|https)://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$')
    date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
    
    # Analyze each column
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Convert to string and drop NaN
        values = df[col].astype(str).replace('nan', '').replace('None', '')
        
        # Skip if no values
        if len(values) == 0:
            logger.warning(f"Column '{col}' has no values, skipping text analysis")
            continue
        
        # Calculate string lengths
        lengths = values.str.len()
        min_length = lengths.min()
        max_length = lengths.max()
        avg_length = lengths.mean()
        
        # Count empty strings
        empty_count = (lengths == 0).sum()
        
        # Analyze patterns
        pattern_counts = {
            'numeric': 0,
            'alpha': 0,
            'alphanumeric': 0,
            'email': 0,
            'url': 0,
            'date': 0,
            'other': 0
        }
        
        for val in values:
            if val == '':
                continue
            elif numeric_pattern.match(val):
                pattern_counts['numeric'] += 1
            elif alpha_pattern.match(val):
                pattern_counts['alpha'] += 1
            elif alphanumeric_pattern.match(val):
                pattern_counts['alphanumeric'] += 1
            elif email_pattern.match(val):
                pattern_counts['email'] += 1
            elif url_pattern.match(val):
                pattern_counts['url'] += 1
            elif date_pattern.match(val):
                pattern_counts['date'] += 1
            else:
                pattern_counts['other'] += 1
        
        # Add to result
        result[col] = {
            'min_length': min_length,
            'max_length': max_length,
            'avg_length': avg_length,
            'empty_count': empty_count,
            'pattern_analysis': pattern_counts
        }
    
    return result