"""
Validation utility module for Excel operations.

This module provides utilities for validating Excel data,
including checking for NaN values, null values, and other validation checks.
"""

from typing import Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.logging_config import get_logger


def check_missing_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.0,
    raise_error: bool = False
) -> Dict[str, float]:
    """
    Check for missing values (NaN, None) in a DataFrame.
    
    Args:
        df: The DataFrame to check.
        columns: Optional list of column names to check. If None, checks all columns.
        threshold: Maximum allowed percentage of missing values (0.0 to 1.0).
            If the percentage of missing values exceeds this threshold, an error is raised.
        raise_error: Whether to raise an error if the threshold is exceeded.
        
    Returns:
        A dictionary mapping column names to the percentage of missing values.
        
    Raises:
        ExcelUtilError: If raise_error is True and any column exceeds the threshold.
    """
    logger = get_logger("excel_utils")
    
    # If columns is None, check all columns
    if columns is None:
        columns = df.columns
    
    # Calculate the percentage of missing values for each column
    missing_percentages = {}
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Count missing values (NaN, None)
        missing_count = df[col].isna().sum()
        total_count = len(df)
        
        # Calculate percentage
        if total_count > 0:
            missing_percentage = missing_count / total_count
        else:
            missing_percentage = 0.0
        
        missing_percentages[col] = missing_percentage
    
    # Check if any column exceeds the threshold
    if raise_error and any(pct > threshold for pct in missing_percentages.values()):
        # Get columns that exceed the threshold
        exceeding_columns = [col for col, pct in missing_percentages.items() if pct > threshold]
        
        # Format the error message
        error_msg = f"The following columns exceed the missing values threshold ({threshold * 100}%):\n"
        for col in exceeding_columns:
            error_msg += f"  - {col}: {missing_percentages[col] * 100:.2f}%\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return missing_percentages


def check_duplicate_rows(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    raise_error: bool = False
) -> Dict[str, int]:
    """
    Check for duplicate rows in a DataFrame.
    
    Args:
        df: The DataFrame to check.
        subset: Optional list of column names to consider when identifying duplicates.
            If None, uses all columns.
        raise_error: Whether to raise an error if duplicates are found.
        
    Returns:
        A dictionary with the count of duplicate rows and the indices of duplicate rows.
        
    Raises:
        ExcelUtilError: If raise_error is True and duplicates are found.
    """
    logger = get_logger("excel_utils")
    
    # Find duplicate rows
    duplicates = df.duplicated(subset=subset, keep='first')
    duplicate_indices = df[duplicates].index.tolist()
    duplicate_count = len(duplicate_indices)
    
    # Create result dictionary
    result = {
        "duplicate_count": duplicate_count,
        "duplicate_indices": duplicate_indices
    }
    
    # Check if duplicates were found
    if raise_error and duplicate_count > 0:
        # Format the error message
        if subset:
            error_msg = f"Found {duplicate_count} duplicate rows based on columns: {subset}"
        else:
            error_msg = f"Found {duplicate_count} duplicate rows"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def check_value_ranges(
    df: pd.DataFrame,
    ranges: Dict[str, Dict[str, Union[int, float, None]]],
    raise_error: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Check if values in specified columns are within the given ranges.
    
    Args:
        df: The DataFrame to check.
        ranges: A dictionary mapping column names to range specifications.
            Each range specification is a dictionary with 'min' and 'max' keys.
            Example: {'age': {'min': 0, 'max': 120}, 'score': {'min': 0, 'max': 100}}
        raise_error: Whether to raise an error if values are outside the ranges.
        
    Returns:
        A dictionary mapping column names to dictionaries with counts of values outside the ranges.
        
    Raises:
        ExcelUtilError: If raise_error is True and values are outside the ranges.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Check each column
    for col, range_spec in ranges.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get min and max values
        min_val = range_spec.get('min')
        max_val = range_spec.get('max')
        
        # Initialize counters
        below_min = 0
        above_max = 0
        
        # Check values
        if min_val is not None:
            below_min = df[df[col] < min_val].shape[0]
        
        if max_val is not None:
            above_max = df[df[col] > max_val].shape[0]
        
        # Store results
        result[col] = {
            'below_min': below_min,
            'above_max': above_max,
            'total_outside_range': below_min + above_max
        }
    
    # Check if any values are outside the ranges
    if raise_error and any(res['total_outside_range'] > 0 for res in result.values()):
        # Format the error message
        error_msg = "Values outside specified ranges found:\n"
        for col, res in result.items():
            if res['total_outside_range'] > 0:
                range_spec = ranges[col]
                min_val = range_spec.get('min', 'None')
                max_val = range_spec.get('max', 'None')
                error_msg += f"  - {col} (range: {min_val} to {max_val}):\n"
                error_msg += f"    - Below minimum: {res['below_min']}\n"
                error_msg += f"    - Above maximum: {res['above_max']}\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def check_data_types(
    df: pd.DataFrame,
    type_specs: Dict[str, str],
    raise_error: bool = False
) -> Dict[str, Dict[str, Union[str, int]]]:
    """
    Check if values in specified columns have the expected data types.
    
    Args:
        df: The DataFrame to check.
        type_specs: A dictionary mapping column names to expected data types.
            Supported types: 'int', 'float', 'str', 'bool', 'date'.
            Example: {'age': 'int', 'name': 'str', 'score': 'float', 'is_active': 'bool', 'birth_date': 'date'}
        raise_error: Whether to raise an error if values have incorrect types.
        
    Returns:
        A dictionary mapping column names to dictionaries with type information and error counts.
        
    Raises:
        ExcelUtilError: If raise_error is True and values have incorrect types.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Check each column
    for col, expected_type in type_specs.items():
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in DataFrame")
            continue
        
        # Get the current type
        current_type = str(df[col].dtype)
        
        # Initialize error count
        error_count = 0
        
        # Check type based on expected_type
        if expected_type == 'int':
            # Check if values can be converted to integers
            try:
                pd.to_numeric(df[col], errors='raise', downcast='integer')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna() & ~df[col].astype(str).str.match(r'^-?\d+$')].shape[0]
        
        elif expected_type == 'float':
            # Check if values can be converted to floats
            try:
                pd.to_numeric(df[col], errors='raise')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna() & ~df[col].astype(str).str.match(r'^-?\d+(\.\d+)?$')].shape[0]
        
        elif expected_type == 'str':
            # Check if values are strings
            error_count = df[~df[col].isna() & ~df[col].apply(lambda x: isinstance(x, str))].shape[0]
        
        elif expected_type == 'bool':
            # Check if values are booleans or can be interpreted as booleans
            valid_bool_values = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f', True, False, 1, 0}
            error_count = df[~df[col].isna() & ~df[col].astype(str).str.lower().isin(valid_bool_values)].shape[0]
        
        elif expected_type == 'date':
            # Check if values can be converted to dates
            try:
                pd.to_datetime(df[col], errors='raise')
            except (ValueError, TypeError):
                error_count = df[~df[col].isna()].shape[0]
        
        else:
            logger.warning(f"Unsupported type specification: {expected_type}")
            continue
        
        # Store results
        result[col] = {
            'expected_type': expected_type,
            'current_type': current_type,
            'error_count': error_count
        }
    
    # Check if any values have incorrect types
    if raise_error and any(res['error_count'] > 0 for res in result.values()):
        # Format the error message
        error_msg = "Values with incorrect data types found:\n"
        for col, res in result.items():
            if res['error_count'] > 0:
                error_msg += f"  - {col} (expected: {res['expected_type']}, current: {res['current_type']}):\n"
                error_msg += f"    - Error count: {res['error_count']}\n"
        
        logger.error(error_msg)
        raise ExcelUtilError(error_msg)
    
    return result


def validate_dataframe(
    df: pd.DataFrame,
    validation_config: Dict[str, Dict],
    raise_error: bool = False
) -> Dict[str, Dict]:
    """
    Validate a DataFrame using multiple validation checks.
    
    Args:
        df: The DataFrame to validate.
        validation_config: A dictionary with validation configurations.
            Example:
            {
                'missing_values': {
                    'columns': ['col1', 'col2'],
                    'threshold': 0.1
                },
                'duplicate_rows': {
                    'subset': ['col1', 'col2']
                },
                'value_ranges': {
                    'age': {'min': 0, 'max': 120},
                    'score': {'min': 0, 'max': 100}
                },
                'data_types': {
                    'age': 'int',
                    'name': 'str',
                    'score': 'float',
                    'is_active': 'bool',
                    'birth_date': 'date'
                }
            }
        raise_error: Whether to raise an error if validation fails.
        
    Returns:
        A dictionary with validation results for each check.
        
    Raises:
        ExcelUtilError: If raise_error is True and validation fails.
    """
    logger = get_logger("excel_utils")
    
    # Initialize result dictionary
    result = {}
    
    # Perform validation checks
    try:
        # Check missing values
        if 'missing_values' in validation_config:
            config = validation_config['missing_values']
            columns = config.get('columns')
            threshold = config.get('threshold', 0.0)
            
            result['missing_values'] = check_missing_values(
                df=df,
                columns=columns,
                threshold=threshold,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check duplicate rows
        if 'duplicate_rows' in validation_config:
            config = validation_config['duplicate_rows']
            subset = config.get('subset')
            
            result['duplicate_rows'] = check_duplicate_rows(
                df=df,
                subset=subset,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check value ranges
        if 'value_ranges' in validation_config:
            ranges = validation_config['value_ranges']
            
            result['value_ranges'] = check_value_ranges(
                df=df,
                ranges=ranges,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check data types
        if 'data_types' in validation_config:
            type_specs = validation_config['data_types']
            
            result['data_types'] = check_data_types(
                df=df,
                type_specs=type_specs,
                raise_error=False  # Don't raise error here, we'll handle it later
            )
        
        # Check if validation failed
        validation_failed = False
        error_msg = "Validation failed:\n"
        
        # Check missing values
        if 'missing_values' in result:
            missing_values = result['missing_values']
            threshold = validation_config['missing_values'].get('threshold', 0.0)
            
            if any(pct > threshold for pct in missing_values.values()):
                validation_failed = True
                error_msg += "Missing values check failed:\n"
                for col, pct in missing_values.items():
                    if pct > threshold:
                        error_msg += f"  - {col}: {pct * 100:.2f}%\n"
        
        # Check duplicate rows
        if 'duplicate_rows' in result:
            duplicate_rows = result['duplicate_rows']
            
            if duplicate_rows['duplicate_count'] > 0:
                validation_failed = True
                subset = validation_config['duplicate_rows'].get('subset')
                
                if subset:
                    error_msg += f"Duplicate rows check failed: Found {duplicate_rows['duplicate_count']} duplicate rows based on columns: {subset}\n"
                else:
                    error_msg += f"Duplicate rows check failed: Found {duplicate_rows['duplicate_count']} duplicate rows\n"
        
        # Check value ranges
        if 'value_ranges' in result:
            value_ranges = result['value_ranges']
            
            if any(res['total_outside_range'] > 0 for res in value_ranges.values()):
                validation_failed = True
                error_msg += "Value ranges check failed:\n"
                
                for col, res in value_ranges.items():
                    if res['total_outside_range'] > 0:
                        range_spec = validation_config['value_ranges'][col]
                        min_val = range_spec.get('min', 'None')
                        max_val = range_spec.get('max', 'None')
                        error_msg += f"  - {col} (range: {min_val} to {max_val}):\n"
                        error_msg += f"    - Below minimum: {res['below_min']}\n"
                        error_msg += f"    - Above maximum: {res['above_max']}\n"
        
        # Check data types
        if 'data_types' in result:
            data_types = result['data_types']
            
            if any(res['error_count'] > 0 for res in data_types.values()):
                validation_failed = True
                error_msg += "Data types check failed:\n"
                
                for col, res in data_types.items():
                    if res['error_count'] > 0:
                        error_msg += f"  - {col} (expected: {res['expected_type']}, current: {res['current_type']}):\n"
                        error_msg += f"    - Error count: {res['error_count']}\n"
        
        # Raise error if validation failed and raise_error is True
        if validation_failed and raise_error:
            logger.error(error_msg)
            raise ExcelUtilError(error_msg)
        
        return result
    
    except Exception as e:
        if not isinstance(e, ExcelUtilError):
            error_msg = f"Error during validation: {str(e)}"
            logger.error(error_msg)
            if raise_error:
                raise ExcelUtilError(error_msg) from e
        else:
            raise
        
        return result