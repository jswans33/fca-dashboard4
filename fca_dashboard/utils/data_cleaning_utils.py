"""
Data cleaning utility module for the FCA Dashboard application.

This module provides utilities for cleaning and preprocessing data,
particularly for Excel data extraction and transformation.
"""

from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from fca_dashboard.utils.logging_config import get_logger


class DataCleaningError(Exception):
    """Exception raised for errors in the data cleaning process."""
    pass


def find_header_row(df: pd.DataFrame, header_patterns: Optional[List[str]] = None) -> Optional[int]:
    """
    Find the header row in a DataFrame based on patterns.
    
    Args:
        df: The DataFrame to search in.
        header_patterns: List of patterns to look for in the header row.
            If None, uses default patterns like 'omniclass number', 'number', 'code'.
            
    Returns:
        The index of the header row, or None if not found.
    """
    logger = get_logger("data_cleaning_utils")
    
    if header_patterns is None:
        header_patterns = ['omniclass number', 'number', 'code', 'title']
    
    # Convert patterns to lowercase for case-insensitive matching
    header_patterns = [pattern.lower() for pattern in header_patterns]
    
    # Log the patterns we're looking for
    logger.info(f"Looking for header patterns: {header_patterns}")
    
    # Search for header row
    for idx, row in df.iterrows():
        # Convert row values to strings and check if any contain the patterns
        row_values = [str(val).lower() for val in row.values]
        
        # Log the row values for debugging
        logger.info(f"Row {idx} values: {row_values}")
        
        # Check if any pattern is in any row value
        matches = [pattern for pattern in header_patterns if any(pattern in value for value in row_values)]
        if matches:
            logger.info(f"Found header row at index {idx} with matching patterns: {matches}")
            return idx
    
    logger.warning("No header row found")
    return None


def remove_copyright_rows(df: pd.DataFrame, patterns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove copyright and other non-data rows from a DataFrame.
    
    Args:
        df: The DataFrame to clean.
        patterns: List of patterns to identify copyright rows.
            If None, uses default patterns.
            
    Returns:
        A new DataFrame with copyright rows removed.
    """
    logger = get_logger("data_cleaning_utils")
    
    if patterns is None:
        patterns = [
            'copyright', 'all rights reserved', 'omniclass®',
            'final', 'national standard', 'consensus candidate'
        ]
    
    # Convert patterns to lowercase for case-insensitive matching
    patterns = [pattern.lower() for pattern in patterns]
    
    # Create a mask for rows to keep
    mask = pd.Series(True, index=df.index)
    
    # Check each row for copyright patterns
    for idx, row in df.iterrows():
        # Convert row values to strings and check if any contain the patterns
        row_values = [str(val).lower() for val in row.values]
        
        # Check if any pattern is in any row value
        if any(pattern in value for pattern in patterns for value in row_values):
            mask.at[idx] = False
    
    # Apply the mask to get the cleaned DataFrame
    cleaned_df = df[mask].copy()
    
    removed_count = len(df) - len(cleaned_df)
    logger.debug(f"Removed {removed_count} copyright rows")
    
    return cleaned_df


def standardize_column_names(
    df: pd.DataFrame,
    column_mapping: Optional[Dict[str, str]] = None
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
    logger = get_logger("data_cleaning_utils")
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Log the current column names
    logger.info(f"Current column names: {result_df.columns.tolist()}")
    
    # Default column mapping
    default_mapping = {
        'OmniClass Number': 'OmniClass_Code',
        'Number': 'OmniClass_Code',
        'OmniClass Title': 'OmniClass_Title',
        'Title': 'OmniClass_Title',
        'Definition': 'Description'
    }
    
    # Use provided mapping or default
    mapping = column_mapping if column_mapping is not None else default_mapping
    
    # Log the mapping being used
    logger.info(f"Using column mapping: {mapping}")
    
    # Only map columns that exist in the DataFrame
    valid_mapping = {k: v for k, v in mapping.items() if k in result_df.columns}
    
    # Log the valid mapping
    logger.info(f"Valid column mapping: {valid_mapping}")
    
    if valid_mapping:
        result_df = result_df.rename(columns=valid_mapping)
        logger.info(f"Renamed columns: {valid_mapping}")
    else:
        logger.warning("No columns matched the mapping")
        
        # Try case-insensitive matching as a fallback
        case_insensitive_mapping = {}
        for col in result_df.columns:
            for k, v in mapping.items():
                if k.lower() == col.lower():
                    case_insensitive_mapping[col] = v
                    break
        
        if case_insensitive_mapping:
            logger.info(f"Found case-insensitive matches: {case_insensitive_mapping}")
            result_df = result_df.rename(columns=case_insensitive_mapping)
            logger.info(f"Renamed columns using case-insensitive matching: {case_insensitive_mapping}")
    
    # Log the final column names
    logger.info(f"Final column names: {result_df.columns.tolist()}")
    
    return result_df


def clean_dataframe(
    df: pd.DataFrame,
    header_patterns: Optional[List[str]] = None,
    copyright_patterns: Optional[List[str]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    is_omniclass: bool = False
) -> pd.DataFrame:
    """
    Clean a DataFrame by removing copyright rows, identifying the header row,
    and standardizing column names.
    
    Args:
        df: The DataFrame to clean.
        header_patterns: List of patterns to identify the header row.
        copyright_patterns: List of patterns to identify copyright rows.
        column_mapping: Dictionary mapping original column names to standardized names.
        is_omniclass: Whether the DataFrame contains OmniClass data, which requires special handling.
            
    Returns:
        A cleaned DataFrame.
        
    Raises:
        DataCleaningError: If no header row is found or other cleaning errors occur.
    """
    logger = get_logger("data_cleaning_utils")
    
    try:
        # Step 1: Remove copyright rows
        df_no_copyright = remove_copyright_rows(df, patterns=copyright_patterns)
        
        # Step 2: Find the header row
        header_idx = find_header_row(df_no_copyright, header_patterns=header_patterns)
        
        if header_idx is None:
            raise DataCleaningError("No header row found in the DataFrame")
        
        # Step 3: Use the header row as column names and keep only data rows
        header_values = df_no_copyright.iloc[header_idx].values
        data_df = df_no_copyright.iloc[header_idx + 1:].copy()
        
        # Special handling for OmniClass headers
        if is_omniclass:
            # Clean the header values to handle OmniClass format
            cleaned_headers = []
            
            # Log the original header values
            logger.info(f"Original header values: {header_values}")
            
            for val in header_values:
                # Convert to string and strip whitespace
                val_str = str(val).strip()
                
                # Log the header value being processed
                logger.info(f"Processing header value: '{val_str}'")
                
                # Check if the value looks like an OmniClass code (e.g., "11-11 00 00")
                if val_str and "-" in val_str and " " in val_str:
                    # Keep only the OmniClass code part
                    cleaned_val = val_str.split(",")[0] if "," in val_str else val_str
                    logger.info(f"Identified as OmniClass code: '{val_str}' -> '{cleaned_val}'")
                    cleaned_headers.append(cleaned_val)
                else:
                    logger.info(f"Not an OmniClass code, keeping as is: '{val_str}'")
                    cleaned_headers.append(val_str)
            
            # Log the cleaned headers
            logger.info(f"Cleaned headers: {cleaned_headers}")
            
            data_df.columns = cleaned_headers
        else:
            logger.info(f"Not OmniClass data, using original header values: {header_values}")
            data_df.columns = header_values
        
        # Step 4: Reset the index
        data_df = data_df.reset_index(drop=True)
        
        # Step 5: Standardize column names
        result_df = standardize_column_names(data_df, column_mapping=column_mapping)
        
        # Step 6: Remove any remaining empty rows
        result_df = result_df.dropna(how='all')
        
        logger.info(f"Successfully cleaned DataFrame, resulting in {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        error_msg = f"Error cleaning DataFrame: {str(e)}"
        logger.error(error_msg)
        raise DataCleaningError(error_msg) from e