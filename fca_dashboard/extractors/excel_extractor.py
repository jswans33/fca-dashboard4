"""
Excel extractor module for the FCA Dashboard application.

This module provides functionality for extracting data from Excel files
and loading it into pandas DataFrames, with features like error handling,
logging, and optional file upload.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path
from fca_dashboard.utils.upload_util import upload_file


class ExcelExtractionError(FCADashboardError):
    """Exception raised for errors during Excel data extraction."""
    pass


def extract_excel_to_dataframe(
    file_path: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    columns: Optional[List[str]] = None,
    upload: bool = False,
    target_filename: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extract data from an Excel file into a pandas DataFrame.
    
    Args:
        file_path: Path to the Excel file. Can be absolute or relative.
        sheet_name: Name or index of the sheet to extract (default: 0, first sheet).
        columns: Optional list of column names to extract. If None, extracts all columns.
        upload: Whether to upload the file to the uploads directory.
        target_filename: Optional filename to use when uploading. If None, uses the source filename.
        
    Returns:
        pandas DataFrame containing the extracted data.
        
    Raises:
        FileNotFoundError: If the source file does not exist.
        ExcelExtractionError: If an error occurs during the extraction process.
    """
    logger = get_logger("excel_extractor")
    
    # Resolve the file path
    source_path = resolve_path(file_path)
    
    # Validate source file
    if not source_path.is_file():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Log the extraction operation
    logger.info(f"Extracting data from Excel file: {source_path}")
    
    try:
        # Read the Excel file into a DataFrame
        df = pd.read_excel(source_path, sheet_name=sheet_name)
        
        # Filter columns if specified
        if columns:
            # Validate that all requested columns exist
            missing_columns = [col for col in columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Columns not found in Excel file: {missing_columns}"
                logger.error(error_msg)
                raise ExcelExtractionError(error_msg)
            
            # Select only the specified columns
            df = df[columns]
        
        # Upload the file if requested
        if upload:
            # Get the uploads directory from settings
            uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
            
            # Ensure the uploads directory exists
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Upload the file
            logger.info(f"Uploading Excel file to: {uploads_dir}")
            upload_file(source_path, uploads_dir, target_filename)
        
        # Log success and return the DataFrame
        logger.info(f"Successfully extracted {len(df)} rows from {source_path}")
        return df
    
    except (EmptyDataError, ParserError) as e:
        # Handle pandas-specific errors
        error_msg = f"Error parsing Excel file {source_path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelExtractionError(error_msg) from e
    
    except Exception as e:
        # Handle any other errors
        error_msg = f"Error extracting data from Excel file {source_path}: {str(e)}"
        logger.error(error_msg)
        raise ExcelExtractionError(error_msg) from e