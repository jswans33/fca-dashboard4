"""
Conversion utility module for Excel operations.

This module provides utilities for converting Excel files to other formats,
merging Excel files, and saving Excel data to databases.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.sheet_utils import get_sheet_names
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


def convert_excel_to_csv(
    input_file: Union[str, Path],
    output_file: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> None:
    """
    Convert an Excel file to CSV.
    
    Args:
        input_file: Path to the input Excel file.
        output_file: Path to the output CSV file.
        sheet_name: Name or index of the sheet to convert (default: 0, first sheet).
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv().
        
    Raises:
        FileNotFoundError: If the input file does not exist.
        ExcelUtilError: If the sheet does not exist or an error occurs during conversion.
    """
    logger = get_logger("excel_utils")
    
    # Resolve the file paths
    input_path = resolve_path(input_file)
    output_path = Path(output_file)  # Don't resolve output path as it may not exist yet
    
    # Check if input file exists
    if not input_path.is_file():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Read the Excel file
        if input_path.suffix.lower() == ".csv":
            # If the input is already a CSV, just copy it
            df = pd.read_csv(input_path)
        else:
            # Check if the sheet exists
            sheet_names = get_sheet_names(input_path)
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {input_path}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            # Read the Excel file
            df = pd.read_excel(input_path, sheet_name=actual_sheet_name)
        
        # Write to CSV
        df.to_csv(output_path, index=False, **kwargs)
        logger.info(f"Converted {input_path} to {output_path}")
    except ExcelUtilError:
        # Re-raise ExcelUtilError
        raise
    except Exception as e:
        error_msg = f"Error converting {input_path} to CSV: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def merge_excel_files(
    input_files: List[Union[str, Path]],
    output_file: Union[str, Path],
    sheet_name: Union[str, int] = 0,
    **kwargs
) -> None:
    """
    Merge multiple Excel files into a single Excel file.
    
    Args:
        input_files: List of paths to the input Excel files.
        output_file: Path to the output Excel file.
        sheet_name: Name or index of the sheet to read from each input file (default: 0, first sheet).
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_excel().
        
    Raises:
        ValueError: If the input file list is empty.
        FileNotFoundError: If any input file does not exist.
        ExcelUtilError: If the files have different columns or an error occurs during merging.
    """
    logger = get_logger("excel_utils")
    
    # Check if input file list is empty
    if not input_files:
        error_msg = "Input file list is empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Resolve the file paths
    input_paths = [resolve_path(file) for file in input_files]
    output_path = Path(output_file)  # Don't resolve output path as it may not exist yet
    
    # Check if all input files exist
    for path in input_paths:
        if not path.is_file():
            logger.error(f"Input file not found: {path}")
            raise FileNotFoundError(f"Input file not found: {path}")
    
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Read the first file to get the columns
        if input_paths[0].suffix.lower() == ".csv":
            first_df = pd.read_csv(input_paths[0])
        else:
            # Handle sheet name for the first file
            sheet_names = get_sheet_names(input_paths[0])
            
            # Handle sheet name as index or name
            if isinstance(sheet_name, int) and sheet_name < len(sheet_names):
                # If sheet_name is an integer index, use it to get the actual sheet name
                actual_sheet_name = sheet_names[sheet_name]
            elif sheet_name in sheet_names:
                # If sheet_name is already a valid sheet name, use it directly
                actual_sheet_name = sheet_name
            else:
                # If sheet_name is neither a valid index nor a valid name, raise an error
                error_msg = f"Sheet '{sheet_name}' not found in {input_paths[0]}. Available sheets: {sheet_names}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            first_df = pd.read_excel(input_paths[0], sheet_name=actual_sheet_name)
        
        first_columns = list(first_df.columns)
        
        # Initialize the merged DataFrame with the first file
        merged_df = first_df.copy()
        
        # Merge the rest of the files
        for path in input_paths[1:]:
            # Read the file
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                # Handle sheet name for each file
                file_sheet_names = get_sheet_names(path)
                
                # Handle sheet name as index or name
                if isinstance(sheet_name, int) and sheet_name < len(file_sheet_names):
                    # If sheet_name is an integer index, use it to get the actual sheet name
                    file_actual_sheet_name = file_sheet_names[sheet_name]
                elif sheet_name in file_sheet_names:
                    # If sheet_name is already a valid sheet name, use it directly
                    file_actual_sheet_name = sheet_name
                else:
                    # If sheet_name is neither a valid index nor a valid name, raise an error
                    error_msg = f"Sheet '{sheet_name}' not found in {path}. Available sheets: {file_sheet_names}"
                    logger.error(error_msg)
                    raise ExcelUtilError(error_msg)
                
                df = pd.read_excel(path, sheet_name=file_actual_sheet_name)
            
            # Check if the columns match
            if list(df.columns) != first_columns:
                error_msg = f"Columns in {path} do not match the columns in {input_paths[0]}"
                logger.error(error_msg)
                raise ExcelUtilError(error_msg)
            
            # Append the data
            merged_df = pd.concat([merged_df, df], ignore_index=True)
        
        # Write to Excel
        merged_df.to_excel(output_path, index=False, **kwargs)
        logger.info(f"Merged {len(input_paths)} files into {output_path}")
    except Exception as e:
        error_msg = f"Error merging Excel files: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def save_excel_to_database(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: Optional[str] = None,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a database table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The database connection string.
        schema: The database schema (optional).
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
        
    Raises:
        ExcelUtilError: If an error occurs while saving to the database.
    """
    logger = get_logger("excel_utils")
    
    try:
        import sqlalchemy
        
        # Create the engine
        engine = sqlalchemy.create_engine(connection_string)
        
        # Save the DataFrame to the database
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=index,
            **kwargs
        )
        
        logger.info(f"Successfully saved {len(df)} rows to table {table_name}")
    except Exception as e:
        error_msg = f"Error saving DataFrame to database table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e


def get_database_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a database table.
    
    Args:
        connection_string: The SQLAlchemy connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        ExcelUtilError: If an error occurs while getting the schema.
    """
    logger = get_logger("excel_utils")
    
    try:
        import sqlalchemy
        
        # Create the engine
        engine = sqlalchemy.create_engine(connection_string)
        
        # Get the schema of the table
        with engine.connect() as conn:
            # For SQLite
            if connection_string.startswith('sqlite'):
                from sqlalchemy import text
                result = conn.execute(text(f"PRAGMA table_info({table_name})"))
                columns = []
                for row in result:
                    columns.append(f"{row[1]} {row[2]}")
                schema = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
            # For PostgreSQL
            elif connection_string.startswith('postgresql'):
                from sqlalchemy import text
                result = conn.execute(text(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"))
                columns = []
                for row in result:
                    columns.append(f"{row[0]} {row[1]}")
                schema = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
            # For MySQL
            elif connection_string.startswith('mysql'):
                from sqlalchemy import text
                result = conn.execute(text(f"DESCRIBE {table_name}"))
                columns = []
                for row in result:
                    columns.append(f"{row[0]} {row[1]}")
                schema = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
            else:
                schema = f"Schema for {table_name} not available for this database type"
        
        logger.info(f"Successfully retrieved schema for table {table_name}")
        return schema
    except Exception as e:
        error_msg = f"Error getting schema for table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise ExcelUtilError(error_msg) from e