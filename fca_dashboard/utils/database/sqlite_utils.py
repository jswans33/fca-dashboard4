"""
SQLite utilities for the FCA Dashboard application.

This module provides utilities for working with SQLite databases.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.logging_config import get_logger


def save_dataframe_to_sqlite(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a SQLite table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The SQLite connection string.
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    """
    logger = get_logger("sqlite_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Save the DataFrame to the database
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists=if_exists,
        index=index,
        **kwargs
    )
    
    logger.info(f"Successfully saved {len(df)} rows to SQLite table {table_name}")


def get_sqlite_table_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a SQLite table.
    
    Args:
        connection_string: The SQLite connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        Exception: If the table does not exist.
    """
    logger = get_logger("sqlite_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Check if the table exists
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"))
        if not result.fetchone():
            error_msg = f"Table '{table_name}' does not exist in the database"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Get the schema of the table
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = []
        for row in result:
            columns.append(f"{row[1]} {row[2]}")
        
        if not columns:
            error_msg = f"Table '{table_name}' exists but has no columns"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        schema = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    
    logger.info(f"Successfully retrieved schema for SQLite table {table_name}")
    return schema