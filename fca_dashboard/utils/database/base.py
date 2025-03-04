"""
Base database utilities for the FCA Dashboard application.

This module provides base classes and utilities for database operations.
"""

from typing import Dict, Optional, Union

import pandas as pd

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger


class DatabaseError(FCADashboardError):
    """Base exception for database operations."""
    pass


def save_dataframe_to_database(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: Optional[str] = None,
    if_exists: str = "replace",
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
        **kwargs: Additional arguments to pass to the database-specific implementation.
        
    Raises:
        DatabaseError: If an error occurs while saving to the database.
    """
    logger = get_logger("database_utils")
    
    try:
        # Determine the database type from the connection string
        if connection_string.startswith('sqlite'):
            from fca_dashboard.utils.database.sqlite_utils import save_dataframe_to_sqlite
            save_dataframe_to_sqlite(df, table_name, connection_string, if_exists, **kwargs)
        elif connection_string.startswith('postgresql'):
            from fca_dashboard.utils.database.postgres_utils import save_dataframe_to_postgres
            save_dataframe_to_postgres(df, table_name, connection_string, schema, if_exists, **kwargs)
        else:
            raise DatabaseError(f"Unsupported database type: {connection_string}")
            
        logger.info(f"Successfully saved {len(df)} rows to table {table_name}")
    except Exception as e:
        error_msg = f"Error saving DataFrame to database table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def get_table_schema(
    connection_string: str,
    table_name: str
) -> str:
    """
    Get the schema of a database table.
    
    Args:
        connection_string: The database connection string.
        table_name: The name of the table to get the schema for.
        
    Returns:
        A string containing the schema of the table.
        
    Raises:
        DatabaseError: If an error occurs while getting the schema.
    """
    logger = get_logger("database_utils")
    
    try:
        # Determine the database type from the connection string
        if connection_string.startswith('sqlite'):
            from fca_dashboard.utils.database.sqlite_utils import get_sqlite_table_schema
            return get_sqlite_table_schema(connection_string, table_name)
        elif connection_string.startswith('postgresql'):
            from fca_dashboard.utils.database.postgres_utils import get_postgres_table_schema
            return get_postgres_table_schema(connection_string, table_name)
        else:
            return f"Schema for {table_name} not available for this database type"
    except Exception as e:
        error_msg = f"Error getting schema for table {table_name}: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e