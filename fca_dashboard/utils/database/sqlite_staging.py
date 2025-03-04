"""
SQLite staging utilities for the FCA Dashboard application.

This module provides utilities for working with the SQLite staging database.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database.base import DatabaseError
from fca_dashboard.utils.logging_config import get_logger

logger = get_logger("sqlite_staging")


def initialize_sqlite_staging_db(db_path: str) -> None:
    """
    Initialize the SQLite staging database by executing the schema SQL file.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Raises:
        DatabaseError: If an error occurs while initializing the database.
    """
    try:
        # Get the path to the schema file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        schema_path = os.path.join(
            current_dir, 
            '..', '..', 'db', 'staging', 'schema', 'staging_schema_sqlite.sql'
        )
        
        # Read the schema file
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Connect to the database and execute the schema
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully initialized SQLite staging database at {db_path}")
    except Exception as e:
        error_msg = f"Error initializing SQLite staging database: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def reset_error_items(connection_string: str) -> int:
    """
    Reset the status of error items to 'PENDING'.
    
    Args:
        connection_string: The SQLite connection string.
    
    Returns:
        The number of rows updated.
    
    Raises:
        DatabaseError: If an error occurs while resetting error items.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Reset error items
        with engine.connect() as conn:
            result = conn.execute(text("""
                UPDATE equipment_staging
                SET processing_status = 'PENDING',
                    error_message = NULL
                WHERE processing_status = 'ERROR'
            """))
            conn.commit()
            updated_count = result.rowcount
        
        logger.info(f"Reset {updated_count} error items to 'PENDING'")
        return updated_count
    except Exception as e:
        error_msg = f"Error resetting error items: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def clear_processed_items(connection_string: str, days_to_keep: int = 7) -> int:
    """
    Clear processed items from the staging table that are older than the specified number of days.
    
    Args:
        connection_string: The SQLite connection string.
        days_to_keep: The number of days to keep processed items (default: 7).
    
    Returns:
        The number of rows deleted.
    
    Raises:
        DatabaseError: If an error occurs while clearing processed items.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Calculate the cutoff date
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
        
        # Clear processed items
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                DELETE FROM equipment_staging
                WHERE is_processed = 1
                AND processed_timestamp < '{cutoff_date}'
            """))
            conn.commit()
            deleted_count = result.rowcount
        
        logger.info(f"Cleared {deleted_count} processed items older than {days_to_keep} days")
        return deleted_count
    except Exception as e:
        error_msg = f"Error clearing processed items: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def save_dataframe_to_staging(
    df: pd.DataFrame,
    connection_string: str,
    source_system: str,
    import_batch_id: str,
    **kwargs
) -> None:
    """
    Save a DataFrame to the SQLite staging table.
    
    Args:
        df: The DataFrame to save.
        connection_string: The SQLite connection string.
        source_system: The source system identifier.
        import_batch_id: The import batch identifier.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    
    Raises:
        DatabaseError: If an error occurs while saving to the staging table.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Add metadata columns
        df['source_system'] = source_system
        df['import_batch_id'] = import_batch_id
        df['import_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['processing_status'] = 'PENDING'
        df['is_processed'] = 0
        
        # Convert any JSON columns to strings
        for col in ['attributes', 'maintenance_data', 'project_data', 
                   'document_data', 'qc_data', 'raw_source_data']:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
        
        # Save the DataFrame to the staging table
        df.to_sql(
            name='equipment_staging',
            con=engine,
            if_exists='append',
            index=False,
            **kwargs
        )
        
        logger.info(f"Successfully saved {len(df)} rows to SQLite staging table")
    except Exception as e:
        error_msg = f"Error saving DataFrame to SQLite staging table: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def get_pending_items(connection_string: str) -> pd.DataFrame:
    """
    Get pending items from the staging table.
    
    Args:
        connection_string: The SQLite connection string.
    
    Returns:
        A DataFrame containing the pending items.
    
    Raises:
        DatabaseError: If an error occurs while getting pending items.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Get pending items
        query = "SELECT * FROM v_pending_items"
        df = pd.read_sql(query, engine)
        
        # Convert JSON strings back to dictionaries
        for col in ['attributes', 'maintenance_data', 'project_data', 
                   'document_data', 'qc_data', 'raw_source_data']:
            if col in df.columns and df[col].notna().any():
                df[col] = df[col].apply(lambda x: json.loads(x) if x is not None else None)
        
        logger.info(f"Retrieved {len(df)} pending items from staging table")
        return df
    except Exception as e:
        error_msg = f"Error getting pending items: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e


def update_item_status(
    connection_string: str,
    staging_id: int,
    status: str,
    error_message: Optional[str] = None,
    is_processed: Optional[bool] = None
) -> None:
    """
    Update the status of an item in the staging table.
    
    Args:
        connection_string: The SQLite connection string.
        staging_id: The ID of the item to update.
        status: The new status ('PENDING', 'PROCESSING', 'COMPLETED', 'ERROR').
        error_message: The error message (if status is 'ERROR').
        is_processed: Whether the item has been processed.
    
    Raises:
        DatabaseError: If an error occurs while updating the item status.
    """
    try:
        # Create a SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Build the update query
        query = f"""
            UPDATE equipment_staging
            SET processing_status = '{status}'
        """
        
        if error_message is not None:
            error_message_escaped = error_message.replace("'", "''")
            query += f", error_message = '{error_message_escaped}'"
        
        if is_processed is not None:
            query += f", is_processed = {1 if is_processed else 0}"
            if is_processed:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                query += f", processed_timestamp = '{timestamp}'"
        
        query += f" WHERE staging_id = {staging_id}"
        
        # Execute the update
        with engine.connect() as conn:
            conn.execute(text(query))
            conn.commit()
        
        logger.info(f"Updated status of item {staging_id} to '{status}'")
    except Exception as e:
        error_msg = f"Error updating item status: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg) from e