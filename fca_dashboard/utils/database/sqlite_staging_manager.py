"""
SQLite staging manager for the FCA Dashboard application.

This module provides a class-based implementation of the SQLite staging utilities
with proper dependency injection.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database.base import DatabaseError
from fca_dashboard.utils.logging_config import get_logger


class SQLiteStagingManager:
    """
    Manager for SQLite staging database operations.
    
    This class provides methods for initializing and interacting with
    a SQLite staging database, with support for dependency injection.
    """
    
    def __init__(
        self,
        logger=None,
        connection_factory=None,
        schema_path=None
    ):
        """
        Initialize the SQLite staging manager.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
            connection_factory: Optional factory function for creating database connections.
                If None, a default SQLAlchemy engine will be used.
            schema_path: Optional path to the SQLite schema file.
                If None, the default schema path will be used.
        """
        self.logger = logger or get_logger("sqlite_staging")
        self.connection_factory = connection_factory or self._default_connection_factory
        
        if schema_path:
            self.schema_path = schema_path
        else:
            # Try to get the schema path from settings
            from fca_dashboard.config.settings import settings
            settings_schema_path = settings.get("databases.staging.schema_path")
            
            if settings_schema_path:
                # Resolve the path relative to the project root
                from fca_dashboard.utils.path_util import resolve_path
                self.schema_path = resolve_path(settings_schema_path)
            else:
                # Use the default schema path
                current_dir = os.path.dirname(os.path.abspath(__file__))
                self.schema_path = os.path.join(
                    current_dir,
                    '..', '..', 'db', 'staging', 'schema', 'staging_schema_sqlite.sql'
                )
    
    def _default_connection_factory(self, connection_string: str):
        """
        Default factory function for creating database connections.
        
        Args:
            connection_string: The SQLite connection string.
            
        Returns:
            A SQLAlchemy engine.
        """
        return create_engine(connection_string)
    
    def initialize_db(self, db_path: str) -> None:
        """
        Initialize the SQLite staging database by executing the schema SQL file.
        
        Args:
            db_path: Path to the SQLite database file.
        
        Raises:
            DatabaseError: If an error occurs while initializing the database.
        """
        try:
            # Read the schema file
            with open(self.schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Connect to the database and execute the schema
            conn = sqlite3.connect(db_path)
            conn.executescript(schema_sql)
            conn.commit()
            conn.close()
            
            self.logger.info(f"Successfully initialized SQLite staging database at {db_path}")
        except Exception as e:
            error_msg = f"Error initializing SQLite staging database: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def reset_error_items(self, connection_string: str) -> int:
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
            # Create a database engine
            engine = self.connection_factory(connection_string)
            
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
            
            self.logger.info(f"Reset {updated_count} error items to 'PENDING'")
            return updated_count
        except Exception as e:
            error_msg = f"Error resetting error items: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def clear_processed_items(self, connection_string: str, days_to_keep: int = None) -> int:
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
            # If days_to_keep is not provided, try to get it from settings
            if days_to_keep is None:
                from fca_dashboard.config.settings import settings
                days_to_keep = settings.get("databases.staging.processed_retention_days", 7)
            
            # Create a database engine
            engine = self.connection_factory(connection_string)
            
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
            
            self.logger.info(f"Cleared {deleted_count} processed items older than {days_to_keep} days")
            return deleted_count
        except Exception as e:
            error_msg = f"Error clearing processed items: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def save_dataframe_to_staging(
        self,
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
            ValueError: If the DataFrame is None or empty.
        """
        if df is None:
            error_msg = "Cannot save None DataFrame to staging table"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        if df.empty:
            self.logger.warning("DataFrame is empty, no rows will be saved to staging table")
            return
            
        try:
            # Create a database engine
            engine = self.connection_factory(connection_string)
            
            # Check if the DataFrame is already mapped (has been processed by a mapper)
            if isinstance(df, pd.DataFrame) and any(col in df.columns for col in ['equipment_type', 'equipment_tag', 'category_name']):
                self.logger.info("DataFrame appears to be already mapped, using as-is")
                staging_df = df.copy()
            else:
                # Use the mapper factory to get the appropriate mapper
                try:
                    from fca_dashboard.mappers.mapper_factory import mapper_factory
                    mapper = mapper_factory.create_mapper(source_system, self.logger)
                    
                    # Map the DataFrame using the mapper
                    self.logger.info(f"Mapping DataFrame using {source_system} mapper")
                    staging_df = mapper.map_dataframe(df)
                    self.logger.info(f"Successfully mapped DataFrame with columns: {list(staging_df.columns)}")
                except ImportError as e:
                    # If mapper module is not available, use the old approach
                    self.logger.warning(f"Mapper module not available: {str(e)}, using direct column mapping")
                    staging_df = df.copy()
                    
                    # Fix column names by replacing spaces with underscores
                    staging_df.columns = [col.replace(' ', '_') if isinstance(col, str) else col for col in staging_df.columns]
                    self.logger.info(f"Normalized column names: {list(staging_df.columns)}")
                except Exception as e:
                    self.logger.error(f"Error using mapper: {str(e)}")
                    self.logger.warning("Falling back to direct column mapping")
                    staging_df = df.copy()
                    
                    # Fix column names by replacing spaces with underscores
                    staging_df.columns = [col.replace(' ', '_') if isinstance(col, str) else col for col in staging_df.columns]
                    self.logger.info(f"Normalized column names: {list(staging_df.columns)}")
            
            # Validate that the DataFrame has at least some required columns
            required_columns = ['equipment_tag', 'equipment_type', 'serial_number']
            missing_required = [col for col in required_columns if col not in staging_df.columns]
            if missing_required:
                self.logger.warning(f"DataFrame is missing some recommended columns: {missing_required}")
            
            # Add metadata columns
            staging_df['source_system'] = source_system
            staging_df['import_batch_id'] = import_batch_id
            staging_df['import_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            staging_df['processing_status'] = 'PENDING'
            staging_df['is_processed'] = 0
            
            # Convert any JSON columns to strings
            for col in ['attributes', 'maintenance_data', 'project_data',
                       'document_data', 'qc_data', 'raw_source_data']:
                if col in staging_df.columns and staging_df[col].notna().any():
                    staging_df[col] = staging_df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            
            # Get a list of valid columns in the equipment_staging table
            with engine.connect() as conn:
                result = conn.execute(text("PRAGMA table_info(equipment_staging)"))
                valid_columns = [row[1] for row in result.fetchall()]
            
            self.logger.info(f"Valid columns in equipment_staging table: {valid_columns}")
            
            # Log all columns in the mapped DataFrame
            self.logger.info(f"Mapped DataFrame columns: {list(staging_df.columns)}")
            
            # Check for columns that will be filtered out
            filtered_out_columns = [col for col in staging_df.columns if col not in valid_columns]
            if filtered_out_columns:
                self.logger.warning(f"The following columns will be filtered out because they don't exist in the staging table: {filtered_out_columns}")
                
                # Check if any critical columns are being filtered out
                critical_columns = ['equipment_tag', 'equipment_type', 'serial_number']
                critical_filtered = [col for col in critical_columns if col in filtered_out_columns]
                if critical_filtered:
                    error_msg = f"Critical columns are being filtered out: {critical_filtered}. This may cause issues with downstream processing."
                    self.logger.error(error_msg)
            
            # Filter out columns that don't exist in the table
            valid_staging_df = staging_df[[col for col in staging_df.columns if col in valid_columns]]
            
            # Log the columns that will be saved to the staging table
            self.logger.info(f"Columns that will be saved to staging table: {list(valid_staging_df.columns)}")
            
            # Verify we have at least some data to save
            if valid_staging_df.empty:
                self.logger.warning("After filtering, no valid columns remain. No data will be saved.")
                return
                
            # Save the DataFrame to the staging table
            valid_staging_df.to_sql(
                name='equipment_staging',
                con=engine,
                if_exists='append',
                index=False,
                **kwargs
            )
            
            self.logger.info(f"Successfully saved {len(valid_staging_df)} rows to SQLite staging table")
        except Exception as e:
            error_msg = f"Error saving DataFrame to SQLite staging table: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def get_pending_items(self, connection_string: str) -> pd.DataFrame:
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
            # Create a database engine
            engine = self.connection_factory(connection_string)
            
            # Get pending items
            query = "SELECT * FROM v_pending_items"
            df = pd.read_sql(query, engine)
            
            # Convert JSON strings back to dictionaries
            for col in ['attributes', 'maintenance_data', 'project_data', 
                       'document_data', 'qc_data', 'raw_source_data']:
                if col in df.columns and df[col].notna().any():
                    df[col] = df[col].apply(lambda x: json.loads(x) if x is not None else None)
            
            self.logger.info(f"Retrieved {len(df)} pending items from staging table")
            return df
        except Exception as e:
            error_msg = f"Error getting pending items: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e
    
    def update_item_status(
        self,
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
            # Create a database engine
            engine = self.connection_factory(connection_string)
            
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
            
            self.logger.info(f"Updated status of item {staging_id} to '{status}'")
        except Exception as e:
            error_msg = f"Error updating item status: {str(e)}"
            self.logger.error(error_msg)
            raise DatabaseError(error_msg) from e


# For backward compatibility
_default_manager = SQLiteStagingManager()

def initialize_sqlite_staging_db(db_path: str) -> None:
    """
    Initialize the SQLite staging database by executing the schema SQL file.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Raises:
        DatabaseError: If an error occurs while initializing the database.
    """
    return _default_manager.initialize_db(db_path)

def reset_error_items(connection_string: str) -> int:
    """
    Reset the status of error items to 'PENDING'.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        connection_string: The SQLite connection string.
    
    Returns:
        The number of rows updated.
    
    Raises:
        DatabaseError: If an error occurs while resetting error items.
    """
    return _default_manager.reset_error_items(connection_string)

def clear_processed_items(connection_string: str, days_to_keep: int = None) -> int:
    """
    Clear processed items from the staging table that are older than the specified number of days.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        connection_string: The SQLite connection string.
        days_to_keep: The number of days to keep processed items (default: 7).
    
    Returns:
        The number of rows deleted.
    
    Raises:
        DatabaseError: If an error occurs while clearing processed items.
    """
    return _default_manager.clear_processed_items(connection_string, days_to_keep)

def save_dataframe_to_staging(
    df: pd.DataFrame,
    connection_string: str,
    source_system: str,
    import_batch_id: str,
    **kwargs
) -> None:
    """
    Save a DataFrame to the SQLite staging table.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        df: The DataFrame to save.
        connection_string: The SQLite connection string.
        source_system: The source system identifier.
        import_batch_id: The import batch identifier.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    
    Raises:
        DatabaseError: If an error occurs while saving to the staging table.
    """
    return _default_manager.save_dataframe_to_staging(
        df, connection_string, source_system, import_batch_id, **kwargs
    )

def get_pending_items(connection_string: str) -> pd.DataFrame:
    """
    Get pending items from the staging table.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        connection_string: The SQLite connection string.
    
    Returns:
        A DataFrame containing the pending items.
    
    Raises:
        DatabaseError: If an error occurs while getting pending items.
    """
    return _default_manager.get_pending_items(connection_string)

def update_item_status(
    connection_string: str,
    staging_id: int,
    status: str,
    error_message: Optional[str] = None,
    is_processed: Optional[bool] = None
) -> None:
    """
    Update the status of an item in the staging table.
    
    This function is maintained for backward compatibility.
    New code should use the SQLiteStagingManager class directly.
    
    Args:
        connection_string: The SQLite connection string.
        staging_id: The ID of the item to update.
        status: The new status ('PENDING', 'PROCESSING', 'COMPLETED', 'ERROR').
        error_message: The error message (if status is 'ERROR').
        is_processed: Whether the item has been processed.
    
    Raises:
        DatabaseError: If an error occurs while updating the item status.
    """
    return _default_manager.update_item_status(
        connection_string, staging_id, status, error_message, is_processed
    )