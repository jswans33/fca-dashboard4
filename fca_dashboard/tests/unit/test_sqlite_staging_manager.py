"""
Unit tests for the SQLite staging manager.

This module contains tests for the SQLiteStagingManager class in the
fca_dashboard.utils.database.sqlite_staging_manager module.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database.base import DatabaseError
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager


def create_test_schema_file():
    """Create a test schema file for SQLite staging."""
    schema_content = """
    CREATE TABLE IF NOT EXISTS equipment_staging (
        staging_id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_system TEXT,
        import_batch_id TEXT,
        import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processing_status TEXT DEFAULT 'PENDING',
        error_message TEXT,
        is_processed INTEGER DEFAULT 0,
        processed_timestamp TIMESTAMP,
        equipment_tag TEXT,
        manufacturer TEXT,
        attributes TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_staging_equipment_tag ON equipment_staging(equipment_tag);
    CREATE INDEX IF NOT EXISTS idx_staging_status ON equipment_staging(processing_status);
    CREATE INDEX IF NOT EXISTS idx_staging_batch ON equipment_staging(import_batch_id);
    CREATE INDEX IF NOT EXISTS idx_staging_processed ON equipment_staging(is_processed);

    CREATE VIEW IF NOT EXISTS v_pending_items AS
    SELECT * FROM equipment_staging 
    WHERE processing_status = 'PENDING';

    CREATE VIEW IF NOT EXISTS v_error_items AS
    SELECT * FROM equipment_staging 
    WHERE processing_status = 'ERROR';
    """
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".sql", delete=False) as tmp:
        tmp.write(schema_content.encode('utf-8'))
        schema_path = tmp.name
    
    return schema_path


class TestSQLiteStagingManager:
    """Tests for the SQLiteStagingManager class."""
    
    @pytest.fixture
    def schema_path(self):
        """Fixture to create a test schema file."""
        path = create_test_schema_file()
        yield path
        # Clean up
        if os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError:
                print(f"Warning: Could not delete file {path} - it may be in use by another process")
    
    @pytest.fixture
    def db_path(self):
        """Fixture to create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            path = tmp.name
        yield path
        # Clean up
        if os.path.exists(path):
            try:
                os.unlink(path)
            except PermissionError:
                print(f"Warning: Could not delete file {path} - it may be in use by another process")
    
    @pytest.fixture
    def connection_string(self, db_path):
        """Fixture to create a SQLite connection string."""
        return f"sqlite:///{db_path}"
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture to create a mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def mock_connection_factory(self):
        """Fixture to create a mock connection factory."""
        return MagicMock(return_value=create_engine(f"sqlite:///:memory:"))
    
    @pytest.fixture
    def manager(self, schema_path, mock_logger):
        """Fixture to create a SQLiteStagingManager instance."""
        return SQLiteStagingManager(
            logger=mock_logger,
            schema_path=schema_path
        )
    
    def test_init(self, schema_path, mock_logger, mock_connection_factory):
        """Test initialization of the SQLiteStagingManager."""
        # Create a manager with all dependencies injected
        manager = SQLiteStagingManager(
            logger=mock_logger,
            connection_factory=mock_connection_factory,
            schema_path=schema_path
        )
        
        # Verify the dependencies were set correctly
        assert manager.logger == mock_logger
        assert manager.connection_factory == mock_connection_factory
        assert manager.schema_path == schema_path
        
        # Create a manager with default dependencies
        manager = SQLiteStagingManager()
        
        # Verify the default dependencies were set
        assert manager.logger is not None
        assert callable(manager.connection_factory)
        assert manager.schema_path is not None
    
    def test_initialize_db(self, manager, db_path, mock_logger):
        """Test initializing the SQLite staging database."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Verify the logger was called
        mock_logger.info.assert_called_once()
        
        # Verify the database was created
        assert os.path.exists(db_path)
        
        # Verify the schema was applied
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the equipment_staging table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='equipment_staging'")
        assert cursor.fetchone() is not None
        
        # Check if the views exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='v_pending_items'")
        assert cursor.fetchone() is not None
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='v_error_items'")
        assert cursor.fetchone() is not None
        
        # Close the connection
        conn.close()
    
    def test_initialize_db_error(self, manager, mock_logger):
        """Test error handling when initializing the database."""
        # Set an invalid schema path
        manager.schema_path = "nonexistent_file.sql"
        
        # Try to initialize the database
        with pytest.raises(DatabaseError) as excinfo:
            manager.initialize_db("test.db")
        
        # Verify the error message
        assert "Error initializing SQLite staging database" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    def test_reset_error_items(self, manager, db_path, connection_string, mock_logger):
        """Test resetting error items."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create a test record with ERROR status
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, error_message
            ) VALUES (
                'Test System', 'BATCH-001', 'ERROR', 'Test error message'
            )
        """)
        conn.commit()
        conn.close()
        
        # Reset error items
        updated_count = manager.reset_error_items(connection_string)
        
        # Verify the return value
        assert updated_count == 1
        
        # Verify the logger was called
        mock_logger.info.assert_called_with("Reset 1 error items to 'PENDING'")
        
        # Verify the record was updated
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT processing_status, error_message FROM equipment_staging")
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == 'PENDING'
        assert row[1] is None
    
    def test_reset_error_items_error(self, manager, connection_string, mock_logger, mock_connection_factory):
        """Test error handling when resetting error items."""
        # Set up the mock connection factory to raise an exception
        mock_connection_factory.side_effect = Exception("Test exception")
        manager.connection_factory = mock_connection_factory
        
        # Try to reset error items
        with pytest.raises(DatabaseError) as excinfo:
            manager.reset_error_items(connection_string)
        
        # Verify the error message
        assert "Error resetting error items" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    def test_clear_processed_items(self, manager, db_path, connection_string, mock_logger):
        """Test clearing processed items."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create test records
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Record 1: Processed 10 days ago
        ten_days_ago = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(f"""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, is_processed, processed_timestamp
            ) VALUES (
                'Test System', 'BATCH-001', 'COMPLETED', 1, '{ten_days_ago}'
            )
        """)
        
        # Record 2: Processed 5 days ago
        five_days_ago = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(f"""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, is_processed, processed_timestamp
            ) VALUES (
                'Test System', 'BATCH-002', 'COMPLETED', 1, '{five_days_ago}'
            )
        """)
        
        # Record 3: Not processed
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, is_processed
            ) VALUES (
                'Test System', 'BATCH-003', 'PENDING', 0
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Clear processed items older than 7 days
        deleted_count = manager.clear_processed_items(connection_string, days_to_keep=7)
        
        # Verify the return value
        assert deleted_count == 1
        
        # Verify the logger was called
        mock_logger.info.assert_called_with("Cleared 1 processed items older than 7 days")
        
        # Verify the records
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM equipment_staging")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 2  # One record was deleted
    
    def test_clear_processed_items_error(self, manager, connection_string, mock_logger, mock_connection_factory):
        """Test error handling when clearing processed items."""
        # Set up the mock connection factory to raise an exception
        mock_connection_factory.side_effect = Exception("Test exception")
        manager.connection_factory = mock_connection_factory
        
        # Try to clear processed items
        with pytest.raises(DatabaseError) as excinfo:
            manager.clear_processed_items(connection_string)
        
        # Verify the error message
        assert "Error clearing processed items" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    def test_save_dataframe_to_staging(self, manager, db_path, connection_string, mock_logger):
        """Test saving a DataFrame to the staging table."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'equipment_tag': ['EQ-001', 'EQ-002'],
            'manufacturer': ['Manufacturer A', 'Manufacturer B'],
            'attributes': [{'color': 'red'}, {'color': 'blue'}]
        })
        
        # Save the DataFrame to the staging table
        manager.save_dataframe_to_staging(
            df=df,
            connection_string=connection_string,
            source_system='Test System',
            import_batch_id='BATCH-001'
        )
        
        # Verify the logger was called
        mock_logger.info.assert_called_with("Successfully saved 2 rows to SQLite staging table")
        
        # Verify the records were inserted
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM equipment_staging")
        count = cursor.fetchone()[0]
        assert count == 2
        
        # Verify the metadata columns
        cursor.execute("""
            SELECT source_system, import_batch_id, processing_status, is_processed
            FROM equipment_staging LIMIT 1
        """)
        row = cursor.fetchone()
        assert row[0] == 'Test System'
        assert row[1] == 'BATCH-001'
        assert row[2] == 'PENDING'
        assert row[3] == 0
        
        # Verify the JSON columns were converted to strings
        cursor.execute("SELECT attributes FROM equipment_staging WHERE equipment_tag = 'EQ-001'")
        attributes_json = cursor.fetchone()[0]
        attributes = json.loads(attributes_json)
        assert attributes == {'color': 'red'}
        
        conn.close()
    
    def test_save_dataframe_to_staging_error(self, manager, connection_string, mock_logger, mock_connection_factory):
        """Test error handling when saving a DataFrame to the staging table."""
        # Set up the mock connection factory to raise an exception
        mock_connection_factory.side_effect = Exception("Test exception")
        manager.connection_factory = mock_connection_factory
        
        # Create a test DataFrame
        df = pd.DataFrame({
            'equipment_tag': ['EQ-001', 'EQ-002'],
            'manufacturer': ['Manufacturer A', 'Manufacturer B']
        })
        
        # Try to save the DataFrame
        with pytest.raises(DatabaseError) as excinfo:
            manager.save_dataframe_to_staging(
                df=df,
                connection_string=connection_string,
                source_system='Test System',
                import_batch_id='BATCH-001'
            )
        
        # Verify the error message
        assert "Error saving DataFrame to SQLite staging table" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    def test_get_pending_items(self, manager, db_path, connection_string, mock_logger):
        """Test getting pending items."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create test records
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Record 1: Pending
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, equipment_tag, attributes
            ) VALUES (
                'Test System', 'BATCH-001', 'PENDING', 'EQ-001', '{"color":"red"}'
            )
        """)
        
        # Record 2: Pending
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, equipment_tag, attributes
            ) VALUES (
                'Test System', 'BATCH-001', 'PENDING', 'EQ-002', '{"color":"blue"}'
            )
        """)
        
        # Record 3: Completed
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, equipment_tag
            ) VALUES (
                'Test System', 'BATCH-001', 'COMPLETED', 'EQ-003'
            )
        """)
        
        conn.commit()
        conn.close()
        
        # Get pending items
        df = manager.get_pending_items(connection_string)
        
        # Verify the return value
        assert len(df) == 2
        assert df['equipment_tag'].tolist() == ['EQ-001', 'EQ-002']
        
        # Verify the JSON columns were converted back to dictionaries
        assert df['attributes'].tolist() == [{'color': 'red'}, {'color': 'blue'}]
        
        # Verify the logger was called
        mock_logger.info.assert_called_with("Retrieved 2 pending items from staging table")
    
    def test_get_pending_items_error(self, manager, connection_string, mock_logger, mock_connection_factory):
        """Test error handling when getting pending items."""
        # Set up the mock connection factory to raise an exception
        mock_connection_factory.side_effect = Exception("Test exception")
        manager.connection_factory = mock_connection_factory
        
        # Try to get pending items
        with pytest.raises(DatabaseError) as excinfo:
            manager.get_pending_items(connection_string)
        
        # Verify the error message
        assert "Error getting pending items" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    def test_update_item_status(self, manager, db_path, connection_string, mock_logger):
        """Test updating item status."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create a test record
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, equipment_tag
            ) VALUES (
                'Test System', 'BATCH-001', 'PENDING', 'EQ-001'
            )
        """)
        conn.commit()
        
        # Get the staging_id
        cursor.execute("SELECT staging_id FROM equipment_staging")
        staging_id = cursor.fetchone()[0]
        conn.close()
        
        # Update the item status
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=staging_id,
            status='COMPLETED',
            is_processed=True
        )
        
        # Verify the logger was called
        mock_logger.info.assert_called_with(f"Updated status of item {staging_id} to 'COMPLETED'")
        
        # Verify the record was updated
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT processing_status, is_processed, processed_timestamp
            FROM equipment_staging
        """)
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == 'COMPLETED'
        assert row[1] == 1
        assert row[2] is not None  # processed_timestamp was set
    
    def test_update_item_status_with_error(self, manager, db_path, connection_string, mock_logger):
        """Test updating item status with an error message."""
        # Initialize the database
        manager.initialize_db(db_path)
        
        # Create a test record
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO equipment_staging (
                source_system, import_batch_id, processing_status, equipment_tag
            ) VALUES (
                'Test System', 'BATCH-001', 'PENDING', 'EQ-001'
            )
        """)
        conn.commit()
        
        # Get the staging_id
        cursor.execute("SELECT staging_id FROM equipment_staging")
        staging_id = cursor.fetchone()[0]
        conn.close()
        
        # Update the item status with an error message
        error_message = "Test error message with 'quotes'"
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=staging_id,
            status='ERROR',
            error_message=error_message
        )
        
        # Verify the logger was called
        mock_logger.info.assert_called_with(f"Updated status of item {staging_id} to 'ERROR'")
        
        # Verify the record was updated
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT processing_status, error_message FROM equipment_staging")
        row = cursor.fetchone()
        conn.close()
        
        assert row[0] == 'ERROR'
        assert row[1] == error_message
    
    def test_update_item_status_error(self, manager, connection_string, mock_logger, mock_connection_factory):
        """Test error handling when updating item status."""
        # Set up the mock connection factory to raise an exception
        mock_connection_factory.side_effect = Exception("Test exception")
        manager.connection_factory = mock_connection_factory
        
        # Try to update item status
        with pytest.raises(DatabaseError) as excinfo:
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=1,
                status='COMPLETED'
            )
        
        # Verify the error message
        assert "Error updating item status" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()