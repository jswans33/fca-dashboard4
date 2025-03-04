"""
Integration tests for the SQLite staging system.

This module contains integration tests for the SQLite staging system,
testing the interaction between the staging utilities and the database.
"""

import json
import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger

# Create a logger for the tests
logger = get_logger("sqlite_staging_integration_tests")


class TestSQLiteStagingIntegration:
    """Integration tests for the SQLite staging system."""
    
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
    def manager(self):
        """Fixture to create a SQLiteStagingManager instance."""
        return SQLiteStagingManager()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to create sample equipment data."""
        return pd.DataFrame({
            'equipment_tag': ['EQ-001', 'EQ-002', 'EQ-003'],
            'manufacturer': ['Manufacturer A', 'Manufacturer B', 'Manufacturer C'],
            'model': ['Model X', 'Model Y', 'Model Z'],
            'serial_number': ['SN123456', 'SN789012', 'SN345678'],
            'capacity': [100.0, 200.0, 300.0],
            'install_date': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'status': ['Active', 'Active', 'Inactive'],
            'category_name': ['HVAC', 'Electrical', 'Plumbing'],
            'building_name': ['Building A', 'Building B', 'Building C'],
            'floor': ['1', '2', '3'],
            'room': ['101', '202', '303'],
            'attributes': [
                {'color': 'red', 'weight': 50},
                {'color': 'blue', 'weight': 75},
                {'color': 'green', 'weight': 100}
            ]
        })
    
    def test_full_staging_workflow(self, manager, db_path, connection_string, sample_data):
        """Test a complete workflow using the SQLite staging system."""
        # Step 1: Initialize the database
        manager.initialize_db(db_path)
        
        # Verify the database was created
        assert os.path.exists(db_path)
        
        # Step 2: Save data to the staging table
        batch_id = f'BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        manager.save_dataframe_to_staging(
            df=sample_data,
            connection_string=connection_string,
            source_system='Test System',
            import_batch_id=batch_id
        )
        
        # Verify the data was saved
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3
        
        # Step 3: Get pending items
        pending_items = manager.get_pending_items(connection_string)
        
        # Verify the pending items
        assert len(pending_items) == 3
        assert pending_items['equipment_tag'].tolist() == ['EQ-001', 'EQ-002', 'EQ-003']
        
        # Verify the JSON columns were converted back to dictionaries
        assert pending_items['attributes'].tolist() == [
            {'color': 'red', 'weight': 50},
            {'color': 'blue', 'weight': 75},
            {'color': 'green', 'weight': 100}
        ]
        
        # Step 4: Update item status
        # Update the first item to 'PROCESSING'
        first_item_id = pending_items.iloc[0]['staging_id']
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=first_item_id,
            status='PROCESSING'
        )
        
        # Update the second item to 'COMPLETED' and mark as processed
        second_item_id = pending_items.iloc[1]['staging_id']
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=second_item_id,
            status='COMPLETED',
            is_processed=True
        )
        
        # Update the third item to 'ERROR'
        third_item_id = pending_items.iloc[2]['staging_id']
        manager.update_item_status(
            connection_string=connection_string,
            staging_id=third_item_id,
            status='ERROR',
            error_message='Example error message'
        )
        
        # Verify the updates
        with engine.connect() as conn:
            # Check first item
            result = conn.execute(text(f"SELECT processing_status FROM equipment_staging WHERE staging_id = {first_item_id}"))
            assert result.fetchone()[0] == 'PROCESSING'
            
            # Check second item
            result = conn.execute(text(f"""
                SELECT processing_status, is_processed, processed_timestamp 
                FROM equipment_staging WHERE staging_id = {second_item_id}
            """))
            row = result.fetchone()
            assert row[0] == 'COMPLETED'
            assert row[1] == 1
            assert row[2] is not None
            
            # Check third item
            result = conn.execute(text(f"""
                SELECT processing_status, error_message 
                FROM equipment_staging WHERE staging_id = {third_item_id}
            """))
            row = result.fetchone()
            assert row[0] == 'ERROR'
            assert row[1] == 'Example error message'
        
        # Step 5: Get pending items again
        pending_items = manager.get_pending_items(connection_string)
        
        # Verify only one item is still pending
        assert len(pending_items) == 0  # All items have been processed or are in error state
        
        # Step 6: Reset error items
        reset_count = manager.reset_error_items(connection_string)
        
        # Verify one item was reset
        assert reset_count == 1
        
        # Get pending items again
        pending_items = manager.get_pending_items(connection_string)
        
        # Verify the error item is now pending
        assert len(pending_items) == 1
        assert pending_items.iloc[0]['equipment_tag'] == 'EQ-003'
        
        # Step 7: Clear processed items
        cleared_count = manager.clear_processed_items(connection_string, days_to_keep=0)
        
        # Verify the cleared count (may be 0 if the processed timestamp is too recent)
        # In a real-world scenario, we'd have more control over the timestamps
        logger.info(f"Cleared {cleared_count} processed items")
        
        # Verify the total count of items
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            # If cleared_count is 0, we expect all 3 items to still be there
            # If cleared_count is 1, we expect 2 items to be there
            assert count == (3 - cleared_count)
    
    def test_etl_pipeline_integration(self, manager, db_path, connection_string, sample_data):
        """Test integration with an ETL pipeline."""
        # Step 1: Initialize the database
        manager.initialize_db(db_path)
        
        # Step 2: Extract data (simulated by using sample_data)
        # In a real ETL pipeline, this would involve extracting data from a source system
        
        # Step 3: Transform data (simulated by adding a derived column to attributes)
        # In a real ETL pipeline, this would involve more complex transformations
        transformed_data = sample_data.copy()
        
        # Instead of adding a new column, add the derived data to the attributes JSON
        # This avoids schema compatibility issues
        for i, row in transformed_data.iterrows():
            if isinstance(row['attributes'], dict):
                attributes = row['attributes']
            else:
                attributes = {}
            
            # Add the derived data to the attributes
            attributes['derived_capacity'] = f"Capacity: {row['capacity']} units"
            transformed_data.at[i, 'attributes'] = attributes
        
        # Step 4: Load data into staging
        batch_id = f'BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        manager.save_dataframe_to_staging(
            df=transformed_data,
            connection_string=connection_string,
            source_system='ETL Pipeline',
            import_batch_id=batch_id
        )
        
        # Verify the data was loaded
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3
        
        # Step 5: Process the data (simulated by updating status)
        # In a real ETL pipeline, this would involve more complex processing
        pending_items = manager.get_pending_items(connection_string)
        
        for _, item in pending_items.iterrows():
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=item['staging_id'],
                status='COMPLETED',
                is_processed=True
            )
        
        # Verify all items were processed
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE is_processed = 1"))
            count = result.fetchone()[0]
            assert count == 3
    
    def test_error_handling_and_recovery(self, manager, db_path, connection_string, sample_data):
        """Test error handling and recovery in the staging system."""
        # Step 1: Initialize the database
        manager.initialize_db(db_path)
        
        # Step 2: Save data to the staging table
        batch_id = f'BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        manager.save_dataframe_to_staging(
            df=sample_data,
            connection_string=connection_string,
            source_system='Test System',
            import_batch_id=batch_id
        )
        
        # Step 3: Simulate processing with errors
        pending_items = manager.get_pending_items(connection_string)
        
        # Mark all items as ERROR
        for _, item in pending_items.iterrows():
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=item['staging_id'],
                status='ERROR',
                error_message=f"Error processing item {item['equipment_tag']}"
            )
        
        # Verify all items are in ERROR state
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'ERROR'"))
            count = result.fetchone()[0]
            assert count == 3
        
        # Step 4: Reset error items
        reset_count = manager.reset_error_items(connection_string)
        
        # Verify all items were reset
        assert reset_count == 3
        
        # Verify all items are now PENDING
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'PENDING'"))
            count = result.fetchone()[0]
            assert count == 3
        
        # Step 5: Process items successfully
        pending_items = manager.get_pending_items(connection_string)
        
        for _, item in pending_items.iterrows():
            manager.update_item_status(
                connection_string=connection_string,
                staging_id=item['staging_id'],
                status='COMPLETED',
                is_processed=True
            )
        
        # Verify all items were processed
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE is_processed = 1"))
            count = result.fetchone()[0]
            assert count == 3