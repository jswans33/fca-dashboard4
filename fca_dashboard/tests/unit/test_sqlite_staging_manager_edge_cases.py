"""
Unit tests for SQLiteStagingManager edge cases.

This module contains unit tests for edge cases in the SQLiteStagingManager,
such as empty DataFrames, DataFrames with missing columns, etc.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fca_dashboard.config.settings import Settings
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger

# Create a logger for the tests
logger = get_logger("sqlite_staging_manager_edge_case_tests")


class TestSQLiteStagingManagerEdgeCases:
    """Unit tests for SQLiteStagingManager edge cases."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Fixture to create a temporary configuration file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
            # Write test configuration to the file
            config = {
                "databases": {
                    "staging": {
                        "schema_path": "fca_dashboard/db/staging/schema/staging_schema_sqlite.sql"
                    }
                }
            }
            
            # Convert to YAML and write to the file
            import yaml
            yaml_content = yaml.dump(config)
            tmp.write(yaml_content.encode('utf-8'))
            config_path = tmp.name
        
        yield config_path
        
        # Clean up
        if os.path.exists(config_path):
            try:
                os.unlink(config_path)
            except PermissionError:
                print(f"Warning: Could not delete file {config_path} - it may be in use by another process")
    
    @pytest.fixture
    def settings(self, temp_config_file):
        """Fixture to create a Settings instance with the test configuration."""
        return Settings(config_path=temp_config_file)
    
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
    def staging_manager(self):
        """Fixture to create a SQLiteStagingManager instance."""
        return SQLiteStagingManager()
    
    def test_empty_dataframe(self, settings, db_path, connection_string, staging_manager, monkeypatch):
        """Test saving an empty DataFrame to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Save the empty DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=empty_df,
            connection_string=connection_string,
            source_system='Test',
            import_batch_id=batch_id
        )
        
        # Verify no rows were saved
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 0
    
    def test_none_dataframe(self, settings, db_path, connection_string, staging_manager, monkeypatch):
        """Test saving a None DataFrame to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Save a None DataFrame to staging - this should raise a ValueError
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        with pytest.raises(ValueError) as excinfo:
            staging_manager.save_dataframe_to_staging(
                df=None,
                connection_string=connection_string,
                source_system='Test',
                import_batch_id=batch_id
            )
        
        # Verify the error message
        assert "Cannot save None DataFrame" in str(excinfo.value)
    
    def test_dataframe_with_no_valid_columns(self, settings, db_path, connection_string, staging_manager, monkeypatch):
        """Test saving a DataFrame with no valid columns to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Create a DataFrame with columns that don't exist in the staging table
        df = pd.DataFrame({
            'invalid_column_1': [1, 2, 3],
            'invalid_column_2': ['A', 'B', 'C']
        })
        
        # Save the DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=df,
            connection_string=connection_string,
            source_system='Test',
            import_batch_id=batch_id
        )
        
        # Verify only the metadata columns were saved
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3  # 3 rows with metadata columns
            
            # Check that the metadata columns were saved
            result = conn.execute(text("""
                SELECT source_system, import_batch_id
                FROM equipment_staging
                LIMIT 1
            """))
            row = result.fetchone()
            assert row[0] == 'Test'
            assert row[1] == batch_id
    
    def test_already_mapped_dataframe(self, settings, db_path, connection_string, staging_manager, monkeypatch):
        """Test saving an already mapped DataFrame to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Create a DataFrame that looks like it's already been mapped
        df = pd.DataFrame({
            'equipment_type': ['Air Handler', 'Chiller', 'Boiler'],
            'equipment_tag': ['AHU-001', 'CH-001', 'B-001'],
            'serial_number': ['SN123', 'SN456', 'SN789'],
            'category_name': ['HVAC', 'HVAC', 'HVAC']
        })
        
        # Save the DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=df,
            connection_string=connection_string,
            source_system='Test',
            import_batch_id=batch_id
        )
        
        # Verify the data was saved correctly
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Check the total count
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3
            
            # Check the mapped columns
            result = conn.execute(text("""
                SELECT equipment_type, equipment_tag, serial_number, category_name
                FROM equipment_staging
                WHERE equipment_tag = 'AHU-001'
            """))
            row = result.fetchone()
            
            # Verify the values were saved correctly
            assert row[0] == 'Air Handler'  # equipment_type
            assert row[1] == 'AHU-001'      # equipment_tag
            assert row[2] == 'SN123'        # serial_number
            assert row[3] == 'HVAC'         # category_name
    
    def test_json_column_handling(self, settings, db_path, connection_string, staging_manager, monkeypatch):
        """Test handling of JSON columns in the staging manager."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Create a DataFrame with JSON columns
        df = pd.DataFrame({
            'equipment_type': ['Air Handler', 'Chiller', 'Boiler'],
            'equipment_tag': ['AHU-001', 'CH-001', 'B-001'],
            'serial_number': ['SN123', 'SN456', 'SN789'],
            'attributes': [
                {'key1': 'value1', 'key2': 123},
                {'key1': 'value2', 'key2': 456},
                {'key1': 'value3', 'key2': 789}
            ],
            'maintenance_data': [
                {'last_service': '2023-01-01', 'next_service': '2023-07-01'},
                {'last_service': '2023-02-01', 'next_service': '2023-08-01'},
                {'last_service': '2023-03-01', 'next_service': '2023-09-01'}
            ]
        })
        
        # Save the DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=df,
            connection_string=connection_string,
            source_system='Test',
            import_batch_id=batch_id
        )
        
        # Verify the JSON columns were saved correctly
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Check the attributes column
            result = conn.execute(text("""
                SELECT attributes, maintenance_data
                FROM equipment_staging
                WHERE equipment_tag = 'AHU-001'
            """))
            row = result.fetchone()
            
            # Verify the JSON columns were saved as strings
            attributes = json.loads(row[0])
            assert attributes['key1'] == 'value1'
            assert attributes['key2'] == 123
            
            maintenance_data = json.loads(row[1])
            assert maintenance_data['last_service'] == '2023-01-01'
            assert maintenance_data['next_service'] == '2023-07-01'