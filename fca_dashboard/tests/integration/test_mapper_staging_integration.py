"""
Integration tests for the mapper and staging system.

This module contains integration tests for the interaction between the mapper
system and the SQLite staging system.
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sqlalchemy import create_engine, text

from fca_dashboard.config.settings import Settings
from fca_dashboard.mappers.mapper_factory import MapperFactory
from fca_dashboard.mappers.medtronics_mapper import MappingError, MedtronicsMapper
from fca_dashboard.utils.database.sqlite_staging_manager import SQLiteStagingManager
from fca_dashboard.utils.logging_config import get_logger

# Create a logger for the tests
logger = get_logger("mapper_staging_integration_tests")


class TestMapperStagingIntegration:
    """Integration tests for the mapper and staging system."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Fixture to create a temporary configuration file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
            # Write test configuration to the file
            config = {
                "mappers": {
                    "medtronics": {
                        "column_mapping": {
                            "asset_name": "equipment_type",
                            "asset_tag": "equipment_tag",
                            "model_number": "model",
                            "serial_number": "serial_number",
                            "system_category": "category_name",
                            "sub_system_type": "mcaa_subsystem_type",
                            "sub_system_classification": "mcaa_subsystem_classification",
                            "date_installed": "install_date",
                            "room_number": "room",
                            "size": "capacity",
                            "floor": "floor",
                            "area": "other_location_info"
                        }
                    }
                },
                "databases": {
                    "staging": {
                        "schema_path": "fca_dashboard/db/staging/schema/staging_schema_sqlite.sql"
                    }
                }
            }
            
            # Convert to YAML and write to the file
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
    
    @pytest.fixture
    def medtronics_data(self):
        """Fixture to create sample Medtronics data for testing."""
        return pd.DataFrame({
            'asset_name': ['Air Handler', 'Chiller', 'Boiler'],
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            'model_number': ['Model A', 'Model B', 'Model C'],
            'serial_number': ['SN123', 'SN456', 'SN789'],
            'system_category': ['HVAC', 'HVAC', 'HVAC'],
            'sub_system_type': ['Air Distribution', 'Cooling', 'Heating'],
            'sub_system_classification': ['Type A', 'Type B', 'Type C'],
            'date_installed': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'room_number': ['101', '102', '103'],
            'size': [1000, 2000, 3000],
            'floor': ['1', '2', '3'],
            'area': ['North Wing', 'South Wing', 'East Wing'],
            'motor_hp': [5.0, 10.0, 15.0],
            'estimated_operating_hours': [8760, 4380, 2190],
            'notes': ['Note 1', 'Note 2', 'Note 3']
        })
    
    def test_mapper_to_staging_workflow(self, settings, db_path, connection_string, 
                                        staging_manager, medtronics_data, monkeypatch):
        """Test a complete workflow from mapper to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Step 1: Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Step 2: Create a mapper for Medtronics
        mapper = MedtronicsMapper()
        
        # Step 3: Map the DataFrame
        mapped_df = mapper.map_dataframe(medtronics_data)
        
        # Step 4: Save the mapped DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=mapped_df,
            connection_string=connection_string,
            source_system='Medtronics',
            import_batch_id=batch_id
        )
        
        # Step 5: Verify the data was saved to staging
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Check the total count
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3
            
            # Check the mapped columns
            result = conn.execute(text("""
                SELECT equipment_type, equipment_tag, model, serial_number, category_name, 
                       capacity, install_date, room, floor, other_location_info, attributes
                FROM equipment_staging
                WHERE equipment_tag = 'AHU-001'
            """))
            row = result.fetchone()
            
            # Verify the values were mapped correctly
            assert row[0] == 'Air Handler'  # equipment_type
            assert row[1] == 'AHU-001'      # equipment_tag
            assert row[2] == 'Model A'       # model
            assert row[3] == 'SN123'         # serial_number
            assert row[4] == 'HVAC'          # category_name
            assert row[5] == 1000            # capacity
            assert row[6] == '2023-01-01'    # install_date
            assert row[7] == '101'           # room
            assert row[8] == '1'             # floor
            assert row[9] == 'North Wing'    # other_location_info
            
            # Verify the attributes JSON
            attributes = json.loads(row[10])
            assert "medtronics_attributes" in attributes
            assert attributes["medtronics_attributes"]["motor_hp"]["0"] == 5.0
            assert attributes["medtronics_attributes"]["estimated_operating_hours"]["0"] == 8760
            assert attributes["medtronics_attributes"]["notes"]["0"] == 'Note 1'
    
    def test_mapper_factory_to_staging_workflow(self, settings, db_path, connection_string, 
                                               staging_manager, medtronics_data, monkeypatch):
        """Test a complete workflow from mapper factory to staging."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Step 1: Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Step 2: Create a mapper factory
        factory = MapperFactory()
        
        # Step 3: Create a mapper for Medtronics
        mapper = factory.create_mapper("Medtronics")
        
        # Step 4: Map the DataFrame
        mapped_df = mapper.map_dataframe(medtronics_data)
        
        # Step 5: Save the mapped DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=mapped_df,
            connection_string=connection_string,
            source_system='Medtronics',
            import_batch_id=batch_id
        )
        
        # Step 6: Verify the data was saved to staging
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            # Check the total count
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging"))
            count = result.fetchone()[0]
            assert count == 3
            
            # Check the source system and batch ID
            result = conn.execute(text(f"""
                SELECT source_system, import_batch_id
                FROM equipment_staging
                LIMIT 1
            """))
            row = result.fetchone()
            assert row[0] == 'Medtronics'
            assert row[1] == batch_id
    
    def test_process_staged_data(self, settings, db_path, connection_string, 
                                staging_manager, medtronics_data, monkeypatch):
        """Test processing staged data after mapping."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Step 1: Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Step 2: Create a mapper for Medtronics
        mapper = MedtronicsMapper()
        
        # Step 3: Map the DataFrame
        mapped_df = mapper.map_dataframe(medtronics_data)
        
        # Step 4: Save the mapped DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=mapped_df,
            connection_string=connection_string,
            source_system='Medtronics',
            import_batch_id=batch_id
        )
        
        # Step 5: Get pending items
        pending_items = staging_manager.get_pending_items(connection_string)
        
        # Verify the pending items
        assert len(pending_items) == 3
        assert pending_items['equipment_type'].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert pending_items['equipment_tag'].tolist() == ['AHU-001', 'CH-001', 'B-001']
        
        # Step 6: Process the items (simulate processing by updating status)
        for _, item in pending_items.iterrows():
            staging_manager.update_item_status(
                connection_string=connection_string,
                staging_id=item['staging_id'],
                status='COMPLETED',
                is_processed=True
            )
        
        # Step 7: Verify all items were processed
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE is_processed = 1"))
            count = result.fetchone()[0]
            assert count == 3
            
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'COMPLETED'"))
            count = result.fetchone()[0]
            assert count == 3
    
    def test_error_handling_in_mapping_workflow(self, settings, db_path, connection_string, 
                                              staging_manager, monkeypatch):
        """Test error handling in the mapping workflow."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Step 1: Initialize the staging database
        staging_manager.initialize_db(db_path)
        
        # Step 2: Create a mapper for Medtronics
        mapper = MedtronicsMapper()
        
        # Step 3: Create a DataFrame with problematic data
        problematic_data = pd.DataFrame({
            'asset_name': ['Air Handler', 'Chiller', 'Boiler'],
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            # Missing required columns
        })
        
        # Step 4: Map the DataFrame - this should raise a MappingError
        with pytest.raises(MappingError) as excinfo:
            mapper.map_dataframe(problematic_data)
        
        # Verify the error message
        assert "Error mapping Medtronics data: Source DataFrame is missing required columns" in str(excinfo.value)
        
        # Create a minimal valid DataFrame for testing the rest of the workflow
        valid_data = pd.DataFrame({
            'asset_name': ['Air Handler', 'Chiller', 'Boiler'],
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            'serial_number': ['SN123', 'SN456', 'SN789']  # Add required column
        })
        
        # Map the valid DataFrame
        mapped_df = mapper.map_dataframe(valid_data)
        
        # Step 5: Save the mapped DataFrame to staging
        batch_id = f'TEST-BATCH-{datetime.now().strftime("%Y%m%d%H%M%S")}'
        staging_manager.save_dataframe_to_staging(
            df=mapped_df,
            connection_string=connection_string,
            source_system='Medtronics',
            import_batch_id=batch_id
        )
        
        # Step 6: Get pending items
        pending_items = staging_manager.get_pending_items(connection_string)
        
        # Verify the pending items
        assert len(pending_items) == 3
        
        # Step 7: Simulate processing with errors
        for _, item in pending_items.iterrows():
            staging_manager.update_item_status(
                connection_string=connection_string,
                staging_id=item['staging_id'],
                status='ERROR',
                error_message=f"Missing required data for {item['equipment_tag']}"
            )
        
        # Step 8: Verify all items are in ERROR state
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'ERROR'"))
            count = result.fetchone()[0]
            assert count == 3
            
            # Check the error message
            result = conn.execute(text("""
                SELECT error_message FROM equipment_staging WHERE equipment_tag = 'AHU-001'
            """))
            error_message = result.fetchone()[0]
            assert "Missing required data for AHU-001" in error_message
        
        # Step 9: Reset error items
        reset_count = staging_manager.reset_error_items(connection_string)
        
        # Verify all items were reset
        assert reset_count == 3
        
        # Verify all items are now PENDING
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM equipment_staging WHERE processing_status = 'PENDING'"))
            count = result.fetchone()[0]
            assert count == 3