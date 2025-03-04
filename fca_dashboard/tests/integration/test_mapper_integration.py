"""
Integration tests for the mapper system.

This module contains integration tests for the mapper system, testing the
interaction between mappers, the mapper factory, and the configuration system.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fca_dashboard.config.settings import Settings
from fca_dashboard.mappers.mapper_factory import MapperFactory
from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.mappers.wichita_mapper import WichitaMapper
from fca_dashboard.utils.logging_config import get_logger

# Create a logger for the tests
logger = get_logger("mapper_integration_tests")


class TestMapperIntegration:
    """Integration tests for the mapper system."""
    
    @pytest.fixture
    def temp_config_file(self):
        """Fixture to create a temporary configuration file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
            # Write test configuration to the file
            config = {
                "mappers": {
                    "registry": {
                        "TestSystem": "fca_dashboard.mappers.medtronics_mapper.MedtronicsMapper"
                    },
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
                            "area": "other_location_info",
                            "test_field": "test_destination"  # Custom field for testing
                        }
                    },
                    "wichita": {
                        "column_mapping": {
                            "Asset_Name": "equipment_type",
                            "Asset_Tag": "equipment_tag",
                            "Manufacturer": "manufacturer",
                            "Model": "model",
                            "Serial_Number": "serial_number",
                            "Asset_Category_Name": "category_name",
                            "Asset_Type": "equipment_type",
                            "Location": "other_location_info",
                            "Install_Date": "install_date",
                            "Test_Field": "test_destination"  # Custom field for testing
                        }
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
            'notes': ['Note 1', 'Note 2', 'Note 3'],
            'test_field': ['Test Value 1', 'Test Value 2', 'Test Value 3']
        })
    
    @pytest.fixture
    def wichita_data(self):
        """Fixture to create sample Wichita data for testing."""
        return pd.DataFrame({
            'Asset_Name': ['Air Handler', 'Chiller', 'Boiler'],
            'Asset_Tag': ['AHU-001', 'CH-001', 'B-001'],
            'Manufacturer': ['Manufacturer A', 'Manufacturer B', 'Manufacturer C'],
            'Model': ['Model A', 'Model B', 'Model C'],
            'Serial_Number': ['SN123', 'SN456', 'SN789'],
            'Asset_Category_Name': ['HVAC', 'HVAC', 'HVAC'],
            'Asset_Type': ['Air Handler', 'Chiller', 'Boiler'],
            'Location': ['North Wing', 'South Wing', 'East Wing'],
            'Install_Date': ['2023-01-01', 'Invalid Date', '2023-03-01'],
            'Test_Field': ['Test Value 1', 'Test Value 2', 'Test Value 3']
        })
    
    def test_mapper_factory_with_config(self, settings, monkeypatch):
        """Test the mapper factory with configuration."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper factory
        factory = MapperFactory()
        
        # Verify the registry includes the default mappers
        assert "Medtronics" in factory._mapper_registry
        assert "Wichita" in factory._mapper_registry
        
        # Create a mapper for Medtronics
        mapper = factory.create_mapper("Medtronics")
        
        # Verify the mapper is of the correct type
        assert isinstance(mapper, MedtronicsMapper)
    
    def test_medtronics_mapper_with_config(self, settings, medtronics_data, monkeypatch):
        """Test the Medtronics mapper with configuration."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Verify the column mapping was loaded from settings
        column_mapping = mapper.get_column_mapping()
        assert "asset_name" in column_mapping
        assert column_mapping["asset_name"] == "equipment_type"
        assert "asset_tag" in column_mapping
        assert column_mapping["asset_tag"] == "equipment_tag"
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(medtronics_data)
        
        # Verify the columns were mapped correctly
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "model" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
    
    def test_wichita_mapper_with_config(self, settings, wichita_data, monkeypatch):
        """Test the Wichita mapper with configuration."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = WichitaMapper()
        
        # Verify the column mapping was loaded from settings
        column_mapping = mapper.get_column_mapping()
        assert "Asset_Name" in column_mapping
        assert column_mapping["Asset_Name"] == "equipment_type"
        assert "Asset_Tag" in column_mapping
        assert column_mapping["Asset_Tag"] == "equipment_tag"
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(wichita_data)
        
        # Verify the columns were mapped correctly
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "manufacturer" in mapped_df.columns
        assert "model" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
    
    def test_end_to_end_mapping_workflow(self, settings, medtronics_data, monkeypatch):
        """Test an end-to-end mapping workflow."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Step 1: Create a mapper factory
        factory = MapperFactory()
        
        # Step 2: Create a mapper for Medtronics
        mapper = factory.create_mapper("Medtronics")
        
        # Step 3: Map the DataFrame
        mapped_df = mapper.map_dataframe(medtronics_data)
        
        # Verify the mapping was successful
        assert len(mapped_df) == len(medtronics_data)
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        
        # Verify the attributes column was created
        assert "attributes" in mapped_df.columns
        
        # Verify the attributes contain the expected JSON data
        attributes = json.loads(mapped_df["attributes"].iloc[0])
        assert "medtronics_attributes" in attributes
        assert "motor_hp" in attributes["medtronics_attributes"]
        assert "estimated_operating_hours" in attributes["medtronics_attributes"]
        assert "notes" in attributes["medtronics_attributes"]
    
    def test_multiple_mappers_with_same_config(self, settings, medtronics_data, wichita_data, monkeypatch):
        """Test using multiple mappers with the same configuration."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper factory
        factory = MapperFactory()
        
        # Create mappers for different source systems
        medtronics_mapper = factory.create_mapper("Medtronics")
        wichita_mapper = factory.create_mapper("Wichita")
        
        # Map the DataFrames
        medtronics_mapped_df = medtronics_mapper.map_dataframe(medtronics_data)
        wichita_mapped_df = wichita_mapper.map_dataframe(wichita_data)
        
        # Verify both mappers used their respective configurations
        assert "equipment_type" in medtronics_mapped_df.columns
        assert "equipment_type" in wichita_mapped_df.columns
        
        # Verify the common fields were mapped correctly in both DataFrames
        assert medtronics_mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert wichita_mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        
        assert medtronics_mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
        assert wichita_mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']