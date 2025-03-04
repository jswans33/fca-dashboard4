"""
Unit tests for mapper edge cases.

This module contains unit tests for edge cases in the mapper system,
such as empty DataFrames, DataFrames with missing columns, etc.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fca_dashboard.config.settings import Settings
from fca_dashboard.mappers.base_mapper import MappingError, ValidationError
from fca_dashboard.mappers.mapper_factory import MapperFactory
from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.utils.logging_config import get_logger

# Create a logger for the tests
logger = get_logger("mapper_edge_case_tests")


class TestMapperEdgeCases:
    """Unit tests for mapper edge cases."""
    
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
    
    def test_empty_dataframe(self, settings, monkeypatch):
        """Test mapping an empty DataFrame."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Create an empty DataFrame
        empty_df = pd.DataFrame()
        
        # Map the DataFrame - this should log a warning but not fail
        mapped_df = mapper.map_dataframe(empty_df)
        
        # Verify the result is also an empty DataFrame
        assert mapped_df.empty
    
    def test_missing_required_columns(self, settings, monkeypatch):
        """Test mapping a DataFrame with missing required columns."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Create a DataFrame with missing required columns
        df = pd.DataFrame({
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            # Missing 'asset_name' and 'serial_number'
        })
        
        # Map the DataFrame - this should raise a ValidationError
        with pytest.raises(ValidationError) as excinfo:
            mapper.map_dataframe(df)
        
        # Verify the error message
        assert "missing required columns" in str(excinfo.value)
        assert "asset_name" in str(excinfo.value)
        assert "serial_number" in str(excinfo.value)
    
    def test_partial_column_mapping(self, settings, monkeypatch):
        """Test mapping a DataFrame with only some of the mapped columns."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Create a DataFrame with only some of the mapped columns
        df = pd.DataFrame({
            'asset_name': ['Air Handler', 'Chiller', 'Boiler'],
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            'serial_number': ['SN123', 'SN456', 'SN789'],
            # Missing other columns like 'model_number', 'system_category', etc.
        })
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(df)
        
        # Verify the result has the expected columns
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
        assert mapped_df["serial_number"].tolist() == ['SN123', 'SN456', 'SN789']
    
    def test_custom_column_mapping(self, settings, monkeypatch):
        """Test mapping a DataFrame with a custom column mapping."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Override the column mapping
        custom_mapping = {
            "custom_name": "equipment_type",
            "custom_tag": "equipment_tag",
            "custom_serial": "serial_number"
        }
        mapper._column_mapping = custom_mapping
        
        # Update required source columns
        mapper.set_required_source_columns(["custom_name", "custom_tag", "custom_serial"])
        
        # Create a DataFrame with the custom columns
        df = pd.DataFrame({
            'custom_name': ['Air Handler', 'Chiller', 'Boiler'],
            'custom_tag': ['AHU-001', 'CH-001', 'B-001'],
            'custom_serial': ['SN123', 'SN456', 'SN789']
        })
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(df)
        
        # Verify the result has the expected columns
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
        assert mapped_df["serial_number"].tolist() == ['SN123', 'SN456', 'SN789']
    
    def test_none_dataframe(self, settings, monkeypatch):
        """Test mapping a None DataFrame."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Map a None DataFrame - this should raise a MappingError
        with pytest.raises(MappingError) as excinfo:
            mapper.map_dataframe(None)
        
        # Verify the error message
        assert "Error mapping Medtronics data: Source DataFrame is None" in str(excinfo.value)
    
    def test_dataframe_with_extra_columns(self, settings, monkeypatch):
        """Test mapping a DataFrame with extra columns not in the mapping."""
        # Monkeypatch the settings module to use our test settings
        import fca_dashboard.config.settings
        monkeypatch.setattr(fca_dashboard.config.settings, "settings", settings)
        
        # Create a mapper
        mapper = MedtronicsMapper()
        
        # Create a DataFrame with extra columns
        df = pd.DataFrame({
            'asset_name': ['Air Handler', 'Chiller', 'Boiler'],
            'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
            'serial_number': ['SN123', 'SN456', 'SN789'],
            'extra_column_1': [1, 2, 3],
            'extra_column_2': ['A', 'B', 'C']
        })
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(df)
        
        # Verify the result has the expected columns
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the extra columns are not in the result
        assert "extra_column_1" not in mapped_df.columns
        assert "extra_column_2" not in mapped_df.columns