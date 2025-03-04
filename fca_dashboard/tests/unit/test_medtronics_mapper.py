"""
Unit tests for the Medtronics mapper module.

This module contains tests for the MedtronicsMapper class in the
fca_dashboard.mappers.medtronics_mapper module.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fca_dashboard.mappers.medtronics_mapper import MappingError, MedtronicsMapper


class TestMedtronicsMapper:
    """Tests for the MedtronicsMapper class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture to create a mock logger."""
        return MagicMock()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture to create sample data for testing."""
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
    
    @patch("fca_dashboard.mappers.medtronics_mapper.settings")
    def test_init_with_config(self, mock_settings, mock_logger):
        """Test initialization with configuration from settings."""
        # Set up the mock settings
        mock_settings.get.return_value = {
            "medtronics": {
                "column_mapping": {
                    "asset_name": "equipment_type",
                    "asset_tag": "equipment_tag",
                    "custom_field": "custom_destination"
                }
            }
        }
        
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Verify the column mapping was loaded from settings
        column_mapping = mapper.get_column_mapping()
        assert column_mapping["asset_name"] == "equipment_type"
        assert column_mapping["asset_tag"] == "equipment_tag"
        assert column_mapping["custom_field"] == "custom_destination"
        
        # Verify the settings were accessed
        mock_settings.get.assert_called_with("mappers", {})
    
    @patch("fca_dashboard.mappers.medtronics_mapper.settings")
    def test_init_with_empty_config(self, mock_settings, mock_logger):
        """Test initialization with empty configuration."""
        # Set up the mock settings to return empty values
        mock_settings.get.return_value = {}
        
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Verify the default column mapping was used
        column_mapping = mapper.get_column_mapping()
        assert column_mapping["asset_name"] == "equipment_type"
        assert column_mapping["asset_tag"] == "equipment_tag"
        assert column_mapping["model_number"] == "model"
        assert column_mapping["serial_number"] == "serial_number"
        
        # Verify the logger was called
        mock_logger.warning.assert_called_with("Medtronics mapping not found in settings, using default mapping")
    
    def test_map_dataframe(self, sample_data, mock_logger):
        """Test mapping a DataFrame."""
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(sample_data)
        
        # Verify the columns were mapped correctly
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "model" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
        
        # Verify the attributes column was created
        assert "attributes" in mapped_df.columns
        
        # Verify the attributes contain the expected JSON data
        attributes = json.loads(mapped_df["attributes"].iloc[0])
        assert "medtronics_attributes" in attributes
        assert "motor_hp" in attributes["medtronics_attributes"]
        assert "estimated_operating_hours" in attributes["medtronics_attributes"]
        assert "notes" in attributes["medtronics_attributes"]
    
    def test_map_dataframe_with_spaces_in_columns(self, mock_logger):
        """Test mapping a DataFrame with spaces in column names."""
        # Create a DataFrame with spaces in column names
        df = pd.DataFrame({
            'asset name': ['Air Handler', 'Chiller'],
            'asset tag': ['AHU-001', 'CH-001'],
            'model number': ['Model A', 'Model B'],
            'serial number': ['SN123', 'SN456']  # Add required serial_number column
        })
        
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(df)
        
        # Verify the columns were normalized and mapped correctly
        assert "equipment_type" in mapped_df.columns
        assert "equipment_tag" in mapped_df.columns
        assert "model" in mapped_df.columns
        assert "serial_number" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["equipment_type"].tolist() == ['Air Handler', 'Chiller']
        assert mapped_df["equipment_tag"].tolist() == ['AHU-001', 'CH-001']
        assert mapped_df["model"].tolist() == ['Model A', 'Model B']
        assert mapped_df["serial_number"].tolist() == ['SN123', 'SN456']
    
    def test_map_dataframe_error(self, mock_logger):
        """Test error handling when mapping a DataFrame."""
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Create a DataFrame that will cause an error (None is not a DataFrame)
        df = None
        
        # Try to map the DataFrame
        with pytest.raises(MappingError) as excinfo:
            mapper.map_dataframe(df)
        
        # Verify the error message
        assert "Error mapping Medtronics data:" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    @patch("fca_dashboard.mappers.medtronics_mapper.settings")
    def test_custom_column_mapping(self, mock_settings, mock_logger, sample_data):
        """Test using a custom column mapping from settings."""
        # Set up the mock settings with a custom mapping
        mock_settings.get.return_value = {
            "medtronics": {
                "column_mapping": {
                    "asset_name": "custom_equipment_type",
                    "asset_tag": "custom_equipment_tag",
                    "model_number": "custom_model"
                }
            }
        }
        
        # Create a mapper
        mapper = MedtronicsMapper(logger=mock_logger)
        
        # Map the DataFrame
        mapped_df = mapper.map_dataframe(sample_data)
        
        # Verify the columns were mapped according to the custom mapping
        assert "custom_equipment_type" in mapped_df.columns
        assert "custom_equipment_tag" in mapped_df.columns
        assert "custom_model" in mapped_df.columns
        
        # Verify the values were mapped correctly
        assert mapped_df["custom_equipment_type"].tolist() == ['Air Handler', 'Chiller', 'Boiler']
        assert mapped_df["custom_equipment_tag"].tolist() == ['AHU-001', 'CH-001', 'B-001']
        assert mapped_df["custom_model"].tolist() == ['Model A', 'Model B', 'Model C']