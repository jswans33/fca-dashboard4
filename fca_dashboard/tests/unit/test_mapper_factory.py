"""
Unit tests for the mapper factory module.

This module contains tests for the MapperFactory class in the
fca_dashboard.mappers.mapper_factory module.
"""

from unittest.mock import MagicMock, patch

import pytest

from fca_dashboard.mappers.base_mapper import BaseMapper
from fca_dashboard.mappers.mapper_factory import MapperFactory, MapperFactoryError
from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.mappers.wichita_mapper import WichitaMapper


class TestMapperFactory:
    """Tests for the MapperFactory class."""
    
    @pytest.fixture
    def mock_logger(self):
        """Fixture to create a mock logger."""
        return MagicMock()
    
    def test_init(self, mock_logger):
        """Test initialization of the MapperFactory."""
        # Create a factory with a custom logger
        factory = MapperFactory(logger=mock_logger)
        
        # Verify the logger was set correctly
        assert factory.logger == mock_logger
        
        # Verify the registry was initialized
        assert isinstance(factory._mapper_registry, dict)
        assert "Medtronics" in factory._mapper_registry
        assert "Wichita" in factory._mapper_registry
        assert factory._mapper_registry["Medtronics"] == MedtronicsMapper
        assert factory._mapper_registry["Wichita"] == WichitaMapper
    
    def test_register_mapper(self, mock_logger):
        """Test registering a mapper."""
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Create a test mapper class
        class TestMapper(BaseMapper):
            def map_dataframe(self, df):
                return df
                
            def get_column_mapping(self):
                return {}
        
        # Register the mapper
        factory.register_mapper("Test", TestMapper)
        
        # Verify the mapper was registered
        assert "Test" in factory._mapper_registry
        assert factory._mapper_registry["Test"] == TestMapper
        
        # Verify the logger was called
        mock_logger.info.assert_called_with("Registering mapper for Test: TestMapper")
    
    def test_create_mapper(self, mock_logger):
        """Test creating a mapper."""
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Create a mapper
        mapper = factory.create_mapper("Medtronics")
        
        # Verify the mapper is of the correct type
        assert isinstance(mapper, MedtronicsMapper)
    
    def test_create_mapper_with_logger(self, mock_logger):
        """Test creating a mapper with a custom logger."""
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Create a mapper with a custom logger
        custom_logger = MagicMock()
        mapper = factory.create_mapper("Medtronics", logger=custom_logger)
        
        # Verify the mapper has the custom logger
        assert mapper.logger == custom_logger
    
    def test_create_mapper_unknown_source(self, mock_logger):
        """Test error handling when creating a mapper for an unknown source."""
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Try to create a mapper for an unknown source
        with pytest.raises(MapperFactoryError) as excinfo:
            factory.create_mapper("Unknown")
        
        # Verify the error message
        assert "No mapper registered for source system: Unknown" in str(excinfo.value)
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()
    
    @patch("fca_dashboard.mappers.mapper_factory.settings")
    def test_initialize_registry_with_settings(self, mock_settings, mock_logger):
        """Test initializing the registry with settings."""
        # Set up the mock settings
        mock_settings.get.return_value = {
            "TestSystem": "fca_dashboard.mappers.medtronics_mapper.MedtronicsMapper"
        }
        
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Verify the registry includes the mapper from settings
        assert "TestSystem" in factory._mapper_registry
        assert factory._mapper_registry["TestSystem"] == MedtronicsMapper
        
        # Verify the logger was called
        mock_logger.info.assert_called_with(
            "Registered mapper for TestSystem: fca_dashboard.mappers.medtronics_mapper.MedtronicsMapper"
        )
    
    @patch("fca_dashboard.mappers.mapper_factory.settings")
    def test_initialize_registry_with_invalid_settings(self, mock_settings, mock_logger):
        """Test error handling when initializing the registry with invalid settings."""
        # Set up the mock settings
        mock_settings.get.return_value = {
            "TestSystem": "nonexistent.module.Mapper"
        }
        
        # Create a factory
        factory = MapperFactory(logger=mock_logger)
        
        # Verify the registry does not include the invalid mapper
        assert "TestSystem" not in factory._mapper_registry
        
        # Verify the logger was called
        mock_logger.error.assert_called_once()