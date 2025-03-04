"""
Unit tests for the base mapper module.

This module contains tests for the BaseMapper class and related functionality
in the fca_dashboard.mappers.base_mapper module.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from fca_dashboard.mappers.base_mapper import BaseMapper, MappingError


class TestBaseMapper:
    """Tests for the BaseMapper class."""
    
    def test_init_with_logger(self):
        """Test initialization with a custom logger."""
        # Create a mock logger
        mock_logger = MagicMock()
        
        # Create a concrete implementation of BaseMapper for testing
        class TestMapper(BaseMapper):
            def map_dataframe(self, df):
                return df
                
            def get_column_mapping(self):
                return {}
        
        # Initialize with the mock logger
        mapper = TestMapper(logger=mock_logger)
        
        # Verify the logger was set correctly
        assert mapper.logger == mock_logger
    
    def test_init_without_logger(self):
        """Test initialization without a logger."""
        # Create a concrete implementation of BaseMapper for testing
        class TestMapper(BaseMapper):
            def map_dataframe(self, df):
                return df
                
            def get_column_mapping(self):
                return {}
        
        # Initialize without a logger
        mapper = TestMapper()
        
        # Verify a logger was created
        assert mapper.logger is not None
        # The logger might not have a name attribute depending on the implementation
        # Just verify it exists
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        # Attempt to instantiate BaseMapper directly
        with pytest.raises(TypeError):
            BaseMapper()
        
        # Create a partial implementation
        class PartialMapper(BaseMapper):
            def map_dataframe(self, df):
                return df
        
        # Attempt to instantiate the partial implementation
        with pytest.raises(TypeError):
            PartialMapper()
    
    def test_mapping_error(self):
        """Test the MappingError exception."""
        # Create a MappingError
        error = MappingError("Test error message")
        
        # Verify it's a subclass of DataTransformationError
        from fca_dashboard.utils.error_handler import DataTransformationError
        assert isinstance(error, DataTransformationError)
        
        # Verify the error message
        assert str(error) == "Test error message"