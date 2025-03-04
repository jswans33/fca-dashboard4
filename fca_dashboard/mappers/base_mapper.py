"""
Base mapper module for the FCA Dashboard application.

This module provides the base interface and abstract classes for mapping data
between different formats and schemas.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from fca_dashboard.utils.error_handler import DataTransformationError
from fca_dashboard.utils.logging_config import get_logger


class MappingError(DataTransformationError):
    """Exception raised for errors during data mapping."""
    pass


class ValidationError(MappingError):
    """Exception raised for validation errors during mapping."""
    pass


class BaseMapper(ABC):
    """
    Abstract base class for all mappers.
    
    This class defines the interface that all mappers must implement.
    """
    
    def __init__(self, logger=None, required_source_columns=None, required_dest_columns=None):
        """
        Initialize the base mapper.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
            required_source_columns: Optional list of required source columns.
            required_dest_columns: Optional list of required destination columns.
        """
        self.logger = logger or get_logger(self.__class__.__name__.lower())
        self.required_source_columns = required_source_columns or []
        self.required_dest_columns = required_dest_columns or []
    
    def validate_source_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the source DataFrame has the required columns.
        
        Args:
            df: The source DataFrame to validate.
            
        Raises:
            ValidationError: If the DataFrame is missing required columns.
        """
        if df is None:
            raise ValidationError("Source DataFrame is None")
        
        if df.empty:
            self.logger.warning("Source DataFrame is empty")
            return  # Skip column validation for empty DataFrames
        
        # Check for required columns
        if self.required_source_columns:
            # Normalize column names by replacing spaces with underscores for comparison
            normalized_columns = [col.replace(' ', '_') if isinstance(col, str) else col for col in df.columns]
            missing_columns = [col for col in self.required_source_columns
                              if col not in df.columns and col not in normalized_columns]
            if missing_columns:
                error_msg = f"Source DataFrame is missing required columns: {missing_columns}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
    
    def validate_mapped_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate that the mapped DataFrame has the required columns.
        
        Args:
            df: The mapped DataFrame to validate.
            
        Raises:
            ValidationError: If the DataFrame is missing required columns.
        """
        if df is None:
            raise ValidationError("Mapped DataFrame is None")
        
        if df.empty:
            self.logger.warning("Mapped DataFrame is empty")
            return  # Skip column validation for empty DataFrames
        
        # Check for required columns
        if self.required_dest_columns:
            missing_columns = [col for col in self.required_dest_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Mapped DataFrame is missing required columns: {missing_columns}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg)
    
    def get_required_source_columns(self) -> List[str]:
        """
        Get the list of required source columns.
        
        Returns:
            A list of required source column names.
        """
        return self.required_source_columns
    
    def get_required_dest_columns(self) -> List[str]:
        """
        Get the list of required destination columns.
        
        Returns:
            A list of required destination column names.
        """
        return self.required_dest_columns
    
    def set_required_source_columns(self, columns: List[str]) -> None:
        """
        Set the list of required source columns.
        
        Args:
            columns: A list of required source column names.
        """
        self.required_source_columns = columns
    
    def set_required_dest_columns(self, columns: List[str]) -> None:
        """
        Set the list of required destination columns.
        
        Args:
            columns: A list of required destination column names.
        """
        self.required_dest_columns = columns
    
    @abstractmethod
    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map a DataFrame from source format to destination format.
        
        Args:
            df: The source DataFrame to map.
            
        Returns:
            The mapped DataFrame.
            
        Raises:
            MappingError: If an error occurs during mapping.
            ValidationError: If the DataFrame fails validation.
        """
        pass
    
    @abstractmethod
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get the column mapping from source to destination.
        
        Returns:
            A dictionary mapping source columns to destination columns.
        """
        pass