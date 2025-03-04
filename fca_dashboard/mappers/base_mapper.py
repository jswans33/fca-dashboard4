"""
Base mapper module for the FCA Dashboard application.

This module provides the base interface and abstract classes for mapping data
between different formats and schemas.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from fca_dashboard.utils.error_handler import DataTransformationError
from fca_dashboard.utils.logging_config import get_logger


class MappingError(DataTransformationError):
    """Exception raised for errors during data mapping."""
    pass


class BaseMapper(ABC):
    """
    Abstract base class for all mappers.
    
    This class defines the interface that all mappers must implement.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the base mapper.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
        """
        self.logger = logger or get_logger(self.__class__.__name__.lower())
    
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