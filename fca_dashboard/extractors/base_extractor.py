"""
Base extractor module for the FCA Dashboard application.

This module provides the base classes and interfaces for data extraction
from various file formats.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import get_root_dir, resolve_path


class ExtractionError(FCADashboardError):
    """Exception raised for errors during data extraction."""
    pass


class DataExtractor(ABC):
    """Base class for all data extractors."""
    
    def __init__(self, logger=None):
        """
        Initialize the extractor with an optional logger.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
    
    @abstractmethod
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if this extractor can handle the given file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if this extractor can handle the file, False otherwise.
        """
        pass
    
    @abstractmethod
    def extract(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from the file into a pandas DataFrame.
        
        Args:
            file_path: Path to the file to extract data from.
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If an error occurs during extraction.
        """
        pass
    
    def extract_and_save(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data and optionally save it to a file.
        
        Args:
            file_path: Path to the file to extract data from.
            output_path: Optional path to save the extracted data to.
                If None, the data is not saved.
            output_format: Format to save the data in (default: "csv").
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If an error occurs during extraction or saving.
        """
        # Extract the data
        df = self.extract(file_path, **kwargs)
        
        # Save the data if an output path is provided
        if output_path:
            output_path = resolve_path(output_path)
            
            # Create the directory if it doesn't exist
            os.makedirs(output_path.parent, exist_ok=True)
            
            try:
                if output_format.lower() == "csv":
                    df.to_csv(output_path, index=False)
                elif output_format.lower() in ["xlsx", "excel"]:
                    df.to_excel(output_path, index=False)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")
                
                self.logger.info(f"Saved extracted data to {output_path}")
            except Exception as e:
                error_msg = f"Error saving data to {output_path}: {str(e)}"
                self.logger.error(error_msg)
                raise ExtractionError(error_msg) from e
        
        return df


class ExtractorFactory:
    """Factory for creating extractors based on file type."""
    
    def __init__(self):
        """Initialize the factory with an empty list of extractors."""
        self.extractors: List[DataExtractor] = []
    
    def register_extractor(self, extractor: DataExtractor) -> None:
        """
        Register an extractor with the factory.
        
        Args:
            extractor: The extractor to register.
        """
        self.extractors.append(extractor)
    
    def get_extractor(self, file_path: Union[str, Path]) -> Optional[DataExtractor]:
        """
        Get an appropriate extractor for the given file.
        
        Args:
            file_path: Path to the file to extract data from.
            
        Returns:
            An appropriate extractor, or None if no suitable extractor is found.
        """
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None
    
    def extract(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data from the file using an appropriate extractor.
        
        Args:
            file_path: Path to the file to extract data from.
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If no suitable extractor is found or an error occurs during extraction.
        """
        extractor = self.get_extractor(file_path)
        if not extractor:
            raise ExtractionError(f"No suitable extractor found for {file_path}")
        
        return extractor.extract(file_path, **kwargs)
    
    def extract_and_save(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        output_format: str = "csv",
        **kwargs
    ) -> pd.DataFrame:
        """
        Extract data and optionally save it to a file.
        
        Args:
            file_path: Path to the file to extract data from.
            output_path: Optional path to save the extracted data to.
                If None, the data is not saved.
            output_format: Format to save the data in (default: "csv").
            **kwargs: Additional extraction options.
            
        Returns:
            pandas DataFrame containing the extracted data.
            
        Raises:
            ExtractionError: If no suitable extractor is found or an error occurs during extraction or saving.
        """
        extractor = self.get_extractor(file_path)
        if not extractor:
            raise ExtractionError(f"No suitable extractor found for {file_path}")
        
        return extractor.extract_and_save(file_path, output_path, output_format, **kwargs)


# Create a global factory instance
extractor_factory = ExtractorFactory()