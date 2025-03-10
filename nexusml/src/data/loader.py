"""
Data Loader Implementation

This module provides implementations for loading data from various sources.
"""

from typing import Any, Dict, Optional

import pandas as pd


class DataLoader:
    """Base class for data loaders."""

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.

        Args:
            path: Path to the data file.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.
        """
        raise NotImplementedError("Subclasses must implement load_data")


class CSVDataLoader(DataLoader):
    """Data loader for CSV files."""

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments for pd.read_csv.

        Returns:
            DataFrame containing the loaded data.
        """
        return pd.read_csv(path, **kwargs)


class ExcelDataLoader(DataLoader):
    """Data loader for Excel files."""

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.

        Args:
            path: Path to the Excel file.
            **kwargs: Additional arguments for pd.read_excel.

        Returns:
            DataFrame containing the loaded data.
        """
        return pd.read_excel(path, **kwargs)


class ConfigurableDataLoader(DataLoader):
    """Configurable data loader that can load data from various sources."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the configurable data loader.

        Args:
            config: Configuration dictionary.
        """
        self.config = config or {}
        self._loaders = {
            "csv": CSVDataLoader(),
            "excel": ExcelDataLoader(),
        }

    def load_data(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from a source determined by the file extension or configuration.

        Args:
            path: Path to the data file.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.
        """
        # Determine the loader to use based on file extension
        if path.lower().endswith(".csv"):
            return self._loaders["csv"].load_data(path, **kwargs)
        elif path.lower().endswith((".xls", ".xlsx")):
            return self._loaders["excel"].load_data(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {path}")
