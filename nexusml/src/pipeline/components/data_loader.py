"""
Standard Data Loader Component

This module provides a standard implementation of the DataLoader interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseDataLoader

# Set up logging
logger = logging.getLogger(__name__)


class StandardDataLoader(BaseDataLoader):
    """
    Standard implementation of the DataLoader interface.

    This class loads data from various sources based on configuration
    provided by the ConfigurationProvider. It handles error cases gracefully
    and provides detailed logging.
    """

    def __init__(
        self,
        name: str = "StandardDataLoader",
        description: str = "Standard data loader using unified configuration",
    ):
        """
        Initialize the StandardDataLoader.

        Args:
            name: Component name.
            description: Component description.
        """
        super().__init__(name, description)
        self._config_provider = ConfigurationProvider()
        logger.info(f"Initialized {name}")

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path or from the configuration.

        Args:
            data_path: Path to the data file. If None, uses the path from configuration.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        try:
            # If no path is provided, use the one from configuration or discover available files
            if data_path is None:
                if kwargs.get("discover_files", False):
                    # Discover available files and select the first one
                    available_files = self.discover_data_files()
                    if not available_files:
                        raise FileNotFoundError(
                            "No data files found in the default search paths"
                        )

                    # Select the first file by default
                    file_name = kwargs.get("file_name", list(available_files.keys())[0])
                    data_path = available_files.get(file_name)

                    if data_path is None:
                        raise FileNotFoundError(f"File not found: {file_name}")

                    logger.info(f"Selected data file: {file_name}")
                else:
                    # Use the default path from configuration
                    config = self._config_provider.config
                    data_config = config.data.training_data
                    data_path = data_config.default_path
                    logger.info(
                        f"Using default data path from configuration: {data_path}"
                    )

            # Resolve the path
            resolved_path = self._resolve_path(data_path)
            logger.info(f"Loading data from: {resolved_path}")

            # Determine file type and load accordingly
            file_extension = Path(resolved_path).suffix.lower()

            if file_extension == ".csv":
                # Get encoding settings from configuration
                encoding = self._config_provider.config.data.training_data.encoding
                fallback_encoding = (
                    self._config_provider.config.data.training_data.fallback_encoding
                )

                # Try to load the data with the primary encoding
                try:
                    logger.debug(f"Attempting to load CSV with {encoding} encoding")
                    df = pd.read_csv(resolved_path, encoding=encoding)
                except UnicodeDecodeError:
                    # Try with fallback encoding
                    logger.warning(
                        f"Failed to load data with {encoding} encoding. "
                        f"Trying fallback encoding: {fallback_encoding}"
                    )
                    df = pd.read_csv(resolved_path, encoding=fallback_encoding)

            elif file_extension == ".xlsx" or file_extension == ".xls":
                logger.debug(f"Loading Excel file: {resolved_path}")
                df = pd.read_excel(resolved_path)

            elif file_extension == ".json":
                logger.debug(f"Loading JSON file: {resolved_path}")
                df = pd.read_json(resolved_path)

            else:
                logger.warning(
                    f"Unknown file extension: {file_extension}, attempting to load as CSV"
                )
                encoding = self._config_provider.config.data.training_data.encoding
                df = pd.read_csv(resolved_path, encoding=encoding)

            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df

        except FileNotFoundError as e:
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}") from e
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data file: {data_path}")
            raise ValueError(f"Empty data file: {data_path}") from e
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing data file: {data_path}")
            raise ValueError(f"Error parsing data file: {data_path}") from e
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise ValueError(f"Unexpected error loading data: {str(e)}") from e

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration.
        """
        return self._config_provider.config.data.model_dump()

    def discover_data_files(
        self,
        search_paths: Optional[List[str]] = None,
        file_extensions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Discover available data files in the specified paths.

        Args:
            search_paths: List of paths to search for data files. If None, uses default paths.
            file_extensions: List of file extensions to include. If None, uses ['.csv', '.xlsx', '.xls', '.json'].

        Returns:
            Dictionary mapping file names to their full paths.
        """
        if file_extensions is None:
            file_extensions = [".csv", ".xlsx", ".xls", ".json"]

        if search_paths is None:
            # Use default search paths
            project_root = self._get_project_root()
            search_paths = [
                os.path.join(project_root, "examples"),
                os.path.join(project_root, "data"),
                os.path.join(os.path.dirname(project_root), "examples"),
                os.path.join(os.path.dirname(project_root), "uploads"),
            ]

        data_files = {}
        for path in search_paths:
            if os.path.exists(path):
                for file in os.listdir(path):
                    file_path = os.path.join(path, file)
                    if os.path.isfile(file_path) and any(
                        file.lower().endswith(ext) for ext in file_extensions
                    ):
                        data_files[file] = file_path

        logger.info(f"Discovered {len(data_files)} data files")
        for file_name, file_path in data_files.items():
            logger.debug(f"  - {file_name}: {file_path}")

        return data_files

    def list_available_data_files(self) -> List[Tuple[str, str]]:
        """
        List all available data files in the default search paths.

        Returns:
            List of tuples containing (file_name, file_path) for each available data file.
        """
        data_files = self.discover_data_files()
        return [(file_name, file_path) for file_name, file_path in data_files.items()]

    def _get_project_root(self) -> str:
        """
        Get the absolute path to the project root directory.

        Returns:
            Absolute path to the project root directory.
        """
        # The package root is 4 levels up from this file:
        # nexusml/core/pipeline/components/data_loader.py
        return str(Path(__file__).resolve().parent.parent.parent.parent)

    def _resolve_path(self, data_path: str) -> str:
        """
        Resolve the data path to an absolute path.

        Args:
            data_path: Path to resolve.

        Returns:
            Resolved absolute path.
        """
        path = Path(data_path)

        # If the path is already absolute, return it
        if path.is_absolute():
            return str(path)

        # Try to resolve relative to the current working directory
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return str(cwd_path)

        # Try to resolve relative to the package root
        package_root = Path(self._get_project_root())
        package_path = package_root / path
        if package_path.exists():
            return str(package_path)

        # Try to resolve relative to the parent of the package root
        parent_path = package_root.parent / path
        if parent_path.exists():
            return str(parent_path)

        # If we can't resolve it, return the original path and let the caller handle it
        logger.warning(f"Could not resolve path: {data_path}")
        return data_path
