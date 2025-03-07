"""
Standard Data Loader Component

This module provides a standard implementation of the DataLoader interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

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
            # If no path is provided, use the one from configuration
            if data_path is None:
                config = self._config_provider.config
                data_config = config.data.training_data
                data_path = data_config.default_path
                logger.info(f"Using default data path from configuration: {data_path}")

            # Resolve the path
            resolved_path = self._resolve_path(data_path)
            logger.info(f"Loading data from: {resolved_path}")

            # Get encoding settings from configuration
            encoding = self._config_provider.config.data.training_data.encoding
            fallback_encoding = (
                self._config_provider.config.data.training_data.fallback_encoding
            )

            # Try to load the data with the primary encoding
            try:
                logger.debug(f"Attempting to load data with {encoding} encoding")
                df = pd.read_csv(resolved_path, encoding=encoding)
            except UnicodeDecodeError:
                # Try with fallback encoding
                logger.warning(
                    f"Failed to load data with {encoding} encoding. "
                    f"Trying fallback encoding: {fallback_encoding}"
                )
                df = pd.read_csv(resolved_path, encoding=fallback_encoding)

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
        package_root = Path(__file__).resolve().parent.parent.parent.parent
        package_path = package_root / path
        if package_path.exists():
            return str(package_path)

        # If we can't resolve it, return the original path and let the caller handle it
        logger.warning(f"Could not resolve path: {data_path}")
        return data_path
