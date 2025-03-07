"""
Data Component Adapters

This module provides adapter classes that maintain backward compatibility
between the new pipeline interfaces and the existing data processing functions.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from fca_dashboard.classifier.data_preprocessing import load_and_preprocess_data
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor

# Set up logging
logger = logging.getLogger(__name__)


class LegacyDataLoaderAdapter(DataLoader):
    """
    Adapter for the legacy data loading function.

    This adapter wraps the existing load_and_preprocess_data function
    to make it compatible with the new DataLoader interface.
    """

    def __init__(self, name: str = "LegacyDataLoaderAdapter"):
        """
        Initialize the LegacyDataLoaderAdapter.

        Args:
            name: Component name.
        """
        self._name = name
        self._description = "Adapter for legacy data loading function"
        self._config_provider = ConfigurationProvider()
        logger.info(f"Initialized {name}")

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data using the legacy load_and_preprocess_data function.

        Args:
            data_path: Path to the data file. If None, uses the default path.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        try:
            logger.info(f"Loading data using legacy function from path: {data_path}")
            df = load_and_preprocess_data(data_path)
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data with legacy function: {str(e)}")
            raise ValueError(
                f"Error loading data with legacy function: {str(e)}"
            ) from e

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration.
        """
        try:
            return self._config_provider.config.data.model_dump()
        except Exception as e:
            logger.warning(f"Error getting configuration: {str(e)}")
            return {
                "training_data": {
                    "default_path": "nexusml/data/training_data/fake_training_data.csv",
                    "encoding": "utf-8",
                    "fallback_encoding": "latin1",
                }
            }

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Basic validation - check if required keys exist
        if "training_data" not in config:
            logger.warning("Missing 'training_data' in configuration")
            return False

        training_data = config.get("training_data", {})
        if not isinstance(training_data, dict):
            logger.warning("'training_data' is not a dictionary")
            return False

        if "default_path" not in training_data:
            logger.warning("Missing 'default_path' in training_data configuration")
            return False

        return True


class LegacyDataPreprocessorAdapter(DataPreprocessor):
    """
    Adapter for legacy data preprocessing functionality.

    This adapter provides compatibility with the new DataPreprocessor interface
    while using the existing data preprocessing logic.
    """

    def __init__(self, name: str = "LegacyDataPreprocessorAdapter"):
        """
        Initialize the LegacyDataPreprocessorAdapter.

        Args:
            name: Component name.
        """
        self._name = name
        self._description = "Adapter for legacy data preprocessing functionality"
        self._config_provider = ConfigurationProvider()
        logger.info(f"Initialized {name}")

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data using legacy functionality.

        Since the legacy load_and_preprocess_data function already includes preprocessing,
        this method only performs additional preprocessing steps not covered by the legacy function.

        Args:
            data: Input DataFrame to preprocess.
            **kwargs: Additional arguments for preprocessing.

        Returns:
            Preprocessed DataFrame.

        Raises:
            ValueError: If the data cannot be preprocessed.
        """
        try:
            logger.info(f"Preprocessing data with shape: {data.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            df = data.copy()

            # Apply any additional preprocessing specified in kwargs
            if "drop_duplicates" in kwargs and kwargs["drop_duplicates"]:
                df = df.drop_duplicates()
                logger.debug("Dropped duplicate rows")

            if "drop_columns" in kwargs and isinstance(kwargs["drop_columns"], list):
                columns_to_drop = [
                    col for col in kwargs["drop_columns"] if col in df.columns
                ]
                if columns_to_drop:
                    df = df.drop(columns=columns_to_drop)
                    logger.debug(f"Dropped columns: {columns_to_drop}")

            # Verify required columns
            df = self.verify_required_columns(df)

            logger.info(f"Preprocessing complete. Output shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise ValueError(f"Error during preprocessing: {str(e)}") from e

    def verify_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that all required columns exist in the DataFrame and create them if they don't.

        Args:
            data: Input DataFrame to verify.

        Returns:
            DataFrame with all required columns.

        Raises:
            ValueError: If required columns cannot be created.
        """
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df = data.copy()

            # Get required columns from configuration
            try:
                required_columns = self._config_provider.config.data.required_columns
            except Exception as e:
                logger.warning(
                    f"Error getting required columns from configuration: {str(e)}"
                )
                required_columns = []

            # Check each required column
            for column_info in required_columns:
                column_name = column_info.name
                default_value = column_info.default_value
                data_type = column_info.data_type

                # Check if the column exists
                if column_name not in df.columns:
                    logger.warning(
                        f"Required column '{column_name}' not found. Creating with default value."
                    )

                    # Create the column with the default value
                    if data_type == "str":
                        df[column_name] = default_value
                    elif data_type == "float":
                        df[column_name] = float(default_value)
                    elif data_type == "int":
                        df[column_name] = int(default_value)
                    else:
                        # Default to string if type is unknown
                        logger.warning(
                            f"Unknown data type '{data_type}' for column '{column_name}'"
                        )
                        df[column_name] = default_value

            logger.debug(f"Verified {len(required_columns)} required columns")
            return df

        except Exception as e:
            logger.error(f"Error verifying required columns: {str(e)}")
            raise ValueError(f"Error verifying required columns: {str(e)}") from e

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Basic validation - check if required keys exist
        if "required_columns" not in config:
            logger.warning("Missing 'required_columns' in configuration")
            return False

        return True


class DataComponentFactory:
    """
    Factory for creating data components.

    This factory creates either the new standard components or the legacy adapters
    based on configuration or feature flags.
    """

    @staticmethod
    def create_data_loader(use_legacy: bool = False, **kwargs) -> DataLoader:
        """
        Create a data loader component.

        Args:
            use_legacy: Whether to use the legacy adapter.
            **kwargs: Additional arguments for the component.

        Returns:
            DataLoader implementation.
        """
        if use_legacy:
            logger.info("Creating legacy data loader adapter")
            return LegacyDataLoaderAdapter(**kwargs)
        else:
            logger.info("Creating standard data loader")
            from nexusml.core.pipeline.components.data_loader import StandardDataLoader

            return StandardDataLoader(**kwargs)

    @staticmethod
    def create_data_preprocessor(
        use_legacy: bool = False, **kwargs
    ) -> DataPreprocessor:
        """
        Create a data preprocessor component.

        Args:
            use_legacy: Whether to use the legacy adapter.
            **kwargs: Additional arguments for the component.

        Returns:
            DataPreprocessor implementation.
        """
        if use_legacy:
            logger.info("Creating legacy data preprocessor adapter")
            return LegacyDataPreprocessorAdapter(**kwargs)
        else:
            logger.info("Creating standard data preprocessor")
            from nexusml.core.pipeline.components.data_preprocessor import (
                StandardDataPreprocessor,
            )

            return StandardDataPreprocessor(**kwargs)
