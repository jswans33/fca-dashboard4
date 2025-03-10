"""
Standard Data Preprocessor Component

This module provides a standard implementation of the DataPreprocessor interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseDataPreprocessor

# Set up logging
logger = logging.getLogger(__name__)


class StandardDataPreprocessor(BaseDataPreprocessor):
    """
    Standard implementation of the DataPreprocessor interface.

    This class preprocesses data based on configuration provided by the
    ConfigurationProvider. It handles error cases gracefully and provides
    detailed logging.
    """

    def __init__(
        self,
        name: str = "StandardDataPreprocessor",
        description: str = "Standard data preprocessor using unified configuration",
    ):
        """
        Initialize the StandardDataPreprocessor.

        Args:
            name: Component name.
            description: Component description.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = ConfigurationProvider()
        # Update the config from the provider
        self.config = self._config_provider.config.data.model_dump()
        logger.info(f"Initialized {name}")

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data.

        This method performs several preprocessing steps:
        1. Cleans column names (removes whitespace)
        2. Fills NaN values appropriately based on data type
        3. Verifies and creates required columns
        4. Applies any additional preprocessing specified in kwargs

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

            # Clean up column names (remove any leading/trailing whitespace)
            df.columns = [col.strip() for col in df.columns]
            logger.debug("Cleaned column names")

            # Fill NaN values appropriately based on data type
            self._fill_na_values(df)
            logger.debug("Filled NaN values")

            # Verify and create required columns
            df = self.verify_required_columns(df)
            logger.debug("Verified required columns")

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
            required_columns = self._get_required_columns()

            # Check each required column
            for column_info in required_columns:
                column_name = column_info["name"]
                default_value = column_info["default_value"]
                data_type = column_info["data_type"]

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

    def _fill_na_values(self, df: pd.DataFrame) -> None:
        """
        Fill NaN values in the DataFrame based on column data types.

        Args:
            df: DataFrame to fill NaN values in (modified in-place).
        """
        # Fill NaN values with empty strings for text columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("")

        # Fill NaN values with 0 for numeric columns
        for col in df.select_dtypes(include=["number"]).columns:
            df[col] = df[col].fillna(0)

        # Fill NaN values with False for boolean columns
        for col in df.select_dtypes(include=["bool"]).columns:
            df[col] = df[col].fillna(False)

    def _get_required_columns(self) -> List[Dict[str, Any]]:
        """
        Get the list of required columns from the configuration.

        Returns:
            List of dictionaries containing required column information.
        """
        try:
            # Get required columns from configuration
            required_columns = self.config.get("required_columns", [])

            # If it's not a list or is empty, log a warning
            if not isinstance(required_columns, list) or not required_columns:
                logger.warning("No required columns found in configuration")
                return []

            return required_columns

        except Exception as e:
            logger.error(f"Error getting required columns from configuration: {str(e)}")
            return []
