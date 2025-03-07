"""
Numeric Cleaner Transformer

This module provides a transformer for cleaning and transforming numeric columns.
It follows the scikit-learn transformer interface and uses the configuration system.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.core.config.provider import ConfigurationProvider

# Set up logging
logger = logging.getLogger(__name__)


class NumericCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer for cleaning and transforming numeric columns.

    This transformer handles numeric columns by:
    - Filling missing values with configurable defaults
    - Converting to specified data types
    - Optionally renaming columns
    """

    def __init__(
        self,
        columns: Optional[List[Dict[str, Any]]] = None,
        fill_value: Union[int, float] = 0,
        dtype: str = "float",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the NumericCleaner transformer.

        Args:
            columns: List of column configurations. Each configuration is a dict with:
                - name: Original column name
                - new_name: New column name (optional)
                - fill_value: Value to use for missing data
                - dtype: Data type for the column
            fill_value: Default value to use for missing data if not specified in columns.
            dtype: Default data type to use if not specified in columns.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.columns = columns or []  # Initialize as empty list if None
        self.fill_value = fill_value
        self.dtype = dtype
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        self._column_configs = []
        logger.debug(f"Initialized NumericCleaner")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method identifies which of the specified columns are available
        in the input data and stores their configurations for later use in transform.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(f"Fitting NumericCleaner on DataFrame with shape: {X.shape}")

            # If columns list is empty, get from configuration
            if not self.columns:
                self._load_columns_from_config()

            # Process each column configuration
            self._column_configs = []
            for col_config in self.columns:
                col_name = col_config.get("name")
                if col_name in X.columns:
                    self._column_configs.append(
                        {
                            "name": col_name,
                            "new_name": col_config.get("new_name"),
                            "fill_value": col_config.get("fill_value", self.fill_value),
                            "dtype": col_config.get("dtype", self.dtype),
                        }
                    )
                else:
                    logger.warning(
                        f"Column '{col_name}' not found in input data. Skipping."
                    )

            if not self._column_configs:
                logger.warning(
                    f"None of the specified columns are available in the input data. "
                    f"No columns will be processed."
                )
            else:
                logger.debug(
                    f"Found {len(self._column_configs)} of {len(self.columns)} "
                    f"specified columns in the input data."
                )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during NumericCleaner fit: {str(e)}")
            raise ValueError(f"Error during NumericCleaner fit: {str(e)}") from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by cleaning numeric columns.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with cleaned numeric columns.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "NumericCleaner must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If no columns to process, return the original DataFrame
            if not self._column_configs:
                logger.warning("No columns to process. Returning original DataFrame.")
                return result

            # Process each column
            for config in self._column_configs:
                col_name = config["name"]
                new_name = config.get("new_name")
                fill_value = config["fill_value"]
                dtype_str = config["dtype"]

                # Fill missing values
                result[col_name] = result[col_name].fillna(fill_value)

                # Convert to specified data type
                try:
                    if dtype_str == "float":
                        result[col_name] = result[col_name].astype(float)
                    elif dtype_str == "int":
                        # Convert to float first to handle NaN values, then to int
                        result[col_name] = result[col_name].astype(float).astype(int)
                    else:
                        logger.warning(
                            f"Unsupported data type '{dtype_str}' for column '{col_name}'. "
                            f"Using string conversion."
                        )
                        result[col_name] = result[col_name].astype(str)
                except Exception as e:
                    logger.error(
                        f"Error converting column '{col_name}' to type '{dtype_str}': {str(e)}. "
                        f"Using original values."
                    )

                # Rename column if specified
                if new_name and new_name != col_name:
                    result[new_name] = result[col_name]
                    # Only drop the original column if it's not used by another configuration
                    if not any(
                        c["name"] == col_name and c.get("new_name") != new_name
                        for c in self._column_configs
                    ):
                        result = result.drop(columns=[col_name])
                    logger.debug(f"Renamed column '{col_name}' to '{new_name}'")

            logger.debug(f"Processed {len(self._column_configs)} numeric columns.")
            return result
        except Exception as e:
            logger.error(f"Error during NumericCleaner transform: {str(e)}")
            raise ValueError(f"Error during NumericCleaner transform: {str(e)}") from e

    def _load_columns_from_config(self):
        """
        Load column configuration from the configuration provider.

        This method loads the numeric column configuration from the
        feature engineering section of the configuration.
        """
        try:
            # Get feature engineering configuration
            config = self._config_provider.config
            feature_config = config.feature_engineering

            # Get numeric column configurations
            self.columns = [
                {
                    "name": col.name,
                    "new_name": col.new_name,
                    "fill_value": col.fill_value,
                    "dtype": col.dtype,
                }
                for col in feature_config.numeric_columns
            ]

            logger.debug(
                f"Loaded configuration for {len(self.columns)} numeric columns."
            )

            if not self.columns:
                logger.warning(
                    "No numeric column configurations found in configuration. "
                    "No columns will be processed."
                )
        except Exception as e:
            logger.error(f"Error loading configuration for NumericCleaner: {str(e)}")
            self.columns = []
