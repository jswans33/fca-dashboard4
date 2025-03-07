"""
Text Combiner Transformer

This module provides a transformer for combining multiple text fields into a single field.
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


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Transformer for combining multiple text fields into a single field.

    This transformer combines specified text columns into a single column
    using a configurable separator. It handles missing values gracefully
    and provides detailed logging.
    """

    def __init__(
        self,
        name: str = "combined_features",
        columns: Optional[List[str]] = None,
        separator: str = " ",
        fill_na: str = "",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the TextCombiner transformer.

        Args:
            name: Name of the output combined column.
            columns: List of columns to combine. If None, uses configuration.
            separator: String to use as separator between combined fields.
            fill_na: Value to use for filling NaN values before combining.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.name = name
        self.columns = columns or []  # Initialize as empty list if None
        self.separator = separator
        self.fill_na = fill_na
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        self._available_columns = []
        logger.debug(f"Initialized TextCombiner with output column: {self.name}")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method identifies which of the specified columns are available
        in the input data and stores them for later use in transform.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(f"Fitting TextCombiner on DataFrame with shape: {X.shape}")

            # If columns list is empty, get from configuration
            if not self.columns:
                self._load_columns_from_config()

            # Identify which columns are available in the input data
            self._available_columns = [col for col in self.columns if col in X.columns]

            if not self._available_columns:
                logger.warning(
                    f"None of the specified columns {self.columns} are available in the input data. "
                    f"The output column {self.name} will contain empty strings."
                )
            else:
                logger.debug(
                    f"Found {len(self._available_columns)} of {len(self.columns)} "
                    f"specified columns in the input data."
                )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during TextCombiner fit: {str(e)}")
            raise ValueError(f"Error during TextCombiner fit: {str(e)}") from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by combining text columns.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with the combined text column added.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "TextCombiner must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If no available columns, create an empty column
            if not self._available_columns:
                result[self.name] = self.fill_na
                logger.warning(
                    f"Created empty column {self.name} as no source columns were available."
                )
                return result

            # Fill NaN values in the columns to be combined
            for col in self._available_columns:
                if col in result.columns:
                    result[col] = result[col].fillna(self.fill_na)

            # Combine the columns
            result[self.name] = (
                result[self._available_columns]
                .astype(str)
                .apply(lambda row: self.separator.join(row), axis=1)
            )

            logger.debug(
                f"Created combined column {self.name} from {len(self._available_columns)} columns."
            )
            return result
        except Exception as e:
            logger.error(f"Error during TextCombiner transform: {str(e)}")
            raise ValueError(f"Error during TextCombiner transform: {str(e)}") from e

    def _load_columns_from_config(self):
        """
        Load column configuration from the configuration provider.

        This method loads the text combination configuration from the
        feature engineering section of the configuration.
        """
        try:
            # Get feature engineering configuration
            config = self._config_provider.config
            feature_config = config.feature_engineering

            # Find the text combination configuration for this output column
            for combo in feature_config.text_combinations:
                if combo.name == self.name:
                    self.columns = combo.columns
                    self.separator = combo.separator
                    logger.debug(
                        f"Loaded configuration for {self.name}: "
                        f"{len(self.columns)} columns with separator '{self.separator}'"
                    )
                    return

            # If no matching configuration found, use default columns
            logger.warning(
                f"No configuration found for text combination {self.name}. "
                f"Using default configuration."
            )
            self.columns = []
        except Exception as e:
            logger.error(f"Error loading configuration for TextCombiner: {str(e)}")
            self.columns = []
