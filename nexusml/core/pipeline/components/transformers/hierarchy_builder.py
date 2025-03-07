"""
Hierarchy Builder Transformer

This module provides a transformer for creating hierarchical category fields.
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


class HierarchyBuilder(BaseEstimator, TransformerMixin):
    """
    Transformer for creating hierarchical category fields.

    This transformer creates new columns by combining parent columns in a hierarchical
    structure using a configurable separator. It handles missing values gracefully
    and provides detailed logging.
    """

    def __init__(
        self,
        hierarchies: Optional[List[Dict[str, Any]]] = None,
        separator: str = "-",
        fill_na: str = "",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the HierarchyBuilder transformer.

        Args:
            hierarchies: List of hierarchy configurations. Each configuration is a dict with:
                - new_col: Name of the new hierarchical column
                - parents: List of parent columns in hierarchy order
                - separator: Separator to use between hierarchy levels
            separator: Default separator to use if not specified in hierarchies.
            fill_na: Value to use for filling NaN values before combining.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.hierarchies = hierarchies or []
        self.separator = separator
        self.fill_na = fill_na
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        self._valid_hierarchies = []
        logger.debug(f"Initialized HierarchyBuilder")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method validates the hierarchy configurations against the input data
        and stores valid configurations for later use in transform.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(f"Fitting HierarchyBuilder on DataFrame with shape: {X.shape}")

            # If hierarchies not explicitly provided, get from configuration
            if not self.hierarchies:
                self._load_hierarchies_from_config()

            # Validate each hierarchy configuration
            self._valid_hierarchies = []
            for hierarchy in self.hierarchies:
                new_col = hierarchy.get("new_col")
                parents = hierarchy.get("parents", [])
                separator = hierarchy.get("separator", self.separator)

                # Check if all parent columns exist in the input data
                missing_parents = [col for col in parents if col not in X.columns]

                if missing_parents:
                    logger.warning(
                        f"Hierarchy '{new_col}' has missing parent columns: {missing_parents}. "
                        f"This hierarchy will be skipped."
                    )
                    continue

                if not parents:
                    logger.warning(
                        f"Hierarchy '{new_col}' has no parent columns specified. "
                        f"This hierarchy will be skipped."
                    )
                    continue

                # Store valid hierarchy configuration
                self._valid_hierarchies.append(
                    {"new_col": new_col, "parents": parents, "separator": separator}
                )
                logger.debug(
                    f"Validated hierarchy '{new_col}' with {len(parents)} parent columns: {parents}"
                )

            if not self._valid_hierarchies:
                logger.warning(
                    f"No valid hierarchy configurations found. "
                    f"No hierarchical columns will be created."
                )
            else:
                logger.debug(
                    f"Found {len(self._valid_hierarchies)} valid hierarchy configurations."
                )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during HierarchyBuilder fit: {str(e)}")
            raise ValueError(f"Error during HierarchyBuilder fit: {str(e)}") from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by creating hierarchical columns.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with the hierarchical columns added.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "HierarchyBuilder must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If no valid hierarchies, return the original DataFrame
            if not self._valid_hierarchies:
                logger.warning(
                    "No valid hierarchies to process. Returning original DataFrame."
                )
                return result

            # Process each hierarchy
            for hierarchy in self._valid_hierarchies:
                new_col = hierarchy["new_col"]
                parents = hierarchy["parents"]
                separator = hierarchy["separator"]

                # Fill NaN values in parent columns
                for col in parents:
                    result[col] = result[col].fillna(self.fill_na)

                # Create the hierarchical column
                try:
                    # Convert all parent columns to strings and combine them
                    result[new_col] = (
                        result[parents]
                        .astype(str)
                        .apply(lambda row: separator.join(row), axis=1)
                    )
                    logger.debug(
                        f"Created hierarchical column '{new_col}' from parent columns: {parents}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error creating hierarchical column '{new_col}': {str(e)}. "
                        f"Skipping this hierarchy."
                    )
                    continue

            logger.debug(
                f"Created {len(self._valid_hierarchies)} hierarchical columns."
            )
            return result
        except Exception as e:
            logger.error(f"Error during HierarchyBuilder transform: {str(e)}")
            raise ValueError(
                f"Error during HierarchyBuilder transform: {str(e)}"
            ) from e

    def _load_hierarchies_from_config(self):
        """
        Load hierarchy configurations from the configuration provider.

        This method loads the hierarchy configurations from the
        feature engineering section of the configuration.
        """
        try:
            # Get feature engineering configuration
            config = self._config_provider.config
            feature_config = config.feature_engineering

            # Get hierarchy configurations
            self.hierarchies = [
                {"new_col": h.new_col, "parents": h.parents, "separator": h.separator}
                for h in feature_config.hierarchies
            ]

            logger.debug(
                f"Loaded configuration for {len(self.hierarchies)} hierarchies."
            )

            if not self.hierarchies:
                logger.warning(
                    "No hierarchy configurations found in configuration. "
                    "No hierarchical columns will be created."
                )
        except Exception as e:
            logger.error(f"Error loading configuration for HierarchyBuilder: {str(e)}")
            self.hierarchies = []
