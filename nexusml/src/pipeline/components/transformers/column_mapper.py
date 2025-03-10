"""
Column Mapper Transformer

This module provides a transformer for mapping columns based on configuration.
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


class ColumnMapper(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping columns based on configuration.

    This transformer maps source columns to target columns based on configuration.
    It can be used for renaming columns, creating copies of columns, or
    standardizing column names across different data sources.
    """

    def __init__(
        self,
        mappings: Optional[List[Dict[str, str]]] = None,
        drop_unmapped: bool = False,
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the ColumnMapper transformer.

        Args:
            mappings: List of column mappings. Each mapping is a dict with:
                - source: Source column name
                - target: Target column name
            drop_unmapped: Whether to drop columns that are not in the mappings.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.mappings = mappings or []
        self.drop_unmapped = drop_unmapped
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        self._valid_mappings = []
        logger.debug(f"Initialized ColumnMapper")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method validates the column mappings against the input data
        and stores valid mappings for later use in transform.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(f"Fitting ColumnMapper on DataFrame with shape: {X.shape}")

            # If mappings not explicitly provided, get from configuration
            if not self.mappings:
                self._load_mappings_from_config()

            # Validate each mapping
            self._valid_mappings = []
            for mapping in self.mappings:
                source = mapping.get("source")
                target = mapping.get("target")

                if not source or not target:
                    logger.warning(
                        f"Invalid mapping: {mapping}. "
                        f"Both source and target must be specified."
                    )
                    continue

                if source not in X.columns:
                    logger.warning(
                        f"Source column '{source}' not found in input data. "
                        f"This mapping will be skipped."
                    )
                    continue

                # Store valid mapping
                self._valid_mappings.append({"source": source, "target": target})
                logger.debug(f"Validated mapping: {source} -> {target}")

            if not self._valid_mappings:
                logger.warning(
                    f"No valid column mappings found. " f"No columns will be mapped."
                )
            else:
                logger.debug(
                    f"Found {len(self._valid_mappings)} valid column mappings."
                )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during ColumnMapper fit: {str(e)}")
            raise ValueError(f"Error during ColumnMapper fit: {str(e)}") from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by mapping columns.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with mapped columns.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "ColumnMapper must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If no valid mappings, return the original DataFrame
            if not self._valid_mappings:
                logger.warning(
                    "No valid mappings to process. Returning original DataFrame."
                )
                return result

            # Apply each mapping
            for mapping in self._valid_mappings:
                source = mapping["source"]
                target = mapping["target"]

                # Skip if source column is not in the DataFrame (should not happen after fit)
                if source not in result.columns:
                    logger.warning(
                        f"Source column '{source}' not found in input data. "
                        f"This mapping will be skipped."
                    )
                    continue

                # Map the column
                result[target] = result[source]
                logger.debug(f"Mapped column: {source} -> {target}")

                # Drop the source column if it's different from the target and drop_unmapped is True
                if self.drop_unmapped and source != target:
                    result = result.drop(columns=[source])
                    logger.debug(f"Dropped source column: {source}")

            # Drop unmapped columns if requested
            if self.drop_unmapped:
                mapped_sources = {m["source"] for m in self._valid_mappings}
                mapped_targets = {m["target"] for m in self._valid_mappings}
                columns_to_keep = mapped_sources.union(mapped_targets)
                columns_to_drop = [
                    col for col in result.columns if col not in columns_to_keep
                ]

                if columns_to_drop:
                    result = result.drop(columns=columns_to_drop)
                    logger.debug(f"Dropped {len(columns_to_drop)} unmapped columns.")

            logger.debug(f"Applied {len(self._valid_mappings)} column mappings.")
            return result
        except Exception as e:
            logger.error(f"Error during ColumnMapper transform: {str(e)}")
            raise ValueError(f"Error during ColumnMapper transform: {str(e)}") from e

    def _load_mappings_from_config(self):
        """
        Load column mappings from the configuration provider.

        This method loads the column mappings from the
        feature engineering section of the configuration.
        """
        try:
            # Get feature engineering configuration
            config = self._config_provider.config
            feature_config = config.feature_engineering

            # Get column mappings
            self.mappings = [
                {"source": mapping.source, "target": mapping.target}
                for mapping in feature_config.column_mappings
            ]

            logger.debug(
                f"Loaded configuration for {len(self.mappings)} column mappings."
            )

            if not self.mappings:
                logger.warning(
                    "No column mappings found in configuration. "
                    "No columns will be mapped."
                )
        except Exception as e:
            logger.error(f"Error loading configuration for ColumnMapper: {str(e)}")
            self.mappings = []
