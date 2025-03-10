"""
Classification System Mapper Transformer

This module provides a transformer for mapping between different classification systems.
It follows the scikit-learn transformer interface and uses the configuration system.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.core.config.provider import ConfigurationProvider

# Set up logging
logger = logging.getLogger(__name__)


class ClassificationSystemMapper(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping between different classification systems.

    This transformer maps between different classification systems (e.g., OmniClass,
    MasterFormat, UniFormat) based on configuration or custom mapping functions.
    """

    def __init__(
        self,
        source_column: str,
        target_column: str,
        mapping_type: str = "direct",
        mapping_function: Optional[Callable[[str], str]] = None,
        mapping_dict: Optional[Dict[str, str]] = None,
        default_value: str = "00 00 00",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the ClassificationSystemMapper transformer.

        Args:
            source_column: Source column containing classification codes.
            target_column: Target column to store the mapped classifications.
            mapping_type: Type of mapping to use ('direct', 'function', 'eav').
            mapping_function: Custom function for mapping classifications.
            mapping_dict: Dictionary mapping source codes to target codes.
            default_value: Default value if no mapping is found.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.source_column = source_column
        self.target_column = target_column
        self.mapping_type = mapping_type
        self.mapping_function = mapping_function
        self.mapping_dict = mapping_dict or {}
        self.default_value = default_value
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        logger.debug(
            f"Initialized ClassificationSystemMapper for {source_column} -> {target_column} "
            f"using {mapping_type} mapping"
        )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method validates the source column and loads mapping configuration.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(
                f"Fitting ClassificationSystemMapper on DataFrame with shape: {X.shape}"
            )

            # Check if source column exists
            if self.source_column not in X.columns:
                logger.warning(
                    f"Source column '{self.source_column}' not found in input data. "
                    f"Mappings will default to '{self.default_value}'."
                )

            # If mapping dictionary not explicitly provided and mapping type is direct, load from configuration
            if not self.mapping_dict and self.mapping_type == "direct":
                self._load_mappings_from_config()

            # If mapping function not provided and mapping type is function, use enhanced_masterformat_mapping
            if not self.mapping_function and self.mapping_type == "function":
                logger.warning(
                    f"No mapping function provided for function mapping type. "
                    f"Mappings will default to '{self.default_value}'."
                )

            # If mapping type is eav, check if EAV integration is enabled
            if self.mapping_type == "eav":
                try:
                    config = self._config_provider.config
                    eav_enabled = config.feature_engineering.eav_integration.enabled
                    if not eav_enabled:
                        logger.warning(
                            f"EAV integration is disabled in configuration. "
                            f"Mappings will default to '{self.default_value}'."
                        )
                except Exception:
                    logger.warning(
                        f"Could not determine EAV integration status. "
                        f"Mappings will default to '{self.default_value}'."
                    )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during ClassificationSystemMapper fit: {str(e)}")
            raise ValueError(
                f"Error during ClassificationSystemMapper fit: {str(e)}"
            ) from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by mapping classifications.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with the mapped classification column added.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "ClassificationSystemMapper must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If source column doesn't exist, create target column with default value
            if self.source_column not in result.columns:
                result[self.target_column] = self.default_value
                logger.warning(
                    f"Source column '{self.source_column}' not found. "
                    f"All mappings set to '{self.default_value}'."
                )
                return result

            # Apply mapping based on mapping type
            if self.mapping_type == "direct":
                result[self.target_column] = result[self.source_column].apply(
                    lambda code: self.mapping_dict.get(code, self.default_value)
                )
                logger.debug(
                    f"Applied direct mapping from '{self.source_column}' to '{self.target_column}' "
                    f"using {len(self.mapping_dict)} mappings."
                )

            elif self.mapping_type == "function" and self.mapping_function:
                # For function mapping, we need additional columns for context
                # This is a simplified implementation - in practice, you'd need to
                # extract the necessary context columns from the DataFrame
                result[self.target_column] = result.apply(
                    lambda row: self._apply_mapping_function(row), axis=1
                )
                logger.debug(
                    f"Applied function mapping from '{self.source_column}' to '{self.target_column}'."
                )

            elif self.mapping_type == "eav":
                # For EAV mapping, we would need to query an EAV database
                # This is a placeholder implementation
                result[self.target_column] = self.default_value
                logger.warning(
                    f"EAV mapping not fully implemented. "
                    f"All mappings set to '{self.default_value}'."
                )

            else:
                # Default to direct mapping with empty dictionary
                result[self.target_column] = self.default_value
                logger.warning(
                    f"Invalid mapping type '{self.mapping_type}' or missing mapping function. "
                    f"All mappings set to '{self.default_value}'."
                )

            return result
        except Exception as e:
            logger.error(f"Error during ClassificationSystemMapper transform: {str(e)}")
            raise ValueError(
                f"Error during ClassificationSystemMapper transform: {str(e)}"
            ) from e

    def _apply_mapping_function(self, row: pd.Series) -> str:
        """
        Apply the mapping function to a row of data.

        Args:
            row: Row of data containing the source column and context columns.

        Returns:
            Mapped classification code.
        """
        try:
            source_value = row[self.source_column]

            # Call the mapping function with the source value
            # In practice, you'd need to extract additional context from the row
            # based on the specific mapping function's requirements
            if self.mapping_function:
                return self.mapping_function(source_value)
            return self.default_value
        except Exception as e:
            logger.error(f"Error applying mapping function: {str(e)}")
            return self.default_value

    def _load_mappings_from_config(self):
        """
        Load classification system mappings from the configuration provider.

        This method loads the classification system mappings from the
        configuration based on the source and target columns.
        """
        try:
            # Get configuration
            config = self._config_provider.config

            # Try to load from masterformat_primary or masterformat_equipment
            if self.target_column.lower() == "masterformat":
                if config.masterformat_primary:
                    # Extract the system type from the source column
                    system_type = self.source_column.split("_")[-1]
                    if system_type in config.masterformat_primary.root:
                        self.mapping_dict = config.masterformat_primary.root[
                            system_type
                        ]
                        logger.debug(
                            f"Loaded {len(self.mapping_dict)} mappings from masterformat_primary "
                            f"for system type '{system_type}'."
                        )
                    else:
                        logger.warning(
                            f"No mappings found in masterformat_primary for system type '{system_type}'. "
                            f"Mappings will default to '{self.default_value}'."
                        )

                elif config.masterformat_equipment:
                    self.mapping_dict = config.masterformat_equipment.root
                    logger.debug(
                        f"Loaded {len(self.mapping_dict)} mappings from masterformat_equipment."
                    )

                else:
                    logger.warning(
                        f"No masterformat mappings found in configuration. "
                        f"Mappings will default to '{self.default_value}'."
                    )

            # For other classification systems, try to find a matching configuration
            else:
                # Look for a matching classification system in the configuration
                # Check if classification_targets exists in the configuration
                if hasattr(config.classification, "classification_targets"):
                    for target in config.classification.classification_targets:
                        if target.name == self.target_column:
                            logger.debug(
                                f"Found matching classification target '{target.name}'."
                            )
                            break
                    else:
                        logger.warning(
                            f"No matching classification target found in configuration for "
                            f"'{self.target_column}'. "
                            f"Mappings will default to '{self.default_value}'."
                        )
                else:
                    logger.warning(
                        f"No classification targets found in configuration. "
                        f"Mappings will default to '{self.default_value}'."
                    )

            if not self.mapping_dict:
                logger.warning(
                    f"No mappings loaded from configuration. "
                    f"Mappings will default to '{self.default_value}'."
                )
        except Exception as e:
            logger.error(
                f"Error loading configuration for ClassificationSystemMapper: {str(e)}"
            )
            self.mapping_dict = {}
