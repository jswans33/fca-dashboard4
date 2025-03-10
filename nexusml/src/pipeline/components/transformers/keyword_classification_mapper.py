"""
Keyword Classification Mapper Transformer

This module provides a transformer for mapping keywords to classifications.
It follows the scikit-learn transformer interface and uses the configuration system.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.core.config.provider import ConfigurationProvider

# Set up logging
logger = logging.getLogger(__name__)


class KeywordClassificationMapper(BaseEstimator, TransformerMixin):
    """
    Transformer for mapping keywords to classifications.

    This transformer maps text columns to classification columns based on keyword patterns.
    It can be used for categorizing equipment based on descriptions or other text fields.
    """

    def __init__(
        self,
        source_column: str = "combined_features",
        target_column: str = "classification",
        keyword_mappings: Optional[Dict[str, List[str]]] = None,
        case_sensitive: bool = False,
        default_value: str = "Other",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the KeywordClassificationMapper transformer.

        Args:
            source_column: Source column containing text to search for keywords.
            target_column: Target column to store the classification.
            keyword_mappings: Dictionary mapping classifications to lists of keywords.
            case_sensitive: Whether keyword matching should be case-sensitive.
            default_value: Default classification value if no keywords match.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self.source_column = source_column
        self.target_column = target_column
        self.keyword_mappings = keyword_mappings or {}
        self.case_sensitive = case_sensitive
        self.default_value = default_value
        self._config_provider = config_provider or ConfigurationProvider()
        self._is_fitted = False
        self._compiled_patterns = {}
        logger.debug(
            f"Initialized KeywordClassificationMapper for {source_column} -> {target_column}"
        )

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer to the data.

        This method validates the source column and compiles regex patterns
        for keyword matching.

        Args:
            X: Input DataFrame.
            y: Ignored (included for scikit-learn compatibility).

        Returns:
            Self for method chaining.
        """
        try:
            logger.debug(
                f"Fitting KeywordClassificationMapper on DataFrame with shape: {X.shape}"
            )

            # Check if source column exists
            if self.source_column not in X.columns:
                logger.warning(
                    f"Source column '{self.source_column}' not found in input data. "
                    f"Classification will default to '{self.default_value}'."
                )

            # If keyword mappings not explicitly provided, get from configuration
            if not self.keyword_mappings:
                self._load_mappings_from_config()

            # Compile regex patterns for each classification
            self._compiled_patterns = {}
            for classification, keywords in self.keyword_mappings.items():
                patterns = []
                for keyword in keywords:
                    try:
                        # Escape special regex characters in the keyword
                        escaped_keyword = re.escape(keyword)
                        # Compile the pattern with word boundaries
                        pattern = re.compile(
                            r"\b" + escaped_keyword + r"\b",
                            flags=0 if self.case_sensitive else re.IGNORECASE,
                        )
                        patterns.append(pattern)
                    except re.error as e:
                        logger.warning(
                            f"Invalid regex pattern for keyword '{keyword}': {str(e)}. "
                            f"This keyword will be skipped."
                        )

                if patterns:
                    self._compiled_patterns[classification] = patterns
                    logger.debug(
                        f"Compiled {len(patterns)} patterns for classification '{classification}'."
                    )
                else:
                    logger.warning(
                        f"No valid patterns for classification '{classification}'. "
                        f"This classification will never be assigned."
                    )

            if not self._compiled_patterns:
                logger.warning(
                    f"No valid keyword patterns found. "
                    f"All classifications will default to '{self.default_value}'."
                )
            else:
                logger.debug(
                    f"Compiled patterns for {len(self._compiled_patterns)} classifications."
                )

            self._is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error during KeywordClassificationMapper fit: {str(e)}")
            raise ValueError(
                f"Error during KeywordClassificationMapper fit: {str(e)}"
            ) from e

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by mapping keywords to classifications.

        Args:
            X: Input DataFrame.

        Returns:
            DataFrame with the classification column added.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "KeywordClassificationMapper must be fitted before transform can be called."
            )

        try:
            logger.debug(f"Transforming DataFrame with shape: {X.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = X.copy()

            # If source column doesn't exist, create classification column with default value
            if self.source_column not in result.columns:
                result[self.target_column] = self.default_value
                logger.warning(
                    f"Source column '{self.source_column}' not found. "
                    f"All classifications set to '{self.default_value}'."
                )
                return result

            # If no patterns, create classification column with default value
            if not self._compiled_patterns:
                result[self.target_column] = self.default_value
                logger.warning(
                    f"No valid keyword patterns. "
                    f"All classifications set to '{self.default_value}'."
                )
                return result

            # Apply classification based on keyword patterns
            result[self.target_column] = result[self.source_column].apply(
                lambda text: self._classify_text(text)
            )

            logger.debug(
                f"Created classification column '{self.target_column}' "
                f"based on keywords in '{self.source_column}'."
            )
            return result
        except Exception as e:
            logger.error(
                f"Error during KeywordClassificationMapper transform: {str(e)}"
            )
            raise ValueError(
                f"Error during KeywordClassificationMapper transform: {str(e)}"
            ) from e

    def _classify_text(self, text: str) -> str:
        """
        Classify text based on keyword patterns.

        Args:
            text: Text to classify.

        Returns:
            Classification based on keyword matches, or default value if no match.
        """
        if not isinstance(text, str):
            return self.default_value

        # Check each classification's patterns
        for classification, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    return classification

        # No match found, return default value
        return self.default_value

    def _load_mappings_from_config(self):
        """
        Load keyword mappings from the configuration provider.

        This method loads the keyword mappings from the
        classification section of the configuration.
        """
        try:
            # Get classification configuration
            config = self._config_provider.config
            classification_config = config.classification

            # Get input field mappings
            input_mappings = classification_config.input_field_mappings

            # Convert input field mappings to keyword mappings
            self.keyword_mappings = {}
            for mapping in input_mappings:
                target = mapping.target
                patterns = mapping.patterns
                if target and patterns:
                    self.keyword_mappings[target] = patterns

            logger.debug(
                f"Loaded configuration for {len(self.keyword_mappings)} keyword mappings."
            )

            if not self.keyword_mappings:
                logger.warning(
                    "No keyword mappings found in configuration. "
                    "All classifications will default to the default value."
                )
        except Exception as e:
            logger.error(
                f"Error loading configuration for KeywordClassificationMapper: {str(e)}"
            )
            self.keyword_mappings = {}
