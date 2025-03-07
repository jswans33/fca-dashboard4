"""
Standard Feature Engineer Component

This module provides a standard implementation of the FeatureEngineer interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseFeatureEngineer
from nexusml.core.pipeline.components.transformers import (
    ClassificationSystemMapper,
    ColumnMapper,
    HierarchyBuilder,
    KeywordClassificationMapper,
    NumericCleaner,
    TextCombiner,
)

# Set up logging
logger = logging.getLogger(__name__)


class StandardFeatureEngineer(BaseFeatureEngineer):
    """
    Standard implementation of the FeatureEngineer interface.

    This class engineers features based on configuration provided by the
    ConfigurationProvider. It uses a pipeline of transformers to process
    the data and provides detailed logging.
    """

    def __init__(
        self,
        name: str = "StandardFeatureEngineer",
        description: str = "Standard feature engineer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the StandardFeatureEngineer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = config_provider or ConfigurationProvider()
        # Update the config from the provider
        self.config = self._config_provider.config.feature_engineering.model_dump()
        self._pipeline = None
        logger.info(f"Initialized {name}")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        This method combines fit and transform in a single call.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.

        Raises:
            ValueError: If features cannot be engineered.
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def fit(self, data: pd.DataFrame, **kwargs) -> "StandardFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        This method builds and fits a pipeline of transformers based on
        the configuration.

        Args:
            data: Input DataFrame to fit to.
            **kwargs: Additional arguments for fitting.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the feature engineer cannot be fit to the data.
        """
        try:
            logger.info(f"Fitting feature engineer to data with shape: {data.shape}")

            # Build the pipeline of transformers
            self._pipeline = self._build_pipeline()

            # Fit the pipeline to the data
            self._pipeline.fit(data)

            logger.info("Feature engineer fitted successfully")
            return self
        except Exception as e:
            logger.error(f"Error fitting feature engineer: {str(e)}")
            raise ValueError(f"Error fitting feature engineer: {str(e)}") from e

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform.
            **kwargs: Additional arguments for transformation.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If the data cannot be transformed.
        """
        if not self._pipeline:
            raise ValueError(
                "Feature engineer must be fitted before transform can be called."
            )

        try:
            logger.info(f"Transforming data with shape: {data.shape}")

            # Transform the data using the pipeline
            result = self._pipeline.transform(data)

            # Ensure the result is a pandas DataFrame
            if not isinstance(result, pd.DataFrame):
                # If the result is a numpy array, convert it to a DataFrame
                # Try to preserve column names if possible
                if isinstance(result, np.ndarray):
                    if result.ndim == 2 and result.shape[1] == len(data.columns):
                        # If the array has the same number of columns as the input data,
                        # use the input column names
                        result = pd.DataFrame(
                            result, columns=data.columns, index=data.index
                        )
                    else:
                        # Otherwise, use generic column names
                        result = pd.DataFrame(result, index=data.index)
                else:
                    # For other types, convert to DataFrame with default settings
                    result = pd.DataFrame(result)

                logger.debug(
                    f"Converted pipeline output to DataFrame with shape: {result.shape}"
                )

            logger.info(f"Data transformed successfully. Output shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise ValueError(f"Error transforming data: {str(e)}") from e

    def _build_pipeline(self) -> Pipeline:
        """
        Build a pipeline of transformers based on configuration.

        Returns:
            Configured pipeline of transformers.
        """
        try:
            logger.debug("Building feature engineering pipeline")

            # Create a list of transformer steps
            steps = []

            # Add TextCombiner for text combinations
            if "text_combinations" in self.config and self.config["text_combinations"]:
                for combo in self.config["text_combinations"]:
                    name = f"text_combiner_{combo['name']}"
                    transformer = TextCombiner(
                        name=combo["name"],
                        columns=combo["columns"],
                        separator=combo["separator"],
                        config_provider=self._config_provider,
                    )
                    steps.append((name, transformer))
                    logger.debug(f"Added {name} to pipeline")

            # Add NumericCleaner for numeric columns
            if "numeric_columns" in self.config and self.config["numeric_columns"]:
                transformer = NumericCleaner(
                    columns=self.config["numeric_columns"],
                    config_provider=self._config_provider,
                )
                steps.append(("numeric_cleaner", transformer))
                logger.debug("Added numeric_cleaner to pipeline")

            # Add HierarchyBuilder for hierarchies
            if "hierarchies" in self.config and self.config["hierarchies"]:
                transformer = HierarchyBuilder(
                    hierarchies=self.config["hierarchies"],
                    config_provider=self._config_provider,
                )
                steps.append(("hierarchy_builder", transformer))
                logger.debug("Added hierarchy_builder to pipeline")

            # Add ColumnMapper for column mappings
            if "column_mappings" in self.config and self.config["column_mappings"]:
                transformer = ColumnMapper(
                    mappings=self.config["column_mappings"],
                    config_provider=self._config_provider,
                )
                steps.append(("column_mapper", transformer))
                logger.debug("Added column_mapper to pipeline")

            # Add ClassificationSystemMapper for each classification system
            if (
                "classification_systems" in self.config
                and self.config["classification_systems"]
            ):
                for i, system in enumerate(self.config["classification_systems"]):
                    name = f"classification_mapper_{i}"
                    transformer = ClassificationSystemMapper(
                        source_column=system["source_column"],
                        target_column=system["target_column"],
                        mapping_type=system["mapping_type"],
                        config_provider=self._config_provider,
                    )
                    steps.append((name, transformer))
                    logger.debug(f"Added {name} to pipeline")

            # Create the pipeline
            pipeline = Pipeline(steps=steps)
            logger.debug(f"Built pipeline with {len(steps)} steps")

            return pipeline
        except Exception as e:
            logger.error(f"Error building feature engineering pipeline: {str(e)}")
            raise ValueError(
                f"Error building feature engineering pipeline: {str(e)}"
            ) from e
