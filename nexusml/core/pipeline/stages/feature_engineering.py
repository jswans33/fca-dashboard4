"""
Feature Engineering Stage Module

This module provides implementations of the FeatureEngineeringStage interface for
transforming raw data into features suitable for model training.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from nexusml.config.manager import ConfigurationManager
from nexusml.core.feature_engineering.config_driven import ConfigDrivenFeatureEngineer
from nexusml.core.feature_engineering.interfaces import FeatureEngineer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import BaseFeatureEngineeringStage


class ConfigDrivenFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage that uses configuration for transformations.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenFeatureEngineering",
        description: str = "Engineers features based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ):
        """
        Initialize the configuration-driven feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading feature engineering configuration.
            feature_engineer: Feature engineer to use. If None, creates a ConfigDrivenFeatureEngineer.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self.feature_engineer = feature_engineer or ConfigDrivenFeatureEngineer(
            self.config_manager
        )

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data using configuration.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.
        """
        # Get the configuration name from kwargs or config
        config_name = kwargs.get(
            "config_name", self.config.get("config_name", "feature_config")
        )

        # Engineer features
        return self.feature_engineer.transform(data, config_name)


class TextFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage for text feature engineering.
    """

    def __init__(
        self,
        name: str = "TextFeatureEngineering",
        description: str = "Engineers text features",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the text feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description, config)

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer text features from the input data.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered text features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = data.copy()

        # Get text combinations from kwargs or config
        text_combinations = kwargs.get(
            "text_combinations", self.config.get("text_combinations", [])
        )

        # Apply text combinations
        for combo in text_combinations:
            name = combo.get("name")
            columns = combo.get("columns", [])
            separator = combo.get("separator", " ")

            # Check if all required columns exist
            available_columns = [col for col in columns if col in result.columns]

            if available_columns:
                # Combine available columns
                result[name] = result[available_columns].fillna("").agg(separator.join, axis=1)
            else:
                # Create empty column if no source columns are available
                result[name] = "Unknown"

        return result


class NumericFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage for numeric feature engineering.
    """

    def __init__(
        self,
        name: str = "NumericFeatureEngineering",
        description: str = "Engineers numeric features",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the numeric feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description, config)

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer numeric features from the input data.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered numeric features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = data.copy()

        # Get numeric transformations from kwargs or config
        numeric_configs = kwargs.get(
            "numeric_configs", self.config.get("numeric_configs", [])
        )

        # Apply numeric transformations
        for config in numeric_configs:
            name = config.get("name")
            new_name = config.get("new_name", name)
            fill_value = config.get("fill_value", 0)
            dtype = config.get("dtype", "float")

            if name in result.columns:
                # Copy and convert the column
                result[new_name] = result[name].fillna(fill_value)
                if dtype == "float":
                    result[new_name] = pd.to_numeric(
                        result[new_name], errors="coerce"
                    ).fillna(fill_value)
                elif dtype == "int":
                    result[new_name] = pd.to_numeric(
                        result[new_name], errors="coerce"
                    ).fillna(fill_value).astype(int)
            else:
                # Create column with default value
                result[new_name] = fill_value

        return result


class HierarchicalFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage for hierarchical feature engineering.
    """

    def __init__(
        self,
        name: str = "HierarchicalFeatureEngineering",
        description: str = "Engineers hierarchical features",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the hierarchical feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description, config)

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer hierarchical features from the input data.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered hierarchical features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = data.copy()

        # Get hierarchies from kwargs or config
        hierarchies = kwargs.get("hierarchies", self.config.get("hierarchies", []))

        # Apply hierarchical combinations
        for hierarchy in hierarchies:
            new_col = hierarchy.get("new_col")
            parents = hierarchy.get("parents", [])
            separator = hierarchy.get("separator", "-")

            # Check if all parent columns exist
            available_parents = [col for col in parents if col in result.columns]

            if available_parents:
                # Combine available parent columns
                result[new_col] = result[available_parents].fillna("").agg(
                    separator.join, axis=1
                )
            else:
                # Create empty column if no parent columns are available
                result[new_col] = "Unknown"

        return result


class CompositeFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage that combines multiple feature engineers.
    """

    def __init__(
        self,
        name: str = "CompositeFeatureEngineering",
        description: str = "Combines multiple feature engineers",
        config: Optional[Dict[str, Any]] = None,
        stages: Optional[List[BaseFeatureEngineeringStage]] = None,
    ):
        """
        Initialize the composite feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            stages: List of feature engineering stages to use.
        """
        super().__init__(name, description, config)
        self.stages = stages or []

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data using multiple feature engineers.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.
        """
        # Start with the original data
        result = data.copy()

        # Apply each feature engineering stage in sequence
        for stage in self.stages:
            result = stage.engineer_features(result, **kwargs)

        return result


class SimpleFeatureEngineeringStage(BaseFeatureEngineeringStage):
    """
    Implementation of FeatureEngineeringStage with simplified feature engineering.
    
    This stage is designed to be compatible with the existing SimpleFeatureEngineer
    used in train_model_pipeline_v2.py.
    """

    def __init__(
        self,
        name: str = "SimpleFeatureEngineering",
        description: str = "Performs simplified feature engineering",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the simple feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description, config)

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Perform simplified feature engineering.

        This implementation combines the description field with other text fields
        and adds a service_life column.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = data.copy()

        # Combine text fields into a single field
        text_fields = ["description", "category_name", "mcaa_system_category"]
        available_fields = [field for field in text_fields if field in result.columns]

        if available_fields:
            result["combined_text"] = result[available_fields].fillna("").agg(" ".join, axis=1)
        else:
            result["combined_text"] = "Unknown"

        # Add service_life column (default to 20 if not present)
        if "service_life" not in result.columns:
            result["service_life"] = 20

        return result