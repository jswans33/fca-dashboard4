"""
Model Building Stage Module

This module provides implementations of the ModelBuildingStage interface for
creating and configuring machine learning models.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nexusml.config.manager import ConfigurationManager
from nexusml.src.models.interfaces import ModelBuilder
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import BaseModelBuildingStage


class RandomForestModelBuildingStage(BaseModelBuildingStage):
    """
    Implementation of ModelBuildingStage for building Random Forest models.
    """

    def __init__(
        self,
        name: str = "RandomForestModelBuilding",
        description: str = "Builds a Random Forest model",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the Random Forest model building stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading model configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a Random Forest model.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.
        """
        # Get model parameters from kwargs or config
        n_estimators = kwargs.get("n_estimators", self.config.get("n_estimators", 100))
        max_depth = kwargs.get("max_depth", self.config.get("max_depth", None))
        min_samples_split = kwargs.get(
            "min_samples_split", self.config.get("min_samples_split", 2)
        )
        min_samples_leaf = kwargs.get(
            "min_samples_leaf", self.config.get("min_samples_leaf", 1)
        )
        random_state = kwargs.get("random_state", self.config.get("random_state", 42))

        # Create the base classifier
        base_classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        # Wrap in MultiOutputClassifier for multi-label classification
        classifier = MultiOutputClassifier(base_classifier)

        # Create the pipeline with text preprocessing
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create the pipeline with text preprocessing
        from sklearn.preprocessing import OneHotEncoder

        model = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            # Use combined_text for TF-IDF
                            (
                                "text",
                                TfidfVectorizer(max_features=1000),
                                "combined_text",
                            ),
                            # Use numeric columns for scaling
                            (
                                "numeric",
                                StandardScaler(),
                                [
                                    "service_life",
                                    "CategoryID",
                                    "OmniClassID",
                                    "UniFormatID",
                                    "MasterFormatID",
                                ],
                            ),
                            # Use categorical columns for one-hot encoding
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                ["equipment_tag", "manufacturer", "model"],
                            ),
                        ],
                        remainder="drop",  # Drop other columns
                    ),
                ),
                ("classifier", classifier),
            ]
        )

        return model


class GradientBoostingModelBuildingStage(BaseModelBuildingStage):
    """
    Implementation of ModelBuildingStage for building Gradient Boosting models.
    """

    def __init__(
        self,
        name: str = "GradientBoostingModelBuilding",
        description: str = "Builds a Gradient Boosting model",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the Gradient Boosting model building stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading model configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a Gradient Boosting model.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.
        """
        # Get model parameters from kwargs or config
        n_estimators = kwargs.get("n_estimators", self.config.get("n_estimators", 100))
        learning_rate = kwargs.get(
            "learning_rate", self.config.get("learning_rate", 0.1)
        )
        max_depth = kwargs.get("max_depth", self.config.get("max_depth", 3))
        min_samples_split = kwargs.get(
            "min_samples_split", self.config.get("min_samples_split", 2)
        )
        min_samples_leaf = kwargs.get(
            "min_samples_leaf", self.config.get("min_samples_leaf", 1)
        )
        random_state = kwargs.get("random_state", self.config.get("random_state", 42))

        # Create the base classifier
        base_classifier = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        # Wrap in MultiOutputClassifier for multi-label classification
        classifier = MultiOutputClassifier(base_classifier)

        # Create the pipeline with text preprocessing
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create the pipeline with text preprocessing
        from sklearn.preprocessing import OneHotEncoder

        model = Pipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer(
                        transformers=[
                            # Use combined_text for TF-IDF
                            (
                                "text",
                                TfidfVectorizer(max_features=1000),
                                "combined_text",
                            ),
                            # Use numeric columns for scaling
                            (
                                "numeric",
                                StandardScaler(),
                                [
                                    "service_life",
                                    "CategoryID",
                                    "OmniClassID",
                                    "UniFormatID",
                                    "MasterFormatID",
                                ],
                            ),
                            # Use categorical columns for one-hot encoding
                            (
                                "categorical",
                                OneHotEncoder(handle_unknown="ignore"),
                                ["equipment_tag", "manufacturer", "model"],
                            ),
                        ],
                        remainder="drop",  # Drop other columns
                    ),
                ),
                ("classifier", classifier),
            ]
        )

        return model


class ConfigDrivenModelBuildingStage(BaseModelBuildingStage):
    """
    Implementation of ModelBuildingStage that uses configuration for model building.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenModelBuilding",
        description: str = "Builds a model based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
        model_builder: Optional[ModelBuilder] = None,
    ):
        """
        Initialize the configuration-driven model building stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading model configuration.
            model_builder: Model builder to use. If None, uses the model type from config.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self.model_builder = model_builder
        self._builders = {
            "random_forest": RandomForestModelBuildingStage(
                config=config, config_manager=config_manager
            ),
            "gradient_boosting": GradientBoostingModelBuildingStage(
                config=config, config_manager=config_manager
            ),
        }

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a model based on configuration.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.
        """
        # If a model builder is provided, use it
        if self.model_builder is not None:
            return self.model_builder.build_model(**kwargs)

        # Get the model type from kwargs or config
        model_type = kwargs.get(
            "model_type", self.config.get("model_type", "random_forest")
        )

        # Get the appropriate builder
        if model_type not in self._builders:
            raise ValueError(f"Unsupported model type: {model_type}")

        builder = self._builders[model_type]

        # Build the model
        return builder.build_model(**kwargs)


class EnsembleModelBuildingStage(BaseModelBuildingStage):
    """
    Implementation of ModelBuildingStage for building ensemble models.
    """

    def __init__(
        self,
        name: str = "EnsembleModelBuilding",
        description: str = "Builds an ensemble of models",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
        model_builders: Optional[List[BaseModelBuildingStage]] = None,
    ):
        """
        Initialize the ensemble model building stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading model configuration.
            model_builders: List of model builders to use.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self.model_builders = model_builders or [
            RandomForestModelBuildingStage(
                config=config, config_manager=config_manager
            ),
            GradientBoostingModelBuildingStage(
                config=config, config_manager=config_manager
            ),
        ]

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build an ensemble of models.

        This implementation is a placeholder. In a real implementation, you would
        build multiple models and combine them using a voting or stacking approach.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.
        """
        # For simplicity, we'll just use the first model builder
        # In a real implementation, you would build multiple models and combine them
        return self.model_builders[0].build_model(**kwargs)
