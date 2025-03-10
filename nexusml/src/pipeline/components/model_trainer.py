"""
Model Trainer Component

This module provides a standard implementation of the ModelTrainer interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseModelTrainer

# Set up logging
logger = logging.getLogger(__name__)


class StandardModelTrainer(BaseModelTrainer):
    """
    Standard implementation of the ModelTrainer interface.

    This class trains models based on configuration provided by the
    ConfigurationProvider. It supports cross-validation and provides
    detailed logging.
    """

    def __init__(
        self,
        name: str = "StandardModelTrainer",
        description: str = "Standard model trainer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the StandardModelTrainer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = config_provider or ConfigurationProvider()

        # Create a default training configuration
        self.config = {
            "training": {"random_state": 42, "test_size": 0.2, "stratify": True},
            "cross_validation": {
                "cv": 5,
                "scoring": ["accuracy", "f1_macro"],
                "return_train_score": True,
            },
        }

        # Try to update from configuration provider if available
        try:
            # Check if there's a classification section in the config
            if hasattr(self._config_provider.config, "classification"):
                classifier_config = (
                    self._config_provider.config.classification.model_dump()
                )
                if "training" in classifier_config:
                    self.config.update(classifier_config["training"])
                    logger.info(
                        "Updated training configuration from classification section"
                    )
            logger.debug(f"Using training configuration: {self.config}")
        except Exception as e:
            logger.warning(f"Could not load training configuration: {e}")
            logger.info("Using default training configuration")

        logger.info(f"Initialized {name}")

    def train(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model on the provided data.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.

        Raises:
            ValueError: If the model cannot be trained.
        """
        try:
            logger.info(f"Training model on data with shape: {x_train.shape}")

            # Extract any training parameters from kwargs
            verbose = kwargs.get("verbose", 1)
            sample_weight = kwargs.get("sample_weight", None)

            # Log training parameters
            logger.debug(f"Training with verbose={verbose}")
            if sample_weight is not None:
                logger.debug("Using sample weights for training")

            # Train the model
            start_time = pd.Timestamp.now()
            model.fit(x_train, y_train, **kwargs)
            end_time = pd.Timestamp.now()

            training_time = (end_time - start_time).total_seconds()
            logger.info(f"Model trained successfully in {training_time:.2f} seconds")

            return model

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ValueError(f"Error training model: {str(e)}") from e

    def cross_validate(
        self, model: Pipeline, x: pd.DataFrame, y: pd.DataFrame, **kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.

        Args:
            model: Model pipeline to validate.
            x: Feature data.
            y: Target data.
            **kwargs: Additional arguments for cross-validation.

        Returns:
            Dictionary of validation metrics.

        Raises:
            ValueError: If cross-validation cannot be performed.
        """
        try:
            logger.info(f"Performing cross-validation on data with shape: {x.shape}")

            # Get cross-validation settings
            cv_settings = self.config.get("cross_validation", {})
            cv = kwargs.get("cv", cv_settings.get("cv", 5))
            scoring = kwargs.get(
                "scoring", cv_settings.get("scoring", ["accuracy", "f1_macro"])
            )
            return_train_score = kwargs.get(
                "return_train_score", cv_settings.get("return_train_score", True)
            )

            # Log cross-validation parameters
            logger.debug(f"Cross-validation with cv={cv}, scoring={scoring}")

            # Perform cross-validation
            start_time = pd.Timestamp.now()
            cv_results = cross_validate(
                model,
                x,
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=return_train_score,
            )
            end_time = pd.Timestamp.now()

            cv_time = (end_time - start_time).total_seconds()
            logger.info(f"Cross-validation completed in {cv_time:.2f} seconds")

            # Convert numpy arrays to lists for better serialization
            results = {}
            for key, value in cv_results.items():
                if hasattr(value, "tolist"):
                    results[key] = value.tolist()
                else:
                    results[key] = value

            # Calculate and log average scores
            for key in results:
                if key.endswith("_score"):
                    avg_score = sum(results[key]) / len(results[key])
                    logger.info(f"Average {key}: {avg_score:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            raise ValueError(f"Error during cross-validation: {str(e)}") from e
