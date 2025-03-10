"""
Model Training Stage Module

This module provides implementations of the ModelTrainingStage interface for
training machine learning models on prepared data.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from nexusml.config.manager import ConfigurationManager
from nexusml.src.models.interfaces import ModelTrainer
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import BaseModelTrainingStage


class StandardModelTrainingStage(BaseModelTrainingStage):
    """
    Implementation of ModelTrainingStage for standard model training.
    """

    def __init__(
        self,
        name: str = "StandardModelTraining",
        description: str = "Trains a model using standard training",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the standard model training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading training configuration.
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
        return (
            context.has("model") and context.has("x_train") and context.has("y_train")
        )

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model using standard training.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        # Train the model
        model.fit(x_train, y_train)
        return model


class CrossValidationTrainingStage(BaseModelTrainingStage):
    """
    Implementation of ModelTrainingStage for cross-validation training.
    """

    def __init__(
        self,
        name: str = "CrossValidationTraining",
        description: str = "Trains a model using cross-validation",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the cross-validation training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading training configuration.
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
        return (
            context.has("model") and context.has("x_train") and context.has("y_train")
        )

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model using cross-validation.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        # Get cross-validation parameters from kwargs or config
        cv = kwargs.get("cv", self.config.get("cv", 5))
        scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))

        # Perform cross-validation
        cv_results = cross_validate(
            model, x_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )

        # Store cross-validation results in the model's metadata
        if not hasattr(model, "metadata"):
            model.metadata = {}
        model.metadata["cv_results"] = {
            "train_score": cv_results["train_score"].tolist(),
            "test_score": cv_results["test_score"].tolist(),
            "fit_time": cv_results["fit_time"].tolist(),
            "score_time": cv_results["score_time"].tolist(),
        }

        # Train the model on the full training set
        model.fit(x_train, y_train)
        return model


class GridSearchTrainingStage(BaseModelTrainingStage):
    """
    Implementation of ModelTrainingStage for grid search hyperparameter optimization.
    """

    def __init__(
        self,
        name: str = "GridSearchTraining",
        description: str = "Trains a model using grid search",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the grid search training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading training configuration.
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
        return (
            context.has("model") and context.has("x_train") and context.has("y_train")
        )

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model using grid search.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        # Get grid search parameters from kwargs or config
        param_grid = kwargs.get("param_grid", self.config.get("param_grid", {}))
        cv = kwargs.get("cv", self.config.get("cv", 5))
        scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))
        n_jobs = kwargs.get("n_jobs", self.config.get("n_jobs", -1))
        verbose = kwargs.get("verbose", self.config.get("verbose", 1))

        # Create the grid search
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        # Train the model
        grid_search.fit(x_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Store grid search results in the model's metadata
        if not hasattr(best_model, "metadata"):
            best_model.metadata = {}
        best_model.metadata["grid_search_results"] = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in grid_search.cv_results_.items()
            },
        }

        return best_model


class RandomizedSearchTrainingStage(BaseModelTrainingStage):
    """
    Implementation of ModelTrainingStage for randomized search hyperparameter optimization.
    """

    def __init__(
        self,
        name: str = "RandomizedSearchTraining",
        description: str = "Trains a model using randomized search",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the randomized search training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading training configuration.
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
        return (
            context.has("model") and context.has("x_train") and context.has("y_train")
        )

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model using randomized search.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        # Get randomized search parameters from kwargs or config
        param_distributions = kwargs.get(
            "param_distributions", self.config.get("param_distributions", {})
        )
        n_iter = kwargs.get("n_iter", self.config.get("n_iter", 10))
        cv = kwargs.get("cv", self.config.get("cv", 5))
        scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))
        n_jobs = kwargs.get("n_jobs", self.config.get("n_jobs", -1))
        verbose = kwargs.get("verbose", self.config.get("verbose", 1))
        random_state = kwargs.get("random_state", self.config.get("random_state", 42))

        # Create the randomized search
        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

        # Train the model
        random_search.fit(x_train, y_train)

        # Get the best model
        best_model = random_search.best_estimator_

        # Store randomized search results in the model's metadata
        if not hasattr(best_model, "metadata"):
            best_model.metadata = {}
        best_model.metadata["random_search_results"] = {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "cv_results": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in random_search.cv_results_.items()
            },
        }

        return best_model


class ConfigDrivenModelTrainingStage(BaseModelTrainingStage):
    """
    Implementation of ModelTrainingStage that uses configuration for model training.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenModelTraining",
        description: str = "Trains a model based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
        model_trainer: Optional[ModelTrainer] = None,
    ):
        """
        Initialize the configuration-driven model training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading training configuration.
            model_trainer: Model trainer to use. If None, uses the training type from config.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self.model_trainer = model_trainer
        self._trainers = {
            "standard": StandardModelTrainingStage(
                config=config, config_manager=config_manager
            ),
            "cross_validation": CrossValidationTrainingStage(
                config=config, config_manager=config_manager
            ),
            "grid_search": GridSearchTrainingStage(
                config=config, config_manager=config_manager
            ),
            "randomized_search": RandomizedSearchTrainingStage(
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
        return (
            context.has("model") and context.has("x_train") and context.has("y_train")
        )

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model based on configuration.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        # If a model trainer is provided, use it
        if self.model_trainer is not None:
            return self.model_trainer.train(model, x_train, y_train, **kwargs)

        # Get the training type from kwargs or config
        training_type = kwargs.get(
            "training_type", self.config.get("training_type", "standard")
        )

        # Get the appropriate trainer
        if training_type not in self._trainers:
            raise ValueError(f"Unsupported training type: {training_type}")

        trainer = self._trainers[training_type]

        # Train the model
        return trainer.train_model(model, x_train, y_train, **kwargs)
