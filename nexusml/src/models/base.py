"""
Model Building Base Implementations Module

This module provides base implementations for the model building interfaces.
These base classes implement common functionality and provide default behavior
where appropriate, following the Template Method pattern.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.model_building.interfaces import (
    ConfigurableModelBuilder,
    ConfigurableModelTrainer,
    HyperparameterOptimizer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
)

# Set up logging
logger = logging.getLogger(__name__)


class BaseModelBuilder(ModelBuilder):
    """
    Base implementation of the ModelBuilder interface.

    Provides common functionality for model building components.
    """

    def __init__(
        self,
        name: str = "BaseModelBuilder",
        description: str = "Base model builder implementation",
    ):
        """
        Initialize the model builder.

        Args:
            name: Component name.
            description: Component description.
        """
        self._name = name
        self._description = description

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific model building logic.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_model()")

    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        This base implementation returns the model unchanged.
        Subclasses should override this method to provide specific optimization logic.

        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for hyperparameter optimization.

        Returns:
            Optimized model pipeline.
        """
        return model

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.

        This base implementation returns an empty dictionary.
        Subclasses should override this method to provide specific default parameters.

        Returns:
            Dictionary of default parameters.
        """
        return {}

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get the parameter grid for hyperparameter optimization.

        This base implementation returns an empty dictionary.
        Subclasses should override this method to provide specific parameter grids.

        Returns:
            Dictionary mapping parameter names to lists of values to try.
        """
        return {}


@injectable
class BaseConfigurableModelBuilder(BaseModelBuilder, ConfigurableModelBuilder):
    """
    Base implementation of the ConfigurableModelBuilder interface.

    Provides common functionality for configurable model building components.
    """

    def __init__(
        self,
        name: str = "BaseConfigurableModelBuilder",
        description: str = "Base configurable model builder implementation",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the configurable model builder.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description)
        self._config_provider = config_provider or ConfigurationProvider()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the provider.

        Returns:
            Configuration dictionary.
        """
        try:
            # Try to get model configuration from the provider
            if hasattr(self._config_provider, "config") and hasattr(
                self._config_provider.config, "model"
            ):
                model_config = self._config_provider.config.model.model_dump()
                logger.info("Loaded model configuration from provider")
                return model_config
        except Exception as e:
            logger.warning(f"Could not load model configuration from provider: {e}")

        # Return default configuration if no configuration is available
        logger.info("Using default model configuration")
        return self.get_default_parameters()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the model builder.

        Returns:
            Dictionary containing the configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the model builder.

        Args:
            config: Configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
        self.config = config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model builder configuration.

        This base implementation always returns True.
        Subclasses should override this method to provide specific validation.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True


class BaseModelTrainer(ModelTrainer):
    """
    Base implementation of the ModelTrainer interface.

    Provides common functionality for model training components.
    """

    def __init__(
        self,
        name: str = "BaseModelTrainer",
        description: str = "Base model trainer implementation",
    ):
        """
        Initialize the model trainer.

        Args:
            name: Component name.
            description: Component description.
        """
        self._name = name
        self._description = description

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

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
        """
        try:
            logger.info(f"Training model {self._name}")
            model.fit(x_train, y_train)
            logger.info(f"Model {self._name} trained successfully")
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
        """
        try:
            logger.info(f"Performing cross-validation for model {self._name}")
            cv = kwargs.get("cv", 5)
            scoring = kwargs.get("scoring", "accuracy")

            # Check if we're dealing with multiclass-multioutput classification
            is_multioutput = isinstance(y, pd.DataFrame) and y.shape[1] > 1

            if is_multioutput:
                # Use our custom cross-validation function for multiclass-multioutput
                from nexusml.core.model_training.scoring import (
                    cross_validate_multioutput,
                )

                logger.info(
                    "Using custom cross-validation for multiclass-multioutput classification"
                )
                # Remove cv and scoring from kwargs to avoid duplicate parameters
                kwargs_copy = kwargs.copy()
                if "cv" in kwargs_copy:
                    del kwargs_copy["cv"]
                if "scoring" in kwargs_copy:
                    del kwargs_copy["scoring"]

                cv_results = cross_validate_multioutput(
                    model, x, y, cv=cv, scoring=scoring, **kwargs_copy
                )
            else:
                # Use scikit-learn's cross_validate for single-output classification
                cv_results = cross_validate(
                    model, x, y, cv=cv, scoring=scoring, return_train_score=True
                )

                # Convert numpy arrays to lists for better serialization
                cv_results = {
                    "train_score": cv_results["train_score"].tolist(),
                    "test_score": cv_results["test_score"].tolist(),
                    "fit_time": cv_results["fit_time"].tolist(),
                    "score_time": cv_results["score_time"].tolist(),
                }

            logger.info(f"Cross-validation completed for model {self._name}")
            return cv_results
        except Exception as e:
            logger.error(f"Error performing cross-validation: {str(e)}")
            # Return dummy results if cross-validation fails
            logger.warning("Returning dummy cross-validation results due to error")
            # Use default cv value if it's not defined in the exception handler
            cv_value = cv if "cv" in locals() else 5
            return {
                "train_score": [0.0] * cv_value,
                "test_score": [0.0] * cv_value,
                "fit_time": [0.0] * cv_value,
                "score_time": [0.0] * cv_value,
            }


@injectable
class BaseConfigurableModelTrainer(BaseModelTrainer, ConfigurableModelTrainer):
    """
    Base implementation of the ConfigurableModelTrainer interface.

    Provides common functionality for configurable model training components.
    """

    def __init__(
        self,
        name: str = "BaseConfigurableModelTrainer",
        description: str = "Base configurable model trainer implementation",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the configurable model trainer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description)
        self._config_provider = config_provider or ConfigurationProvider()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the provider.

        Returns:
            Configuration dictionary.
        """
        try:
            # Try to get training configuration from the provider
            if hasattr(self._config_provider, "config") and hasattr(
                self._config_provider.config, "training"
            ):
                training_config = self._config_provider.config.training.model_dump()
                logger.info("Loaded training configuration from provider")
                return training_config
        except Exception as e:
            logger.warning(f"Could not load training configuration from provider: {e}")

        # Return default configuration if no configuration is available
        logger.info("Using default training configuration")
        return {
            "cv": 5,
            "scoring": "accuracy",
            "random_state": 42,
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the model trainer.

        Returns:
            Dictionary containing the configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the model trainer.

        Args:
            config: Configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
        self.config = config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model trainer configuration.

        This base implementation always returns True.
        Subclasses should override this method to provide specific validation.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True

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
        """
        # Update kwargs with configuration
        for key, value in self.config.items():
            if key not in kwargs:
                kwargs[key] = value

        return super().train(model, x_train, y_train, **kwargs)

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
        """
        # Update kwargs with configuration
        for key, value in self.config.items():
            if key not in kwargs:
                kwargs[key] = value

        return super().cross_validate(model, x, y, **kwargs)


@injectable
class BaseHyperparameterOptimizer(HyperparameterOptimizer):
    """
    Base implementation of the HyperparameterOptimizer interface.

    Provides common functionality for hyperparameter optimization components.
    """

    def __init__(
        self,
        name: str = "BaseHyperparameterOptimizer",
        description: str = "Base hyperparameter optimizer implementation",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the hyperparameter optimizer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self._name = name
        self._description = description
        self._config_provider = config_provider or ConfigurationProvider()
        self.config = self._load_config()
        self._best_params = {}
        self._best_score = 0.0
        self._is_optimized = False

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the provider.

        Returns:
            Configuration dictionary.
        """
        try:
            # Try to get hyperparameter optimization configuration from the provider
            if hasattr(self._config_provider, "config") and hasattr(
                self._config_provider.config, "hyperparameter_optimization"
            ):
                hp_config = (
                    self._config_provider.config.hyperparameter_optimization.model_dump()
                )
                logger.info(
                    "Loaded hyperparameter optimization configuration from provider"
                )
                return hp_config
        except Exception as e:
            logger.warning(
                f"Could not load hyperparameter optimization configuration from provider: {e}"
            )

        # Return default configuration if no configuration is available
        logger.info("Using default hyperparameter optimization configuration")
        return {
            "cv": 3,
            "scoring": "f1_macro",
            "verbose": 1,
        }

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def optimize(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for optimization.

        Returns:
            Optimized model pipeline.

        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        try:
            logger.info(f"Optimizing hyperparameters for model {self._name}")

            # Get hyperparameter optimization settings
            param_grid = kwargs.get("param_grid", {})
            cv = kwargs.get("cv", self.config.get("cv", 3))
            scoring = kwargs.get("scoring", self.config.get("scoring", "f1_macro"))
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))

            if not param_grid:
                logger.warning(
                    "No parameter grid provided for hyperparameter optimization"
                )
                self._is_optimized = False
                return model

            # Use GridSearchCV for hyperparameter optimization
            grid_search = GridSearchCV(
                model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose
            )

            # Fit the grid search to the data
            logger.info(f"Fitting GridSearchCV with {len(param_grid)} parameters")
            grid_search.fit(x_train, y_train)

            # Store the best parameters and score
            self._best_params = grid_search.best_params_
            self._best_score = grid_search.best_score_
            self._is_optimized = True

            logger.info(f"Best parameters: {self._best_params}")
            logger.info(f"Best cross-validation score: {self._best_score}")

            return grid_search.best_estimator_

        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise ValueError(f"Error optimizing hyperparameters: {str(e)}") from e

    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.

        Returns:
            Dictionary of best parameters.

        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_params

    def get_best_score(self) -> float:
        """
        Get the best score achieved during optimization.

        Returns:
            Best score.

        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_score


@injectable
class BaseModelEvaluator(ModelEvaluator):
    """
    Base implementation of the ModelEvaluator interface.

    Provides common functionality for model evaluation components.
    """

    def __init__(
        self,
        name: str = "BaseModelEvaluator",
        description: str = "Base model evaluator implementation",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the model evaluator.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self._name = name
        self._description = description
        self._config_provider = config_provider or ConfigurationProvider()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the provider.

        Returns:
            Configuration dictionary.
        """
        try:
            # Try to get evaluation configuration from the provider
            if hasattr(self._config_provider, "config") and hasattr(
                self._config_provider.config, "evaluation"
            ):
                eval_config = self._config_provider.config.evaluation.model_dump()
                logger.info("Loaded evaluation configuration from provider")
                return eval_config
        except Exception as e:
            logger.warning(
                f"Could not load evaluation configuration from provider: {e}"
            )

        # Return default configuration if no configuration is available
        logger.info("Using default evaluation configuration")
        return {}

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def evaluate(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        try:
            logger.info(f"Evaluating model {self._name}")
            from sklearn.metrics import accuracy_score, classification_report, f1_score

            # Make predictions
            y_pred = model.predict(x_test)

            # Convert to DataFrame if it's not already
            if not isinstance(y_pred, pd.DataFrame):
                y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

            # Calculate metrics for each target column
            metrics = {}
            for col in y_test.columns:
                # Get the column values
                y_test_col = y_test[col]
                y_pred_col = y_pred[col]

                col_metrics = {
                    "accuracy": accuracy_score(y_test_col, y_pred_col),
                    "f1_macro": f1_score(y_test_col, y_pred_col, average="macro"),
                    "classification_report": classification_report(
                        y_test_col, y_pred_col
                    ),
                }
                metrics[col] = col_metrics

            # Add overall metrics
            metrics["overall"] = {
                "accuracy_mean": np.mean(
                    [metrics[col]["accuracy"] for col in y_test.columns]
                ),
                "f1_macro_mean": np.mean(
                    [metrics[col]["f1_macro"] for col in y_test.columns]
                ),
            }

            logger.info(f"Model {self._name} evaluation completed")
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise ValueError(f"Error evaluating model: {str(e)}") from e

    def analyze_predictions(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze model predictions in detail.

        Args:
            model: Trained model pipeline.
            x_test: Test features.
            y_test: Test targets.
            y_pred: Model predictions.
            **kwargs: Additional arguments for analysis.

        Returns:
            Dictionary of analysis results.
        """
        try:
            logger.info(f"Analyzing predictions for model {self._name}")
            analysis = {}

            # Analyze each target column
            for col in y_test.columns:
                # Make sure the indices match
                y_test_col = y_test[col].reset_index(drop=True)
                y_pred_col = y_pred[col].reset_index(drop=True)

                # Calculate confusion metrics
                tp = ((y_test_col == y_pred_col) & (y_pred_col != "Other")).sum()
                fp = ((y_test_col != y_pred_col) & (y_pred_col != "Other")).sum()
                tn = ((y_test_col == y_pred_col) & (y_pred_col == "Other")).sum()
                fn = ((y_test_col != y_pred_col) & (y_pred_col == "Other")).sum()

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                analysis[col] = {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                }

                # Analyze "Other" category if present
                if "Other" in y_test_col.unique():
                    other_indices = y_test_col == "Other"
                    other_accuracy = (
                        y_test_col[other_indices] == y_pred_col[other_indices]
                    ).mean()

                    # Calculate confusion metrics for "Other" category
                    tp_other = ((y_test_col == "Other") & (y_pred_col == "Other")).sum()
                    fp_other = ((y_test_col != "Other") & (y_pred_col == "Other")).sum()
                    fn_other = ((y_test_col == "Other") & (y_pred_col != "Other")).sum()

                    precision_other = (
                        tp_other / (tp_other + fp_other)
                        if (tp_other + fp_other) > 0
                        else 0
                    )
                    recall_other = (
                        tp_other / (tp_other + fn_other)
                        if (tp_other + fn_other) > 0
                        else 0
                    )
                    f1_other = (
                        2
                        * precision_other
                        * recall_other
                        / (precision_other + recall_other)
                        if (precision_other + recall_other) > 0
                        else 0
                    )

                    analysis[col]["other_category"] = {
                        "accuracy": float(other_accuracy),
                        "true_positives": int(tp_other),
                        "false_positives": int(fp_other),
                        "false_negatives": int(fn_other),
                        "precision": float(precision_other),
                        "recall": float(recall_other),
                        "f1_score": float(f1_other),
                    }

            logger.info(f"Prediction analysis completed for model {self._name}")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            raise ValueError(f"Error analyzing predictions: {str(e)}") from e


@injectable
class BaseModelSerializer(ModelSerializer):
    """
    Base implementation of the ModelSerializer interface.

    Provides common functionality for model serialization components.
    """

    def __init__(
        self,
        name: str = "BaseModelSerializer",
        description: str = "Base model serializer implementation",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the model serializer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self._name = name
        self._description = description
        self._config_provider = config_provider or ConfigurationProvider()
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the provider.

        Returns:
            Configuration dictionary.
        """
        try:
            # Try to get serialization configuration from the provider
            if hasattr(self._config_provider, "config") and hasattr(
                self._config_provider.config, "serialization"
            ):
                serial_config = self._config_provider.config.serialization.model_dump()
                logger.info("Loaded serialization configuration from provider")
                return serial_config
        except Exception as e:
            logger.warning(
                f"Could not load serialization configuration from provider: {e}"
            )

        # Return default configuration if no configuration is available
        logger.info("Using default serialization configuration")
        return {
            "model_dir": "models",
        }

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def save_model(self, model: Pipeline, path: str, **kwargs) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            **kwargs: Additional arguments for saving.

        Raises:
            IOError: If the model cannot be saved.
        """
        try:
            logger.info(f"Saving model {self._name} to {path}")
            import pickle

            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the model using pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)

            logger.info(f"Model {self._name} saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise IOError(f"Error saving model: {str(e)}") from e

    def load_model(self, path: str, **kwargs) -> Pipeline:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model.
            **kwargs: Additional arguments for loading.

        Returns:
            Loaded model pipeline.

        Raises:
            IOError: If the model cannot be loaded.
            ValueError: If the loaded file is not a valid model.
        """
        try:
            logger.info(f"Loading model from {path}")
            import pickle

            # Check if the file exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")

            # Load the model using pickle
            with open(path, "rb") as f:
                model = pickle.load(f)

            # Verify that the loaded object is a Pipeline
            if not isinstance(model, Pipeline):
                raise ValueError(f"Loaded object is not a Pipeline: {type(model)}")

            logger.info(f"Model loaded successfully from {path}")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise IOError(f"Error loading model: {str(e)}") from e
