"""
Prediction Stage Module

This module provides implementations of the PredictionStage interface for
making predictions using trained models.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.config.manager import ConfigurationManager
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import BasePredictionStage


class StandardPredictionStage(BasePredictionStage):
    """
    Implementation of PredictionStage for standard predictions.
    """

    def __init__(
        self,
        name: str = "StandardPrediction",
        description: str = "Makes standard predictions",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the standard prediction stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading prediction configuration.
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
            (context.has("trained_model") or context.has("model"))
            and (context.has("engineered_data") or context.has("data"))
        )

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make standard predictions.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing predictions.
        """
        # Make predictions
        predictions = model.predict(data)

        # Convert to DataFrame if it's not already
        if not isinstance(predictions, pd.DataFrame):
            # For MultiOutputClassifier, predictions will be a 2D array
            # with one column per target
            if hasattr(predictions, "shape") and len(predictions.shape) > 1:
                # Create column names for each target
                column_names = [f"prediction_{i}" for i in range(predictions.shape[1])]
                predictions_df = pd.DataFrame(predictions, columns=column_names)
            else:
                # Single output prediction
                predictions_df = pd.DataFrame(predictions, columns=["prediction"])
            
            return predictions_df

        return predictions


class ProbabilityPredictionStage(BasePredictionStage):
    """
    Implementation of PredictionStage for probability predictions.
    """

    def __init__(
        self,
        name: str = "ProbabilityPrediction",
        description: str = "Makes probability predictions",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the probability prediction stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading prediction configuration.
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
            (context.has("trained_model") or context.has("model"))
            and (context.has("engineered_data") or context.has("data"))
        )

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make probability predictions.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing probability predictions.

        Raises:
            ValueError: If the model does not support probability predictions.
        """
        # Check if the model supports predict_proba
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")

        # Make probability predictions
        probas = model.predict_proba(data)

        # Convert to DataFrame
        if isinstance(probas, list):
            # MultiOutputClassifier returns a list of arrays
            try:
                # Try to get target names from the model
                target_names = getattr(model, "classes_", None)
                if target_names is None:
                    # If that fails, use generic target names
                    target_names = [f"target_{i}" for i in range(len(probas))]
            except (AttributeError, TypeError):
                # If that fails, use generic target names
                target_names = [f"target_{i}" for i in range(len(probas))]

            # Create a DataFrame for each target
            result_dfs = []
            for i, proba in enumerate(probas):
                target_name = (
                    target_names[i] if i < len(target_names) else f"target_{i}"
                )

                try:
                    # Try to get class names from the model's estimators
                    estimators = getattr(model, "estimators_", None)
                    if estimators is not None and i < len(estimators):
                        class_names = getattr(estimators[i], "classes_", None)
                    else:
                        class_names = None

                    if class_names is None:
                        # If that fails, use generic class names
                        if hasattr(proba, "shape") and len(proba.shape) > 1:
                            class_names = [
                                f"class_{j}" for j in range(proba.shape[1])
                            ]
                        else:
                            class_names = ["class_0"]
                except (AttributeError, IndexError, TypeError):
                    # If that fails, use generic class names
                    if hasattr(proba, "shape") and len(proba.shape) > 1:
                        class_names = [f"class_{j}" for j in range(proba.shape[1])]
                    else:
                        class_names = ["class_0"]

                # Create column names with target and class
                columns = [f"{target_name}_{cls}" for cls in class_names]
                result_dfs.append(pd.DataFrame(proba, columns=columns))

            # Concatenate all DataFrames
            result = pd.concat(result_dfs, axis=1)
        else:
            # Single output classifier returns a single array
            try:
                # Try to get class names from the model
                class_names = getattr(model, "classes_", None)
                if class_names is None:
                    # If that fails, use generic class names
                    if hasattr(probas, "shape") and len(probas.shape) > 1:
                        class_names = [f"class_{j}" for j in range(probas.shape[1])]
                    else:
                        class_names = ["class_0"]
            except (AttributeError, TypeError):
                # If that fails, use generic class names
                if hasattr(probas, "shape") and len(probas.shape) > 1:
                    class_names = [f"class_{j}" for j in range(probas.shape[1])]
                else:
                    class_names = ["class_0"]

            # Create column names with class
            columns = [f"probability_{cls}" for cls in class_names]
            result = pd.DataFrame(probas, columns=columns)

        return result


class ThresholdPredictionStage(BasePredictionStage):
    """
    Implementation of PredictionStage for predictions with custom thresholds.
    """

    def __init__(
        self,
        name: str = "ThresholdPrediction",
        description: str = "Makes predictions with custom thresholds",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the threshold prediction stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading prediction configuration.
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
            (context.has("trained_model") or context.has("model"))
            and (context.has("engineered_data") or context.has("data"))
        )

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make predictions with custom thresholds.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing predictions.

        Raises:
            ValueError: If the model does not support probability predictions.
        """
        # Check if the model supports predict_proba
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support probability predictions")

        # Get thresholds from kwargs or config
        thresholds = kwargs.get("thresholds", self.config.get("thresholds", {}))

        # Make probability predictions
        probas = model.predict_proba(data)

        # Convert to DataFrame
        if isinstance(probas, list):
            # MultiOutputClassifier returns a list of arrays
            try:
                # Try to get target names from the model
                target_names = getattr(model, "classes_", None)
                if target_names is None:
                    # If that fails, use generic target names
                    target_names = [f"target_{i}" for i in range(len(probas))]
            except (AttributeError, TypeError):
                # If that fails, use generic target names
                target_names = [f"target_{i}" for i in range(len(probas))]

            # Create a DataFrame for each target
            result_dfs = []
            for i, proba in enumerate(probas):
                target_name = (
                    target_names[i] if i < len(target_names) else f"target_{i}"
                )

                try:
                    # Try to get class names from the model's estimators
                    estimators = getattr(model, "estimators_", None)
                    if estimators is not None and i < len(estimators):
                        class_names = getattr(estimators[i], "classes_", None)
                    else:
                        class_names = None

                    if class_names is None:
                        # If that fails, use generic class names
                        if hasattr(proba, "shape") and len(proba.shape) > 1:
                            class_names = [
                                f"class_{j}" for j in range(proba.shape[1])
                            ]
                        else:
                            class_names = ["class_0"]
                except (AttributeError, IndexError, TypeError):
                    # If that fails, use generic class names
                    if hasattr(proba, "shape") and len(proba.shape) > 1:
                        class_names = [f"class_{j}" for j in range(proba.shape[1])]
                    else:
                        class_names = ["class_0"]

                # Apply thresholds
                threshold = thresholds.get(target_name, 0.5)
                predictions = (proba >= threshold).astype(int)

                # Map predictions to class names
                prediction_df = pd.DataFrame(
                    {target_name: [class_names[p.argmax()] for p in predictions]}
                )
                result_dfs.append(prediction_df)

            # Concatenate all DataFrames
            result = pd.concat(result_dfs, axis=1)
        else:
            # Single output classifier returns a single array
            try:
                # Try to get class names from the model
                class_names = getattr(model, "classes_", None)
                if class_names is None:
                    # If that fails, use generic class names
                    if hasattr(probas, "shape") and len(probas.shape) > 1:
                        class_names = [f"class_{j}" for j in range(probas.shape[1])]
                    else:
                        class_names = ["class_0"]
            except (AttributeError, TypeError):
                # If that fails, use generic class names
                if hasattr(probas, "shape") and len(probas.shape) > 1:
                    class_names = [f"class_{j}" for j in range(probas.shape[1])]
                else:
                    class_names = ["class_0"]

            # Apply threshold
            threshold = thresholds.get("default", 0.5)
            predictions = (probas >= threshold).astype(int)

            # Map predictions to class names
            result = pd.DataFrame(
                {"prediction": [class_names[p.argmax()] for p in predictions]}
            )

        return result


class ConfigDrivenPredictionStage(BasePredictionStage):
    """
    Implementation of PredictionStage that uses configuration for predictions.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenPrediction",
        description: str = "Makes predictions based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the configuration-driven prediction stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading prediction configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self._predictors = {
            "standard": StandardPredictionStage(
                config=config, config_manager=config_manager
            ),
            "probability": ProbabilityPredictionStage(
                config=config, config_manager=config_manager
            ),
            "threshold": ThresholdPredictionStage(
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
            (context.has("trained_model") or context.has("model"))
            and (context.has("engineered_data") or context.has("data"))
        )

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make predictions based on configuration.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing predictions.
        """
        # Get the prediction type from kwargs or config
        prediction_type = kwargs.get(
            "prediction_type", self.config.get("prediction_type", "standard")
        )

        # Get the appropriate predictor
        if prediction_type not in self._predictors:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        predictor = self._predictors[prediction_type]

        # Make predictions
        return predictor.predict(model, data, **kwargs)