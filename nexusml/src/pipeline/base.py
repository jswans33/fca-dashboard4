"""
Pipeline Base Implementations Module

This module provides base implementations for the pipeline interfaces.
These base classes implement common functionality and provide default behavior
where appropriate, following the Template Method pattern.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from nexusml.src.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    PipelineComponent,
    Predictor,
)


class BasePipelineComponent(PipelineComponent):
    """
    Base implementation of the PipelineComponent interface.

    Provides common functionality for all pipeline components.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the component with a name and description.

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

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        This base implementation always returns True.
        Subclasses should override this method to provide specific validation.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True


class BaseDataLoader(BasePipelineComponent, DataLoader):
    """
    Base implementation of the DataLoader interface.

    Provides common functionality for data loading components.
    """

    def __init__(
        self,
        name: str = "BaseDataLoader",
        description: str = "Base data loader implementation",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the data loader.

        Args:
            name: Component name.
            description: Component description.
            config_path: Path to the configuration file. If None, uses default paths.
        """
        super().__init__(name, description)
        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from a YAML file.

        Returns:
            Configuration dictionary.
        """
        try:
            if self.config_path:
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f)

            # Try to load from standard locations
            config_paths = [
                Path(__file__).resolve().parent.parent.parent
                / "config"
                / "data_config.yml",
                Path(os.environ.get("NEXUSML_CONFIG", "")).parent / "data_config.yml",
            ]

            for path in config_paths:
                if path.exists():
                    with open(path, "r") as f:
                        return yaml.safe_load(f)

            # Return default configuration if no file is found
            return {
                "required_columns": [],
                "training_data": {
                    "default_path": "ingest/data/eq_ids.csv",
                    "encoding": "utf-8",
                    "fallback_encoding": "latin1",
                },
            }
        except Exception as e:
            print(f"Warning: Could not load data configuration: {e}")
            # Return a minimal default configuration
            return {
                "required_columns": [],
                "training_data": {
                    "default_path": "ingest/data/eq_ids.csv",
                    "encoding": "utf-8",
                    "fallback_encoding": "latin1",
                },
            }

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration.
        """
        return self._config

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.

        This base implementation loads data from a CSV file.
        Subclasses can override this method to support other data sources.

        Args:
            data_path: Path to the data file. If None, uses the default path from config.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        # Use default path if none provided
        if data_path is None:
            training_data_config = self._config.get("training_data", {})
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.csv"
            )
            data_path = str(
                Path(__file__).resolve().parent.parent.parent / default_path
            )

        # Read CSV file using pandas
        encoding = self._config.get("training_data", {}).get("encoding", "utf-8")
        fallback_encoding = self._config.get("training_data", {}).get(
            "fallback_encoding", "latin1"
        )

        try:
            df = pd.read_csv(data_path, encoding=encoding)
        except UnicodeDecodeError:
            # Try with a different encoding if the primary one fails
            print(
                f"Warning: Failed to read with {encoding} encoding. Trying {fallback_encoding}."
            )
            df = pd.read_csv(data_path, encoding=fallback_encoding)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {data_path}. Please provide a valid path."
            )

        return df


class BaseDataPreprocessor(BasePipelineComponent, DataPreprocessor):
    """
    Base implementation of the DataPreprocessor interface.

    Provides common functionality for data preprocessing components.
    """

    def __init__(
        self,
        name: str = "BaseDataPreprocessor",
        description: str = "Base data preprocessor implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data preprocessor.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data.

        This base implementation cleans column names and fills NaN values.
        Subclasses should override this method to provide specific preprocessing.

        Args:
            data: Input DataFrame to preprocess.
            **kwargs: Additional arguments for preprocessing.

        Returns:
            Preprocessed DataFrame.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df = data.copy()

        # Clean up column names (remove any leading/trailing whitespace)
        df.columns = [col.strip() for col in df.columns]

        # Fill NaN values with empty strings for text columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna("")

        # Verify and create required columns
        df = self.verify_required_columns(df)

        return df

    def verify_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that all required columns exist in the DataFrame and create them if they don't.

        Args:
            data: Input DataFrame to verify.

        Returns:
            DataFrame with all required columns.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df = data.copy()

        required_columns = self.config.get("required_columns", [])

        # Check each required column
        for column_info in required_columns:
            column_name = column_info["name"]
            default_value = column_info["default_value"]
            data_type = column_info["data_type"]

            # Check if the column exists
            if column_name not in df.columns:
                print(
                    f"Warning: Required column '{column_name}' not found. Creating with default value."
                )

                # Create the column with the default value
                if data_type == "str":
                    df[column_name] = default_value
                elif data_type == "float":
                    df[column_name] = float(default_value)
                elif data_type == "int":
                    df[column_name] = int(default_value)
                else:
                    # Default to string if type is unknown
                    df[column_name] = default_value

        return df


class BaseFeatureEngineer(BasePipelineComponent, FeatureEngineer):
    """
    Base implementation of the FeatureEngineer interface.

    Provides common functionality for feature engineering components.
    """

    def __init__(
        self,
        name: str = "BaseFeatureEngineer",
        description: str = "Base feature engineer implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the feature engineer.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}
        self._is_fitted = False

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        This method combines fit and transform in a single call.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def fit(self, data: pd.DataFrame, **kwargs) -> "BaseFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        This base implementation simply marks the engineer as fitted.
        Subclasses should override this method to provide specific fitting logic.

        Args:
            data: Input DataFrame to fit to.
            **kwargs: Additional arguments for fitting.

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        This base implementation returns the input data unchanged.
        Subclasses should override this method to provide specific transformation logic.

        Args:
            data: Input DataFrame to transform.
            **kwargs: Additional arguments for transformation.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If the feature engineer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                "Feature engineer must be fitted before transform can be called."
            )

        return data.copy()


class BaseModelBuilder(BasePipelineComponent, ModelBuilder):
    """
    Base implementation of the ModelBuilder interface.

    Provides common functionality for model building components.
    """

    def __init__(
        self,
        name: str = "BaseModelBuilder",
        description: str = "Base model builder implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model builder.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

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


class BaseModelTrainer(BasePipelineComponent, ModelTrainer):
    """
    Base implementation of the ModelTrainer interface.

    Provides common functionality for model training components.
    """

    def __init__(
        self,
        name: str = "BaseModelTrainer",
        description: str = "Base model trainer implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model trainer.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

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
        model.fit(x_train, y_train)
        return model

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
        cv = kwargs.get("cv", 5)
        scoring = kwargs.get("scoring", "accuracy")

        cv_results = cross_validate(
            model, x, y, cv=cv, scoring=scoring, return_train_score=True
        )

        return {
            "train_score": cv_results["train_score"].tolist(),
            "test_score": cv_results["test_score"].tolist(),
            "fit_time": cv_results["fit_time"].tolist(),
            "score_time": cv_results["score_time"].tolist(),
        }


class BaseModelEvaluator(BasePipelineComponent, ModelEvaluator):
    """
    Base implementation of the ModelEvaluator interface.

    Provides common functionality for model evaluation components.
    """

    def __init__(
        self,
        name: str = "BaseModelEvaluator",
        description: str = "Base model evaluator implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model evaluator.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

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
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        # Make predictions
        y_pred = model.predict(x_test)

        # Convert to DataFrame if it's not already
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

        # Calculate metrics for each target column
        metrics = {}
        for col in y_test.columns:
            # Get the column values using .loc to avoid Pylance errors
            y_test_col = y_test.loc[:, col]
            y_pred_col = y_pred.loc[:, col]

            col_metrics = {
                "accuracy": accuracy_score(y_test_col, y_pred_col),
                "f1_macro": f1_score(y_test_col, y_pred_col, average="macro"),
                "classification_report": classification_report(y_test_col, y_pred_col),
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

        return metrics

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
        analysis = {}

        # Analyze each target column
        for col in y_test.columns:
            # Calculate confusion metrics
            tp = ((y_test[col] == y_pred[col]) & (y_pred[col] != "Other")).sum()
            fp = ((y_test[col] != y_pred[col]) & (y_pred[col] != "Other")).sum()
            tn = ((y_test[col] == y_pred[col]) & (y_pred[col] == "Other")).sum()
            fn = ((y_test[col] != y_pred[col]) & (y_pred[col] == "Other")).sum()

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
            if "Other" in y_test[col].unique():
                other_indices = y_test[col] == "Other"
                other_accuracy = (
                    y_test[col][other_indices] == y_pred[col][other_indices]
                ).mean()

                # Calculate confusion metrics for "Other" category
                tp_other = ((y_test[col] == "Other") & (y_pred[col] == "Other")).sum()
                fp_other = ((y_test[col] != "Other") & (y_pred[col] == "Other")).sum()
                fn_other = ((y_test[col] == "Other") & (y_pred[col] != "Other")).sum()

                precision_other = (
                    tp_other / (tp_other + fp_other) if (tp_other + fp_other) > 0 else 0
                )
                recall_other = (
                    tp_other / (tp_other + fn_other) if (tp_other + fn_other) > 0 else 0
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

        return analysis


class BaseModelSerializer(BasePipelineComponent, ModelSerializer):
    """
    Base implementation of the ModelSerializer interface.

    Provides common functionality for model serialization components.
    """

    def __init__(
        self,
        name: str = "BaseModelSerializer",
        description: str = "Base model serializer implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model serializer.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

    def save_model(self, model: Pipeline, path: Union[str, Path], **kwargs) -> None:
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
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save the model using pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")

    def load_model(self, path: Union[str, Path], **kwargs) -> Pipeline:
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
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Check if the file exists
            if not path.exists():
                raise FileNotFoundError(f"Model file not found at {path}")

            # Load the model using pickle
            with open(path, "rb") as f:
                model = pickle.load(f)

            # Verify that the loaded object is a Pipeline
            if not isinstance(model, Pipeline):
                raise ValueError(f"Loaded object is not a Pipeline: {type(model)}")

            return model
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")


class BasePredictor(BasePipelineComponent, Predictor):
    """
    Base implementation of the Predictor interface.

    Provides common functionality for prediction components.
    """

    def __init__(
        self,
        name: str = "BasePredictor",
        description: str = "Base predictor implementation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description)
        self.config = config or {}

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make predictions using a trained model.

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
            # Try to get column names from the model
            try:
                column_names = model.classes_
            except AttributeError:
                # If that fails, use generic column names
                try:
                    # Try to safely access shape
                    if isinstance(predictions, np.ndarray):
                        if len(predictions.shape) > 1:
                            column_names = [
                                f"target_{i}" for i in range(predictions.shape[1])
                            ]
                        else:
                            column_names = ["target"]
                    else:
                        # For other types, use a safer approach
                        column_names = ["target"]
                except (AttributeError, TypeError):
                    column_names = ["target"]

            predictions = pd.DataFrame(predictions, columns=column_names)

        return predictions

    def predict_proba(
        self, model: Pipeline, data: pd.DataFrame, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Make probability predictions using a trained model.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            Dictionary mapping target columns to DataFrames of class probabilities.

        Raises:
            ValueError: If the model does not support probability predictions.
        """
        try:
            # Check if the model supports predict_proba
            if not hasattr(model, "predict_proba"):
                raise ValueError("Model does not support probability predictions")

            # Make probability predictions
            probas = model.predict_proba(data)

            # Convert to dictionary of DataFrames
            result = {}

            # Handle different model types
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

                    result[target_name] = pd.DataFrame(proba, columns=class_names)
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

                result["target"] = pd.DataFrame(probas, columns=class_names)

            return result
        except Exception as e:
            raise ValueError(f"Failed to make probability predictions: {e}")
