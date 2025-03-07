"""
Pipeline Interfaces Module

This module defines the interfaces for all pipeline components in the NexusML suite.
Each interface follows the Interface Segregation Principle (ISP) from SOLID,
defining a minimal set of methods that components must implement.
"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class DataLoader(abc.ABC):
    """
    Interface for data loading components.

    Responsible for loading data from various sources and returning it in a standardized format.
    """

    @abc.abstractmethod
    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.

        Args:
            data_path: Path to the data file. If None, uses a default path.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        pass

    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration.
        """
        pass


class DataPreprocessor(abc.ABC):
    """
    Interface for data preprocessing components.

    Responsible for cleaning and preparing data for feature engineering.
    """

    @abc.abstractmethod
    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data.

        Args:
            data: Input DataFrame to preprocess.
            **kwargs: Additional arguments for preprocessing.

        Returns:
            Preprocessed DataFrame.

        Raises:
            ValueError: If the data cannot be preprocessed.
        """
        pass

    @abc.abstractmethod
    def verify_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that all required columns exist in the DataFrame and create them if they don't.

        Args:
            data: Input DataFrame to verify.

        Returns:
            DataFrame with all required columns.

        Raises:
            ValueError: If required columns cannot be created.
        """
        pass


class FeatureEngineer(abc.ABC):
    """
    Interface for feature engineering components.

    Responsible for transforming raw data into features suitable for model training.
    """

    @abc.abstractmethod
    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.

        Raises:
            ValueError: If features cannot be engineered.
        """
        pass

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> "FeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to.
            **kwargs: Additional arguments for fitting.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the feature engineer cannot be fit to the data.
        """
        pass

    @abc.abstractmethod
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
        pass


class ModelBuilder(abc.ABC):
    """
    Interface for model building components.

    Responsible for creating and configuring machine learning models.
    """

    @abc.abstractmethod
    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.

        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        pass

    @abc.abstractmethod
    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for hyperparameter optimization.

        Returns:
            Optimized model pipeline.

        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        pass


class ModelTrainer(abc.ABC):
    """
    Interface for model training components.

    Responsible for training machine learning models on prepared data.
    """

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass


class ModelEvaluator(abc.ABC):
    """
    Interface for model evaluation components.

    Responsible for evaluating trained models and analyzing their performance.
    """

    @abc.abstractmethod
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

        Raises:
            ValueError: If the model cannot be evaluated.
        """
        pass

    @abc.abstractmethod
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

        Raises:
            ValueError: If predictions cannot be analyzed.
        """
        pass


class ModelSerializer(abc.ABC):
    """
    Interface for model serialization components.

    Responsible for saving and loading trained models.
    """

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
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
        pass


class Predictor(abc.ABC):
    """
    Interface for prediction components.

    Responsible for making predictions using trained models.
    """

    @abc.abstractmethod
    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make predictions using a trained model.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing predictions.

        Raises:
            ValueError: If predictions cannot be made.
        """
        pass

    @abc.abstractmethod
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
            ValueError: If probability predictions cannot be made.
        """
        pass


class PipelineComponent(abc.ABC):
    """
    Base interface for all pipeline components.

    Provides common functionality for pipeline components.
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        pass

    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.

        Raises:
            ValueError: If the configuration is invalid.
        """
        pass
