"""
Pipeline Stage Interfaces Module

This module defines the interfaces for all pipeline stages in the NexusML suite.
Each stage represents a distinct step in the pipeline execution process and follows
the Single Responsibility Principle (SRP) from SOLID.
"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.src.pipeline.context import PipelineContext


class PipelineStage(abc.ABC):
    """
    Base interface for all pipeline stages.

    A pipeline stage represents a distinct step in the pipeline execution process.
    Each stage has a single responsibility and can be composed with other stages
    to form a complete pipeline.
    """

    @abc.abstractmethod
    def execute(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the stage.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the stage cannot be executed.
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the stage.

        Returns:
            Stage name.
        """
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the stage.

        Returns:
            Stage description.
        """
        pass

    @abc.abstractmethod
    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.

        Raises:
            ValueError: If the context is invalid.
        """
        pass


class DataLoadingStage(PipelineStage):
    """
    Interface for data loading stages.

    Responsible for loading data from various sources and storing it in the context.
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


class ValidationStage(PipelineStage):
    """
    Interface for data validation stages.

    Responsible for validating data against requirements and storing validation
    results in the context.
    """

    @abc.abstractmethod
    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the input data.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.

        Raises:
            ValueError: If the data cannot be validated.
        """
        pass


class FeatureEngineeringStage(PipelineStage):
    """
    Interface for feature engineering stages.

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


class ModelBuildingStage(PipelineStage):
    """
    Interface for model building stages.

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


class ModelTrainingStage(PipelineStage):
    """
    Interface for model training stages.

    Responsible for training machine learning models on prepared data.
    """

    @abc.abstractmethod
    def train_model(
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


class ModelEvaluationStage(PipelineStage):
    """
    Interface for model evaluation stages.

    Responsible for evaluating trained models and analyzing their performance.
    """

    @abc.abstractmethod
    def evaluate_model(
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


class ModelSavingStage(PipelineStage):
    """
    Interface for model saving stages.

    Responsible for saving trained models and associated metadata.
    """

    @abc.abstractmethod
    def save_model(
        self,
        model: Pipeline,
        path: Union[str, Path],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Save a trained model and its metadata to disk.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            metadata: Model metadata to save.
            **kwargs: Additional arguments for saving.

        Raises:
            IOError: If the model cannot be saved.
        """
        pass


class DataSplittingStage(PipelineStage):
    """
    Interface for data splitting stages.

    Responsible for splitting data into training and testing sets.
    """

    @abc.abstractmethod
    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).

        Raises:
            ValueError: If the data cannot be split.
        """
        pass


class PredictionStage(PipelineStage):
    """
    Interface for prediction stages.

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
