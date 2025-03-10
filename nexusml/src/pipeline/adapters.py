"""
Pipeline Adapters Module

This module provides adapter classes that implement the pipeline interfaces
but delegate to existing code. These adapters ensure backward compatibility
while allowing the new interface-based architecture to be used.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.core import (
    data_preprocessing,
    evaluation,
    feature_engineering,
    model_building,
)
from nexusml.core.pipeline.base import (
    BaseDataLoader,
    BaseDataPreprocessor,
    BaseFeatureEngineer,
    BaseModelBuilder,
    BaseModelEvaluator,
    BaseModelSerializer,
    BaseModelTrainer,
    BasePredictor,
)


class LegacyDataLoaderAdapter(BaseDataLoader):
    """
    Adapter for the legacy data loading functionality.

    This adapter implements the DataLoader interface but delegates to the
    existing data_preprocessing module.
    """

    def __init__(
        self,
        name: str = "LegacyDataLoader",
        description: str = "Adapter for legacy data loading",
        config_path: Optional[str] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config_path: Path to the configuration file. If None, uses default paths.
        """
        super().__init__(name, description, config_path)

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data using the legacy data_preprocessing module.

        Args:
            data_path: Path to the data file. If None, uses the default path.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        return data_preprocessing.load_and_preprocess_data(data_path)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration.
        """
        return data_preprocessing.load_data_config()


class LegacyDataPreprocessorAdapter(BaseDataPreprocessor):
    """
    Adapter for the legacy data preprocessing functionality.

    This adapter implements the DataPreprocessor interface but delegates to the
    existing data_preprocessing module.
    """

    def __init__(
        self,
        name: str = "LegacyDataPreprocessor",
        description: str = "Adapter for legacy data preprocessing",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, loads from file.
        """
        if config is None:
            config = data_preprocessing.load_data_config()
        super().__init__(name, description, config)

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data using the legacy data_preprocessing module.

        Args:
            data: Input DataFrame to preprocess.
            **kwargs: Additional arguments for preprocessing.

        Returns:
            Preprocessed DataFrame.

        Raises:
            ValueError: If the data cannot be preprocessed.
        """
        # The legacy load_and_preprocess_data function already includes preprocessing
        # So we just need to verify the required columns
        return self.verify_required_columns(data)

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
        return data_preprocessing.verify_required_columns(data, self.config)


class LegacyFeatureEngineerAdapter(BaseFeatureEngineer):
    """
    Adapter for the legacy feature engineering functionality.

    This adapter implements the FeatureEngineer interface but delegates to the
    existing feature_engineering module.
    """

    def __init__(
        self,
        name: str = "LegacyFeatureEngineer",
        description: str = "Adapter for legacy feature engineering",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)
        self._generic_engineer = feature_engineering.GenericFeatureEngineer()

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features using the legacy feature_engineering module.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.

        Raises:
            ValueError: If features cannot be engineered.
        """
        return feature_engineering.enhance_features(data)

    def fit(self, data: pd.DataFrame, **kwargs) -> "LegacyFeatureEngineerAdapter":
        """
        Fit the feature engineer to the input data.

        The legacy feature engineering doesn't have a separate fit step,
        so this method just marks the engineer as fitted.

        Args:
            data: Input DataFrame to fit to.
            **kwargs: Additional arguments for fitting.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If the feature engineer cannot be fit to the data.
        """
        self._is_fitted = True
        return self

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
        if not self._is_fitted:
            raise ValueError(
                "Feature engineer must be fitted before transform can be called."
            )

        return self._generic_engineer.transform(data)


class LegacyModelBuilderAdapter(BaseModelBuilder):
    """
    Adapter for the legacy model building functionality.

    This adapter implements the ModelBuilder interface but delegates to the
    existing model_building module.
    """

    def __init__(
        self,
        name: str = "LegacyModelBuilder",
        description: str = "Adapter for legacy model building",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a model using the legacy model_building module.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.

        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        return model_building.build_enhanced_model(**kwargs)

    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model using the legacy model_building module.

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
        return model_building.optimize_hyperparameters(model, x_train, y_train)


class LegacyModelEvaluatorAdapter(BaseModelEvaluator):
    """
    Adapter for the legacy model evaluation functionality.

    This adapter implements the ModelEvaluator interface but delegates to the
    existing evaluation module.
    """

    def __init__(
        self,
        name: str = "LegacyModelEvaluator",
        description: str = "Adapter for legacy model evaluation",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)

    def evaluate(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model using the legacy evaluation module.

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
        # The legacy enhanced_evaluation function returns predictions, not metrics
        # So we need to convert the predictions to metrics
        y_pred = evaluation.enhanced_evaluation(model, x_test, y_test)

        # Calculate metrics using the base class implementation
        return super().evaluate(model, x_test, y_test, **kwargs)

    def analyze_predictions(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze model predictions using the legacy evaluation module.

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
        # Call the legacy analysis functions
        # The legacy functions expect a Series for x_test, but we have a DataFrame
        # So we need to convert it to a Series if it has a single column
        if isinstance(x_test, pd.DataFrame) and len(x_test.columns) == 1:
            x_test_series = x_test.iloc[:, 0]
            evaluation.analyze_other_category_features(
                model, x_test_series, y_test, y_pred
            )
            evaluation.analyze_other_misclassifications(x_test_series, y_test, y_pred)
        else:
            # If x_test has multiple columns, we can't convert it to a Series
            # So we'll skip the legacy analysis functions
            print(
                "Warning: Skipping legacy analysis functions because x_test has multiple columns"
            )

        # Return the analysis results from the base class implementation
        return super().analyze_predictions(model, x_test, y_test, y_pred, **kwargs)


class LegacyModelSerializerAdapter(BaseModelSerializer):
    """
    Adapter for model serialization.

    This adapter implements the ModelSerializer interface but uses the
    standard pickle module for serialization.
    """

    def __init__(
        self,
        name: str = "LegacyModelSerializer",
        description: str = "Adapter for model serialization",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)

    # The base class implementation already uses pickle for serialization,
    # so we don't need to override the methods


class LegacyPredictorAdapter(BasePredictor):
    """
    Adapter for making predictions.

    This adapter implements the Predictor interface but uses the
    standard scikit-learn predict method.
    """

    def __init__(
        self,
        name: str = "LegacyPredictor",
        description: str = "Adapter for making predictions",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)

    # The base class implementation already uses the standard predict method,
    # so we don't need to override the methods


class LegacyModelTrainerAdapter(BaseModelTrainer):
    """
    Adapter for model training.

    This adapter implements the ModelTrainer interface but uses the
    standard scikit-learn fit method.
    """

    def __init__(
        self,
        name: str = "LegacyModelTrainer",
        description: str = "Adapter for model training",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter.

        Args:
            name: Component name.
            description: Component description.
            config: Configuration dictionary. If None, uses an empty dictionary.
        """
        super().__init__(name, description, config)

    # The base class implementation already uses the standard fit method,
    # so we don't need to override the methods
