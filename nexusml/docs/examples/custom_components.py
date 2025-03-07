#!/usr/bin/env python
"""
Custom Components Example for NexusML

This example demonstrates how to create custom components for NexusML.
It covers:
- Creating custom data loaders
- Creating custom feature engineers
- Creating custom model builders
- Registering custom components
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry


# Custom Data Loader
class ExcelDataLoader(DataLoader):
    """Custom data loader for Excel files with specific sheet handling."""

    def __init__(self, file_path: Optional[str] = None, sheet_name: str = "Sheet1"):
        """
        Initialize a new ExcelDataLoader.

        Args:
            file_path: Path to the Excel file
            sheet_name: Name of the sheet to load
        """
        self.file_path = file_path
        self.sheet_name = sheet_name

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.

        Args:
            data_path: Path to the Excel file (if None, uses self.file_path)
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the Excel file cannot be found
            ValueError: If the Excel format is invalid
        """
        path = data_path or self.file_path
        if path is None:
            raise ValueError("No data path provided")

        # Override sheet_name from kwargs if provided
        sheet_name = kwargs.pop("sheet_name", self.sheet_name)

        try:
            print(f"Loading data from {path}, sheet: {sheet_name}")
            return pd.read_excel(path, sheet_name=sheet_name, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Excel file not found: {path}")
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration
        """
        return {"file_path": self.file_path, "sheet_name": self.sheet_name}


# Custom Feature Engineer
class AdvancedTextFeatureEngineer(FeatureEngineer):
    """Custom feature engineer for text data with advanced processing."""

    def __init__(
        self,
        text_columns: Optional[List[str]] = None,
        combined_column: str = "combined_text",
        separator: str = " ",
        min_df: int = 2,
        max_features: Optional[int] = None,
    ):
        """
        Initialize a new AdvancedTextFeatureEngineer.

        Args:
            text_columns: List of text columns to combine
            combined_column: Name of the combined text column
            separator: Separator to use between combined fields
            min_df: Minimum document frequency for TF-IDF vectorizer
            max_features: Maximum number of features for TF-IDF vectorizer
        """
        self.text_columns = text_columns or ["description"]
        self.combined_column = combined_column
        self.separator = separator
        self.min_df = min_df
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features
            **kwargs: Additional arguments for feature engineering

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If features cannot be engineered
        """
        # Check if fitted
        if not self._is_fitted:
            # Fit and transform
            return self.fit(data, **kwargs).transform(data, **kwargs)
        else:
            # Transform only
            return self.transform(data, **kwargs)

    def fit(self, data: pd.DataFrame, **kwargs) -> "AdvancedTextFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the feature engineer cannot be fit to the data
        """
        # Create a copy of the input data
        result = data.copy()

        # Combine text columns
        result[self.combined_column] = (
            result[self.text_columns].fillna("").agg(self.separator.join, axis=1)
        )

        # Create vectorizer
        from sklearn.feature_extraction.text import CountVectorizer

        # Use CountVectorizer instead of TfidfVectorizer to avoid sparse matrix issues
        self._vectorizer = CountVectorizer(
            min_df=self.min_df,
            max_features=self.max_features,
            binary=True,  # Use binary features (0/1) instead of counts
            **kwargs.get("vectorizer_params", {}),
        )

        # Fit vectorizer
        self._vectorizer.fit(result[self.combined_column])
        self._is_fitted = True

        # Check if vectorizer is fitted and has feature names
        if hasattr(self._vectorizer, "get_feature_names_out"):
            feature_count = len(self._vectorizer.get_feature_names_out())
            print(f"Fitted vectorizer with {feature_count} features")
        else:
            print("Fitted vectorizer (feature count unavailable)")

        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If the data cannot be transformed
        """
        # Check if fitted
        if not self._is_fitted or self._vectorizer is None:
            raise ValueError("Feature engineer not fitted or vectorizer is None")

        # Create a copy of the input data
        result = data.copy()

        # Combine text columns
        result[self.combined_column] = (
            result[self.text_columns].fillna("").agg(self.separator.join, axis=1)
        )

        # Transform data - convert to dense array immediately to avoid sparse matrix issues
        features_array = self._vectorizer.transform(
            result[self.combined_column]
        ).todense()

        # Get feature names safely
        if hasattr(self._vectorizer, "get_feature_names_out"):
            feature_names = self._vectorizer.get_feature_names_out()
        else:
            # Fallback for older scikit-learn versions
            feature_names = [f"feature_{i}" for i in range(features_array.shape[1])]

        # Create feature DataFrame with prefix
        prefix = kwargs.get("feature_prefix", "text_")

        # Create DataFrame from feature array
        feature_df = pd.DataFrame(
            features_array,
            columns=[f"{prefix}{name}" for name in feature_names],
            index=result.index,
        )

        # Concatenate with original data
        result = pd.concat([result, feature_df], axis=1)

        print(f"Transformed data with {len(feature_names)} features")
        return result


# Custom Model Builder
class GradientBoostingModelBuilder(ModelBuilder):
    """Custom model builder using gradient boosting."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        """
        Initialize a new GradientBoostingModelBuilder.

        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of the individual regression estimators
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        Args:
            **kwargs: Configuration parameters for the model

        Returns:
            Configured model pipeline

        Raises:
            ValueError: If the model cannot be built with the given parameters
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Override parameters from kwargs if provided
        n_estimators = kwargs.get("n_estimators", self.n_estimators)
        learning_rate = kwargs.get("learning_rate", self.learning_rate)
        max_depth = kwargs.get("max_depth", self.max_depth)
        random_state = kwargs.get("random_state", self.random_state)

        # Create pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        print(
            f"Built gradient boosting model with {n_estimators} estimators, "
            f"learning rate {learning_rate}, max depth {max_depth}"
        )
        return pipeline

    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        Args:
            model: Model pipeline to optimize
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for hyperparameter optimization

        Returns:
            Optimized model pipeline

        Raises:
            ValueError: If hyperparameters cannot be optimized
        """
        from sklearn.model_selection import GridSearchCV

        # Define parameter grid
        param_grid = kwargs.get(
            "param_grid",
            {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__max_depth": [3, 5, 7],
            },
        )

        # Create grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=kwargs.get("cv", 5),
            scoring=kwargs.get("scoring", "accuracy"),
            n_jobs=kwargs.get("n_jobs", -1),
        )

        # Fit grid search
        print("Optimizing hyperparameters...")
        grid_search.fit(x_train, y_train)

        # Print best parameters
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")

        # Return best model
        return grid_search.best_estimator_


def main():
    """Main function to demonstrate custom components in NexusML."""
    print("NexusML Custom Components Example")
    print("=================================")

    # Step 1: Set up the component registry, DI container, and factory
    print("\nStep 1: Setting up the component registry, DI container, and factory")
    registry = ComponentRegistry()
    container = DIContainer()

    # Step 2: Register custom components
    print("\nStep 2: Registering custom components")
    registry.register(DataLoader, "excel", ExcelDataLoader)
    registry.register(FeatureEngineer, "advanced_text", AdvancedTextFeatureEngineer)
    registry.register(ModelBuilder, "gradient_boosting", GradientBoostingModelBuilder)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "excel")
    registry.set_default_implementation(FeatureEngineer, "advanced_text")
    registry.set_default_implementation(ModelBuilder, "gradient_boosting")

    # Create factory and orchestrator
    factory = PipelineFactory(registry, container)
    context = PipelineContext()
    orchestrator = PipelineOrchestrator(factory, context)

    # Step 3: Create components with custom configuration
    print("\nStep 3: Creating components with custom configuration")
    data_loader = factory.create_data_loader(
        file_path="examples/sample_data.xlsx", sheet_name="Sheet1"
    )
    feature_engineer = factory.create_feature_engineer(
        text_columns=["description", "manufacturer", "model"],
        combined_column="combined_text",
        separator=" | ",
        min_df=1,
        max_features=100,
    )
    model_builder = factory.create_model_builder(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )

    # Step 4: Use the components
    print("\nStep 4: Using the components")

    # Load data
    data = data_loader.load_data()
    print(f"Loaded data with {len(data)} rows and {len(data.columns)} columns")

    # Engineer features
    features = feature_engineer.engineer_features(data)
    print(
        f"Engineered features with {len(features)} rows and {len(features.columns)} columns"
    )

    # Build model
    model = model_builder.build_model()
    print(f"Built model: {model}")

    # Step 5: Train a model using the orchestrator with custom components
    print("\nStep 5: Training a model using the orchestrator with custom components")

    try:
        model, metrics = orchestrator.train_model(
            data_path="examples/sample_data.xlsx",
            test_size=0.3,
            random_state=42,
            optimize_hyperparameters=True,
            output_dir="outputs/models",
            model_name="custom_components_example",
        )

        print("Model training completed successfully")
        print(f"Model saved to: {orchestrator.context.get('model_path')}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error training model: {e}")


if __name__ == "__main__":
    main()
