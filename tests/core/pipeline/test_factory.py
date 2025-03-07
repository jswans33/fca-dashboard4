"""
Tests for the PipelineFactory class.

This module contains tests for the PipelineFactory class, which is responsible
for creating pipeline components with proper dependencies.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory, PipelineFactoryError
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
from nexusml.core.pipeline.registry import ComponentRegistry


class MockDataLoader(DataLoader):
    """Mock implementation of DataLoader for testing."""

    def load_data(self, data_path=None, **kwargs):
        return pd.DataFrame()

    def get_config(self):
        return {}


class MockDataPreprocessor(DataPreprocessor):
    """Mock implementation of DataPreprocessor for testing."""

    def preprocess(self, data, **kwargs):
        return pd.DataFrame()

    def verify_required_columns(self, data):
        return pd.DataFrame()


class MockFeatureEngineer(FeatureEngineer):
    """Mock implementation of FeatureEngineer for testing."""

    def engineer_features(self, data, **kwargs):
        return pd.DataFrame()

    def fit(self, data, **kwargs):
        return self

    def transform(self, data, **kwargs):
        return pd.DataFrame()


class MockModelBuilder(ModelBuilder):
    """Mock implementation of ModelBuilder for testing."""

    def build_model(self, **kwargs):
        return Pipeline([("mock", MagicMock())])

    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        return model


class MockModelTrainer(ModelTrainer):
    """Mock implementation of ModelTrainer for testing."""

    def train(self, model, x_train, y_train, **kwargs):
        return model

    def cross_validate(self, model, x, y, **kwargs):
        return {"accuracy": [0.9]}


class MockModelEvaluator(ModelEvaluator):
    """Mock implementation of ModelEvaluator for testing."""

    def evaluate(self, model, x_test, y_test, **kwargs):
        return {"accuracy": 0.9}

    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        return {"confusion_matrix": [[1, 0], [0, 1]]}


class MockModelSerializer(ModelSerializer):
    """Mock implementation of ModelSerializer for testing."""

    def save_model(self, model, path, **kwargs):
        pass

    def load_model(self, path, **kwargs):
        return Pipeline([("mock", MagicMock())])


class MockPredictor(Predictor):
    """Mock implementation of Predictor for testing."""

    def predict(self, model, data, **kwargs):
        return pd.DataFrame()

    def predict_proba(self, model, data, **kwargs):
        return {"target": pd.DataFrame()}


class TestPipelineFactory(unittest.TestCase):
    """Test cases for the PipelineFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()
        self.container = DIContainer()

        # Register mock components
        self.registry.register(DataLoader, "mock", MockDataLoader)
        self.registry.register(DataPreprocessor, "mock", MockDataPreprocessor)
        self.registry.register(FeatureEngineer, "mock", MockFeatureEngineer)
        self.registry.register(ModelBuilder, "mock", MockModelBuilder)
        self.registry.register(ModelTrainer, "mock", MockModelTrainer)
        self.registry.register(ModelEvaluator, "mock", MockModelEvaluator)
        self.registry.register(ModelSerializer, "mock", MockModelSerializer)
        self.registry.register(Predictor, "mock", MockPredictor)

        # Set default implementations
        self.registry.set_default_implementation(DataLoader, "mock")
        self.registry.set_default_implementation(DataPreprocessor, "mock")
        self.registry.set_default_implementation(FeatureEngineer, "mock")
        self.registry.set_default_implementation(ModelBuilder, "mock")
        self.registry.set_default_implementation(ModelTrainer, "mock")
        self.registry.set_default_implementation(ModelEvaluator, "mock")
        self.registry.set_default_implementation(ModelSerializer, "mock")
        self.registry.set_default_implementation(Predictor, "mock")

        # Create factory
        self.factory = PipelineFactory(self.registry, self.container)

    def test_create_data_loader(self):
        """Test creating a data loader."""
        # Create a data loader
        data_loader = self.factory.create_data_loader()

        # Verify it's the correct type
        self.assertIsInstance(data_loader, MockDataLoader)

    def test_create_data_loader_with_name(self):
        """Test creating a data loader with a specific name."""

        # Register another data loader
        class AnotherDataLoader(DataLoader):
            def load_data(self, data_path=None, **kwargs):
                return pd.DataFrame()

            def get_config(self):
                return {}

        self.registry.register(DataLoader, "another", AnotherDataLoader)

        # Create a data loader with a specific name
        data_loader = self.factory.create_data_loader("another")

        # Verify it's the correct type
        self.assertIsInstance(data_loader, AnotherDataLoader)

    def test_create_data_loader_nonexistent(self):
        """Test creating a data loader that doesn't exist."""
        with pytest.raises(PipelineFactoryError):
            self.factory.create_data_loader("nonexistent")

    def test_create_data_preprocessor(self):
        """Test creating a data preprocessor."""
        # Create a data preprocessor
        preprocessor = self.factory.create_data_preprocessor()

        # Verify it's the correct type
        self.assertIsInstance(preprocessor, MockDataPreprocessor)

    def test_create_feature_engineer(self):
        """Test creating a feature engineer."""
        # Create a feature engineer
        engineer = self.factory.create_feature_engineer()

        # Verify it's the correct type
        self.assertIsInstance(engineer, MockFeatureEngineer)

    def test_create_model_builder(self):
        """Test creating a model builder."""
        # Create a model builder
        builder = self.factory.create_model_builder()

        # Verify it's the correct type
        self.assertIsInstance(builder, MockModelBuilder)

    def test_create_model_trainer(self):
        """Test creating a model trainer."""
        # Create a model trainer
        trainer = self.factory.create_model_trainer()

        # Verify it's the correct type
        self.assertIsInstance(trainer, MockModelTrainer)

    def test_create_model_evaluator(self):
        """Test creating a model evaluator."""
        # Create a model evaluator
        evaluator = self.factory.create_model_evaluator()

        # Verify it's the correct type
        self.assertIsInstance(evaluator, MockModelEvaluator)

    def test_create_model_serializer(self):
        """Test creating a model serializer."""
        # Create a model serializer
        serializer = self.factory.create_model_serializer()

        # Verify it's the correct type
        self.assertIsInstance(serializer, MockModelSerializer)

    def test_create_predictor(self):
        """Test creating a predictor."""
        # Create a predictor
        predictor = self.factory.create_predictor()

        # Verify it's the correct type
        self.assertIsInstance(predictor, MockPredictor)

    def test_create_with_dependencies(self):
        """Test creating a component with dependencies."""

        # Mock a component class with dependencies
        class ComponentWithDependencies:
            def __init__(self, data_loader, preprocessor):
                self.data_loader = data_loader
                self.preprocessor = preprocessor

        # Register the component
        self.registry.register(ComponentWithDependencies, "mock", ComponentWithDependencies)
        self.registry.set_default_implementation(ComponentWithDependencies, "mock")

        # Create the component
        component = self.factory.create(ComponentWithDependencies)

        # Verify it has the correct dependencies
        self.assertIsInstance(component.data_loader, MockDataLoader)
        self.assertIsInstance(component.preprocessor, MockDataPreprocessor)

    def test_create_with_config(self):
        """Test creating a component with configuration."""

        # Mock a component class with configuration
        class ComponentWithConfig:
            def __init__(self, config=None):
                self.config = config or {}

        # Register the component
        self.registry.register(ComponentWithConfig, "mock", ComponentWithConfig)
        self.registry.set_default_implementation(ComponentWithConfig, "mock")

        # Create the component with configuration
        config = {"param": "value"}
        component = self.factory.create(ComponentWithConfig, config=config)

        # Verify it has the correct configuration
        self.assertEqual(component.config, config)
