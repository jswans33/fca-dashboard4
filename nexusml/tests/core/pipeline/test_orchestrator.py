"""
Unit tests for the PipelineOrchestrator class.
"""

import json
import logging
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
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
from nexusml.core.pipeline.orchestrator import (
    PipelineOrchestrator,
    PipelineOrchestratorError,
)
from nexusml.core.pipeline.registry import ComponentRegistry


class MockDataLoader:
    """Mock implementation of DataLoader for testing."""

    def load_data(self, data_path=None, **kwargs):
        """Mock implementation of load_data."""
        return pd.DataFrame(
            {
                "equipment_tag": ["tag1", "tag2", "tag3"],
                "manufacturer": ["mfg1", "mfg2", "mfg3"],
                "model": ["model1", "model2", "model3"],
                "category_name": ["cat1", "cat2", "cat3"],
                "uniformat_code": ["code1", "code2", "code3"],
                "mcaa_system_category": ["sys1", "sys2", "sys3"],
                "Equipment_Type": ["type1", "type2", "type3"],
                "System_Subtype": ["subtype1", "subtype2", "subtype3"],
            }
        )

    def get_config(self):
        """Mock implementation of get_config."""
        return {"key": "value"}


class MockDataPreprocessor:
    """Mock implementation of DataPreprocessor for testing."""

    def preprocess(self, data, **kwargs):
        """Mock implementation of preprocess."""
        return data

    def verify_required_columns(self, data):
        """Mock implementation of verify_required_columns."""
        return data


class MockFeatureEngineer:
    """Mock implementation of FeatureEngineer for testing."""

    def engineer_features(self, data, **kwargs):
        """Mock implementation of engineer_features."""
        return data

    def fit(self, data, **kwargs):
        """Mock implementation of fit."""
        data["combined_text"] = (
            data["equipment_tag"] + " " + data["manufacturer"] + " " + data["model"]
        )
        data["service_life"] = 20.0
        return self

    def transform(self, data, **kwargs):
        """Mock implementation of transform."""
        data["combined_text"] = (
            data["equipment_tag"] + " " + data["manufacturer"] + " " + data["model"]
        )
        data["service_life"] = 20.0
        return data


class MockModelBuilder:
    """Mock implementation of ModelBuilder for testing."""

    def build_model(self, **kwargs):
        """Mock implementation of build_model."""
        return Pipeline([("classifier", RandomForestClassifier())])

    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """Mock implementation of optimize_hyperparameters."""
        return model


class MockModelTrainer:
    """Mock implementation of ModelTrainer for testing."""

    def train(self, model, x_train, y_train, **kwargs):
        """Mock implementation of train."""
        # For testing purposes, we'll just mock the training
        # Instead of actually fitting the model, we'll just pretend it's fitted

        # Set fitted_ attribute on the classifier to avoid NotFittedError
        if hasattr(model, "steps") and len(model.steps) > 0:
            classifier = model.steps[-1][1]
            # Set attributes that make the classifier appear fitted
            classifier.n_features_in_ = 1
            classifier.classes_ = np.array(["cat1", "cat2", "cat3"])
            classifier.n_classes_ = 3
            classifier.n_outputs_ = 1

            # For RandomForestClassifier
            if isinstance(classifier, RandomForestClassifier):
                from sklearn.tree import DecisionTreeClassifier

                # Create a dummy estimator
                dummy_estimator = DecisionTreeClassifier()
                dummy_estimator.tree_ = mock.MagicMock()
                dummy_estimator.classes_ = classifier.classes_
                dummy_estimator.n_outputs_ = 1
                dummy_estimator.n_classes_ = len(classifier.classes_)
                dummy_estimator.n_features_in_ = 1

                # Set estimators for RandomForestClassifier
                classifier.estimators_ = [dummy_estimator]

                # Add predict method to the classifier
                def mock_predict(X):
                    return np.array(["cat1"] * len(X))

                classifier.predict = mock_predict

                # Add predict_proba method to the classifier
                def mock_predict_proba(X):
                    return np.array([[0.8, 0.1, 0.1]] * len(X))

                classifier.predict_proba = mock_predict_proba

            # Add predict method to the model
            def mock_model_predict(X):
                return np.array(["cat1"] * len(X))

            model.predict = mock_model_predict

        return model

    def cross_validate(self, model, x, y, **kwargs):
        """Mock implementation of cross_validate."""
        return {"accuracy": [0.8, 0.9, 0.85], "f1_macro": [0.75, 0.85, 0.8]}


class MockModelEvaluator:
    """Mock implementation of ModelEvaluator for testing."""

    def evaluate(self, model, x_test, y_test, **kwargs):
        """Mock implementation of evaluate."""
        return {
            "accuracy": 0.85,
            "f1_macro": 0.8,
            "precision": 0.82,
            "recall": 0.79,
        }

    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        """Mock implementation of analyze_predictions."""
        return {
            "confusion_matrix": [[10, 2], [3, 15]],
            "classification_report": "mock_report",
        }


class MockModelSerializer:
    """Mock implementation of ModelSerializer for testing."""

    def save_model(self, model, path, **kwargs):
        """Mock implementation of save_model."""
        # Just create an empty file for testing
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Don't actually try to pickle the model, just create an empty file
        with open(path, "wb") as f:
            # Write a simple placeholder instead of trying to pickle the model
            f.write(b"mock_model_data")

    def load_model(self, path, **kwargs):
        """Mock implementation of load_model."""
        return Pipeline([("classifier", RandomForestClassifier())])


class MockPredictor:
    """Mock implementation of Predictor for testing."""

    def predict(self, model, data, **kwargs):
        """Mock implementation of predict."""
        return pd.DataFrame(
            {
                "category_name": ["cat1", "cat2", "cat3"],
                "uniformat_code": ["code1", "code2", "code3"],
                "mcaa_system_category": ["sys1", "sys2", "sys3"],
                "Equipment_Type": ["type1", "type2", "type3"],
                "System_Subtype": ["subtype1", "subtype2", "subtype3"],
            }
        )

    def predict_proba(self, model, data, **kwargs):
        """Mock implementation of predict_proba."""
        return {
            "category_name": pd.DataFrame(
                {
                    "cat1": [0.8, 0.1, 0.1],
                    "cat2": [0.1, 0.8, 0.1],
                    "cat3": [0.1, 0.1, 0.8],
                }
            ),
        }


@pytest.fixture
def mock_factory():
    """Create a mock PipelineFactory for testing."""
    factory = mock.MagicMock(spec=PipelineFactory)
    factory.create_data_loader.return_value = MockDataLoader()
    factory.create_data_preprocessor.return_value = MockDataPreprocessor()
    factory.create_feature_engineer.return_value = MockFeatureEngineer()
    factory.create_model_builder.return_value = MockModelBuilder()
    factory.create_model_trainer.return_value = MockModelTrainer()
    factory.create_model_evaluator.return_value = MockModelEvaluator()
    factory.create_model_serializer.return_value = MockModelSerializer()
    factory.create_predictor.return_value = MockPredictor()
    return factory


@pytest.fixture
def orchestrator(mock_factory):
    """Create a PipelineOrchestrator for testing."""
    context = PipelineContext()
    return PipelineOrchestrator(mock_factory, context)


class TestPipelineOrchestrator:
    """Tests for the PipelineOrchestrator class."""

    def test_initialization(self, mock_factory):
        """Test that the orchestrator is initialized correctly."""
        context = PipelineContext()
        orchestrator = PipelineOrchestrator(mock_factory, context)
        assert orchestrator.factory == mock_factory
        assert orchestrator.context == context

    def test_initialization_with_defaults(self, mock_factory):
        """Test that the orchestrator is initialized correctly with defaults."""
        orchestrator = PipelineOrchestrator(mock_factory)
        assert orchestrator.factory == mock_factory
        assert isinstance(orchestrator.context, PipelineContext)

    def test_train_model(self, orchestrator, tmp_path):
        """Test that the train_model method executes the pipeline correctly."""
        output_dir = tmp_path / "models"
        model, metrics = orchestrator.train_model(
            data_path="dummy_path",
            feature_config_path="dummy_config",
            output_dir=str(output_dir),
            model_name="test_model",
        )

        # Check that the model is returned
        assert isinstance(model, Pipeline)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

        # Check that the model is saved
        assert (output_dir / "test_model.pkl").exists()
        assert (output_dir / "test_model_metadata.json").exists()

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("metrics") is not None
        assert orchestrator.context.get("model_path") is not None
        assert orchestrator.context.get("metadata_path") is not None

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_loading" in execution_times
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "data_splitting" in execution_times
        assert "model_building" in execution_times
        assert "model_training" in execution_times
        assert "model_evaluation" in execution_times
        assert "model_saving" in execution_times

    def test_train_model_with_hyperparameter_optimization(self, orchestrator):
        """Test that the train_model method executes hyperparameter optimization."""
        model, metrics = orchestrator.train_model(
            data_path="dummy_path",
            feature_config_path="dummy_config",
            optimize_hyperparameters=True,
        )

        # Check that the model is returned
        assert isinstance(model, Pipeline)

        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("metrics") is not None

    def test_train_model_error_handling(self, mock_factory):
        """Test that the train_model method handles errors correctly."""
        # Create a factory that raises an error
        factory = mock.MagicMock(spec=PipelineFactory)
        factory.create_data_loader.side_effect = Exception("Test error")

        # Create an orchestrator with the factory
        orchestrator = PipelineOrchestrator(factory)

        # Test that the error is handled correctly
        with pytest.raises(PipelineOrchestratorError) as excinfo:
            orchestrator.train_model(data_path="dummy_path")

        # Check that the error message is correct
        assert "Test error" in str(excinfo.value)

        # Check that the context is updated
        assert orchestrator.context.status == "failed"

    def test_predict(self, orchestrator):
        """Test that the predict method executes the pipeline correctly."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Test the predict method
        predictions = orchestrator.predict(
            model=model,
            data=pd.DataFrame(
                {
                    "equipment_tag": ["tag1", "tag2", "tag3"],
                    "manufacturer": ["mfg1", "mfg2", "mfg3"],
                    "model": ["model1", "model2", "model3"],
                }
            ),
        )

        # Check that predictions are returned
        assert isinstance(predictions, pd.DataFrame)
        assert "category_name" in predictions.columns
        assert "uniformat_code" in predictions.columns
        assert "mcaa_system_category" in predictions.columns
        assert "Equipment_Type" in predictions.columns
        assert "System_Subtype" in predictions.columns

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("predictions") is not None

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "prediction" in execution_times

    def test_predict_with_model_path(self, orchestrator, tmp_path):
        """Test that the predict method works with a model path."""
        # Create a model file for testing
        model = Pipeline([("classifier", RandomForestClassifier())])
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Test the predict method
        predictions = orchestrator.predict(
            model_path=str(model_path),
            data=pd.DataFrame(
                {
                    "equipment_tag": ["tag1", "tag2", "tag3"],
                    "manufacturer": ["mfg1", "mfg2", "mfg3"],
                    "model": ["model1", "model2", "model3"],
                }
            ),
        )

        # Check that predictions are returned
        assert isinstance(predictions, pd.DataFrame)
        assert "category_name" in predictions.columns

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("predictions") is not None

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "model_loading" in execution_times
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "prediction" in execution_times

    def test_predict_with_data_path(self, orchestrator, tmp_path):
        """Test that the predict method works with a data path."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Create a data file for testing
        data = pd.DataFrame(
            {
                "equipment_tag": ["tag1", "tag2", "tag3"],
                "manufacturer": ["mfg1", "mfg2", "mfg3"],
                "model": ["model1", "model2", "model3"],
            }
        )
        data_path = tmp_path / "data.csv"
        data.to_csv(data_path, index=False)

        # Test the predict method
        predictions = orchestrator.predict(
            model=model,
            data_path=str(data_path),
        )

        # Check that predictions are returned
        assert isinstance(predictions, pd.DataFrame)
        assert "category_name" in predictions.columns

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("predictions") is not None

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_loading" in execution_times
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "prediction" in execution_times

    def test_predict_with_output_path(self, orchestrator, tmp_path):
        """Test that the predict method saves predictions to a file."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Test the predict method
        output_path = tmp_path / "predictions.csv"
        predictions = orchestrator.predict(
            model=model,
            data=pd.DataFrame(
                {
                    "equipment_tag": ["tag1", "tag2", "tag3"],
                    "manufacturer": ["mfg1", "mfg2", "mfg3"],
                    "model": ["model1", "model2", "model3"],
                }
            ),
            output_path=str(output_path),
        )

        # Check that predictions are returned
        assert isinstance(predictions, pd.DataFrame)
        assert "category_name" in predictions.columns

        # Check that the predictions are saved
        assert output_path.exists()

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("predictions") is not None
        assert orchestrator.context.get("output_path") == str(output_path)

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "prediction" in execution_times
        assert "saving_predictions" in execution_times

    def test_predict_error_handling(self, mock_factory):
        """Test that the predict method handles errors correctly."""
        # Create a factory that raises an error
        factory = mock.MagicMock(spec=PipelineFactory)
        factory.create_data_preprocessor.side_effect = Exception("Test error")

        # Create an orchestrator with the factory
        orchestrator = PipelineOrchestrator(factory)

        # Test that the error is handled correctly
        with pytest.raises(PipelineOrchestratorError) as excinfo:
            orchestrator.predict(
                model=Pipeline([("classifier", RandomForestClassifier())]),
                data=pd.DataFrame(
                    {
                        "equipment_tag": ["tag1", "tag2", "tag3"],
                        "manufacturer": ["mfg1", "mfg2", "mfg3"],
                        "model": ["model1", "model2", "model3"],
                    }
                ),
            )

        # Check that the error message is correct
        assert "Test error" in str(excinfo.value)

        # Check that the context is updated
        assert orchestrator.context.status == "failed"

    def test_evaluate(self, orchestrator):
        """Test that the evaluate method executes the pipeline correctly."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Mock the model to appear fitted
        classifier = model.steps[0][1]
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["cat1", "cat2", "cat3"])
        classifier.n_classes_ = 3
        classifier.n_outputs_ = 1

        # Add predict method to the model
        def mock_predict(X, **kwargs):
            # Return a 2D array with the correct number of columns
            return np.array([["cat1", "code1", "sys1", "type1", "subtype1"]] * len(X))

        model.predict = mock_predict
        classifier.predict = mock_predict

        # Create dummy estimators for RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        dummy_estimator = DecisionTreeClassifier()
        dummy_estimator.tree_ = mock.MagicMock()
        dummy_estimator.classes_ = classifier.classes_
        dummy_estimator.n_outputs_ = 1
        dummy_estimator.n_classes_ = len(classifier.classes_)
        dummy_estimator.n_features_in_ = 1
        classifier.estimators_ = [dummy_estimator]

        # Test the evaluate method
        results = orchestrator.evaluate(
            model=model,
            data=pd.DataFrame(
                {
                    "equipment_tag": ["tag1", "tag2", "tag3"],
                    "manufacturer": ["mfg1", "mfg2", "mfg3"],
                    "model": ["model1", "model2", "model3"],
                    "category_name": ["cat1", "cat2", "cat3"],
                    "uniformat_code": ["code1", "code2", "code3"],
                    "mcaa_system_category": ["sys1", "sys2", "sys3"],
                    "Equipment_Type": ["type1", "type2", "type3"],
                    "System_Subtype": ["subtype1", "subtype2", "subtype3"],
                }
            ),
        )

        # Check that results are returned
        assert isinstance(results, dict)
        assert "metrics" in results
        assert "analysis" in results
        assert "accuracy" in results["metrics"]
        assert "confusion_matrix" in results["analysis"]

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("metrics") is not None
        assert orchestrator.context.get("analysis") is not None

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "data_preparation" in execution_times
        assert "model_evaluation" in execution_times

    def test_evaluate_with_output_path(self, orchestrator, tmp_path):
        """Test that the evaluate method saves results to a file."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Mock the model to appear fitted
        classifier = model.steps[0][1]
        classifier.n_features_in_ = 1
        classifier.classes_ = np.array(["cat1", "cat2", "cat3"])
        classifier.n_classes_ = 3
        classifier.n_outputs_ = 1

        # Add predict method to the model
        def mock_predict(X, **kwargs):
            # Return a 2D array with the correct number of columns
            return np.array([["cat1", "code1", "sys1", "type1", "subtype1"]] * len(X))

        model.predict = mock_predict
        classifier.predict = mock_predict

        # Create dummy estimators for RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        dummy_estimator = DecisionTreeClassifier()
        dummy_estimator.tree_ = mock.MagicMock()
        dummy_estimator.classes_ = classifier.classes_
        dummy_estimator.n_outputs_ = 1
        dummy_estimator.n_classes_ = len(classifier.classes_)
        dummy_estimator.n_features_in_ = 1
        classifier.estimators_ = [dummy_estimator]

        # Test the evaluate method
        output_path = tmp_path / "evaluation.json"
        results = orchestrator.evaluate(
            model=model,
            data=pd.DataFrame(
                {
                    "equipment_tag": ["tag1", "tag2", "tag3"],
                    "manufacturer": ["mfg1", "mfg2", "mfg3"],
                    "model": ["model1", "model2", "model3"],
                    "category_name": ["cat1", "cat2", "cat3"],
                    "uniformat_code": ["code1", "code2", "code3"],
                    "mcaa_system_category": ["sys1", "sys2", "sys3"],
                    "Equipment_Type": ["type1", "type2", "type3"],
                    "System_Subtype": ["subtype1", "subtype2", "subtype3"],
                }
            ),
            output_path=str(output_path),
        )

        # Check that results are returned
        assert isinstance(results, dict)
        assert "metrics" in results
        assert "analysis" in results

        # Check that the results are saved
        assert output_path.exists()
        with open(output_path, "r") as f:
            saved_results = json.load(f)
        assert "metrics" in saved_results
        assert "analysis" in saved_results
        assert "component_execution_times" in saved_results

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("data") is not None
        assert orchestrator.context.get("metrics") is not None
        assert orchestrator.context.get("analysis") is not None
        assert orchestrator.context.get("output_path") == str(output_path)

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "data_preprocessing" in execution_times
        assert "feature_engineering" in execution_times
        assert "data_preparation" in execution_times
        assert "model_evaluation" in execution_times
        assert "saving_evaluation" in execution_times

    def test_save_model(self, orchestrator, tmp_path):
        """Test that the save_model method saves a model to disk."""
        # Create a model for testing
        model = Pipeline([("classifier", RandomForestClassifier())])

        # Test the save_model method
        model_path = tmp_path / "model.pkl"
        saved_path = orchestrator.save_model(model, model_path)

        # Check that the model is saved
        assert model_path.exists()
        assert saved_path == str(model_path)

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("path") == str(model_path)

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "model_saving" in execution_times

    def test_load_model(self, orchestrator, tmp_path):
        """Test that the load_model method loads a model from disk."""
        # Create a model file for testing
        model = Pipeline([("classifier", RandomForestClassifier())])
        model_path = tmp_path / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Test the load_model method
        loaded_model = orchestrator.load_model(model_path)

        # Check that the model is loaded
        assert isinstance(loaded_model, Pipeline)

        # Check that the context is updated
        assert orchestrator.context.status == "completed"
        assert orchestrator.context.get("model") is not None
        assert orchestrator.context.get("path") == str(model_path)

        # Check that component execution times are recorded
        execution_times = orchestrator.context.get_component_execution_times()
        assert "model_loading" in execution_times

    def test_get_execution_summary(self, orchestrator):
        """Test that the get_execution_summary method returns a summary of the execution."""
        # Execute a pipeline
        orchestrator.train_model(data_path="dummy_path")

        # Get the execution summary
        summary = orchestrator.get_execution_summary()

        # Check that the summary contains the expected keys
        assert "status" in summary
        assert "metrics" in summary
        assert "component_execution_times" in summary
        assert "accessed_keys" in summary
        assert "modified_keys" in summary
        assert "start_time" in summary
        assert "end_time" in summary
        assert "total_execution_time" in summary

        # Check that the status is correct
        assert summary["status"] == "completed"

        # Check that component execution times are recorded
        assert "data_loading" in summary["component_execution_times"]
        assert "data_preprocessing" in summary["component_execution_times"]
        assert "feature_engineering" in summary["component_execution_times"]
        assert "data_splitting" in summary["component_execution_times"]
        assert "model_building" in summary["component_execution_times"]
        assert "model_training" in summary["component_execution_times"]
        assert "model_evaluation" in summary["component_execution_times"]
