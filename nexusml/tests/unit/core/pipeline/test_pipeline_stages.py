"""
Unit tests for the pipeline stages.

This module contains tests for the pipeline stage interfaces, base implementations,
and concrete implementations.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages import (
    BasePipelineStage,
    CSVDataLoadingStage,
    ColumnValidationStage,
    ConfigDrivenDataSplittingStage,
    ConfigDrivenFeatureEngineeringStage,
    ConfigDrivenModelBuildingStage,
    ConfigDrivenModelEvaluationStage,
    ConfigDrivenModelSavingStage,
    ConfigDrivenModelTrainingStage,
    ConfigDrivenPredictionStage,
    ConfigDrivenValidationStage,
    RandomSplittingStage,
    SimpleFeatureEngineeringStage,
    StandardModelTrainingStage,
    StandardPredictionStage,
)


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "equipment_tag": ["EQ001", "EQ002", "EQ003", "EQ004", "EQ005"],
            "description": ["Pump", "Fan", "Chiller", "Boiler", "Compressor"],
            "category_name": ["HVAC", "HVAC", "HVAC", "Plumbing", "HVAC"],
            "mcaa_system_category": ["Pump", "Fan", "Chiller", "Boiler", "Compressor"],
            "uniformat_code": ["D3010", "D3040", "D3050", "D3020", "D3060"],
            "Equipment_Type": ["Pump", "Fan", "Chiller", "Boiler", "Compressor"],
            "System_Subtype": ["Water", "Air", "Water", "Steam", "Air"],
            "service_life": [20, 15, 25, 30, 20],
        }
    )


@pytest.fixture
def sample_csv_file(sample_data):
    """Create a temporary CSV file with sample data for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_data.to_csv(f, index=False)

    file_path = f.name
    yield file_path

    # Clean up the temporary file
    os.unlink(file_path)


@pytest.fixture
def pipeline_context():
    """Create a pipeline context for testing."""
    context = PipelineContext()
    context.start()
    return context


class TestBasePipelineStage:
    """Tests for the BasePipelineStage class."""

    def test_init(self):
        """Test initialization of BasePipelineStage."""
        stage = BasePipelineStage("TestStage", "Test description")
        assert stage.get_name() == "TestStage"
        assert stage.get_description() == "Test description"

    def test_validate_context(self):
        """Test context validation."""
        stage = BasePipelineStage("TestStage", "Test description")
        context = PipelineContext()
        assert stage.validate_context(context) is True

    def test_execute(self, pipeline_context):
        """Test execute method."""
        # Create a subclass that implements _execute_impl
        class TestStage(BasePipelineStage):
            def _execute_impl(self, context, **kwargs):
                context.set("test_key", "test_value")

        stage = TestStage("TestStage", "Test description")
        stage.execute(pipeline_context)
        assert pipeline_context.get("test_key") == "test_value"

    def test_execute_error(self, pipeline_context):
        """Test execute method with error."""
        # Create a subclass that raises an error in _execute_impl
        class ErrorStage(BasePipelineStage):
            def _execute_impl(self, context, **kwargs):
                raise ValueError("Test error")

        stage = ErrorStage("ErrorStage", "Test description")
        with pytest.raises(ValueError, match="Test error"):
            stage.execute(pipeline_context)


class TestDataLoadingStage:
    """Tests for the data loading stages."""

    def test_csv_data_loading_stage(self, sample_csv_file, pipeline_context):
        """Test CSVDataLoadingStage."""
        stage = CSVDataLoadingStage()
        stage.execute(pipeline_context, data_path=sample_csv_file)
        
        # Check that data was loaded and stored in the context
        assert pipeline_context.has("data")
        data = pipeline_context.get("data")
        assert isinstance(data, pd.DataFrame)
        assert "equipment_tag" in data.columns
        assert "description" in data.columns
        assert len(data) > 0

    def test_csv_data_loading_stage_file_not_found(self, pipeline_context):
        """Test CSVDataLoadingStage with file not found."""
        stage = CSVDataLoadingStage()
        with pytest.raises(FileNotFoundError):
            stage.execute(pipeline_context, data_path="nonexistent_file.csv")


class TestValidationStage:
    """Tests for the validation stages."""

    def test_column_validation_stage(self, sample_data, pipeline_context):
        """Test ColumnValidationStage."""
        # Store sample data in the context
        pipeline_context.set("data", sample_data)
        
        # Create a validation stage with required columns
        stage = ColumnValidationStage(
            config={
                "required_columns": [
                    "equipment_tag",
                    "description",
                    "category_name",
                    "mcaa_system_category",
                ],
                "critical_columns": ["equipment_tag", "category_name"],
            }
        )
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that validation results were stored in the context
        assert pipeline_context.has("validation_results")
        results = pipeline_context.get("validation_results")
        assert results["valid"] is True
        assert len(results["issues"]) == 0

    def test_column_validation_stage_missing_columns(self, pipeline_context):
        """Test ColumnValidationStage with missing columns."""
        # Create a DataFrame with missing columns
        data = pd.DataFrame(
            {
                "equipment_tag": ["EQ001", "EQ002", "EQ003"],
                # Missing description, category_name, and mcaa_system_category
            }
        )
        pipeline_context.set("data", data)
        
        # Create a validation stage with required columns
        stage = ColumnValidationStage(
            config={
                "required_columns": [
                    "equipment_tag",
                    "description",
                    "category_name",
                    "mcaa_system_category",
                ],
                "critical_columns": ["equipment_tag", "category_name"],
            }
        )
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that validation results were stored in the context
        assert pipeline_context.has("validation_results")
        results = pipeline_context.get("validation_results")
        assert results["valid"] is False
        assert len(results["issues"]) > 0
        assert "Missing required columns" in results["issues"][0]


class TestFeatureEngineeringStage:
    """Tests for the feature engineering stages."""

    def test_simple_feature_engineering_stage(self, sample_data, pipeline_context):
        """Test SimpleFeatureEngineeringStage."""
        # Store sample data in the context
        pipeline_context.set("data", sample_data)
        
        # Create a feature engineering stage
        stage = SimpleFeatureEngineeringStage()
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that engineered data was stored in the context
        assert pipeline_context.has("engineered_data")
        engineered_data = pipeline_context.get("engineered_data")
        assert isinstance(engineered_data, pd.DataFrame)
        assert "combined_text" in engineered_data.columns
        assert "service_life" in engineered_data.columns


class TestDataSplittingStage:
    """Tests for the data splitting stages."""

    def test_random_splitting_stage(self, sample_data, pipeline_context):
        """Test RandomSplittingStage."""
        # Store sample data in the context
        pipeline_context.set("data", sample_data)
        
        # Create a data splitting stage
        stage = RandomSplittingStage(
            config={
                "test_size": 0.2,
                "random_state": 42,
            }
        )
        
        # Execute the stage
        stage.execute(
            pipeline_context,
            target_columns=["category_name", "mcaa_system_category"],
        )
        
        # Check that split data was stored in the context
        assert pipeline_context.has("x_train")
        assert pipeline_context.has("x_test")
        assert pipeline_context.has("y_train")
        assert pipeline_context.has("y_test")
        
        x_train = pipeline_context.get("x_train")
        x_test = pipeline_context.get("x_test")
        y_train = pipeline_context.get("y_train")
        y_test = pipeline_context.get("y_test")
        
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
        assert isinstance(y_test, pd.DataFrame)
        
        # Check that the split was done correctly
        assert len(x_train) + len(x_test) == len(sample_data)
        assert len(y_train) + len(y_test) == len(sample_data)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)


class TestModelBuildingStage:
    """Tests for the model building stages."""

    def test_config_driven_model_building_stage(self, pipeline_context):
        """Test ConfigDrivenModelBuildingStage."""
        # Create a model building stage
        stage = ConfigDrivenModelBuildingStage(
            config={
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42,
            }
        )
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that model was stored in the context
        assert pipeline_context.has("model")
        model = pipeline_context.get("model")
        assert isinstance(model, Pipeline)


class TestModelTrainingStage:
    """Tests for the model training stages."""

    def test_standard_model_training_stage(self, sample_data, pipeline_context):
        """Test StandardModelTrainingStage."""
        # Create a simple model
        model = Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        pipeline_context.set("model", model)
        
        # Create training data
        x_train = sample_data[["description", "service_life"]]
        y_train = sample_data[["category_name", "mcaa_system_category"]]
        pipeline_context.set("x_train", x_train)
        pipeline_context.set("y_train", y_train)
        
        # Create a model training stage
        stage = StandardModelTrainingStage()
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that trained model was stored in the context
        assert pipeline_context.has("trained_model")
        trained_model = pipeline_context.get("trained_model")
        assert isinstance(trained_model, Pipeline)


class TestModelEvaluationStage:
    """Tests for the model evaluation stages."""

    def test_config_driven_model_evaluation_stage(self, sample_data, pipeline_context):
        """Test ConfigDrivenModelEvaluationStage."""
        # Create a simple model and train it
        model = Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Create training and test data
        x = sample_data[["description", "service_life"]]
        y = sample_data[["category_name", "mcaa_system_category"]]
        
        # Train the model
        model.fit(x, y)
        pipeline_context.set("trained_model", model)
        
        # Set test data in the context
        pipeline_context.set("x_test", x)
        pipeline_context.set("y_test", y)
        
        # Create a model evaluation stage
        stage = ConfigDrivenModelEvaluationStage(
            config={
                "evaluation_type": "classification",
            }
        )
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that evaluation results were stored in the context
        assert pipeline_context.has("evaluation_results")
        results = pipeline_context.get("evaluation_results")
        assert isinstance(results, dict)
        assert "overall" in results


class TestModelSavingStage:
    """Tests for the model saving stages."""

    def test_config_driven_model_saving_stage(self, sample_data, pipeline_context):
        """Test ConfigDrivenModelSavingStage."""
        # Create a simple model and train it
        model = Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Create training data
        x = sample_data[["description", "service_life"]]
        y = sample_data[["category_name", "mcaa_system_category"]]
        
        # Train the model
        model.fit(x, y)
        pipeline_context.set("trained_model", model)
        
        # Create a temporary file for saving the model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            model_path = f.name
        
        try:
            # Create a model saving stage
            stage = ConfigDrivenModelSavingStage(
                config={
                    "saving_type": "pickle",
                }
            )
            
            # Create metadata
            metadata = {
                "created_at": "2025-03-08T23:00:00",
                "model_type": "random_forest",
                "features": ["description", "service_life"],
                "targets": ["category_name", "mcaa_system_category"],
            }
            
            # Execute the stage
            stage.save_model(model, model_path, metadata)
            
            # Check that the model was saved
            assert os.path.exists(model_path)
            assert os.path.getsize(model_path) > 0
            
            # Check that metadata was saved
            metadata_path = model_path.replace(".pkl", ".json")
            assert os.path.exists(metadata_path)
            assert os.path.getsize(metadata_path) > 0
        finally:
            # Clean up temporary files
            if os.path.exists(model_path):
                os.unlink(model_path)
            metadata_path = model_path.replace(".pkl", ".json")
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)


class TestPredictionStage:
    """Tests for the prediction stages."""

    def test_standard_prediction_stage(self, sample_data, pipeline_context):
        """Test StandardPredictionStage."""
        # Create a simple model and train it
        model = Pipeline([
            ("classifier", RandomForestClassifier(n_estimators=10, random_state=42))
        ])
        
        # Create training data
        x = sample_data[["description", "service_life"]]
        y = sample_data[["category_name", "mcaa_system_category"]]
        
        # Train the model
        model.fit(x, y)
        pipeline_context.set("trained_model", model)
        pipeline_context.set("data", sample_data)
        
        # Create a prediction stage
        stage = StandardPredictionStage()
        
        # Execute the stage
        stage.execute(pipeline_context)
        
        # Check that predictions were stored in the context
        assert pipeline_context.has("predictions")
        predictions = pipeline_context.get("predictions")
        assert isinstance(predictions, pd.DataFrame)
        assert len(predictions) == len(sample_data)