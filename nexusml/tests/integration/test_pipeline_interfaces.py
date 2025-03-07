"""
Integration test for the pipeline interfaces.

This test verifies that the entire pipeline works correctly using the new interfaces.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from nexusml.core.pipeline.adapters import (
    LegacyDataLoaderAdapter,
    LegacyDataPreprocessorAdapter,
    LegacyFeatureEngineerAdapter,
    LegacyModelBuilderAdapter,
    LegacyModelEvaluatorAdapter,
    LegacyModelSerializerAdapter,
    LegacyModelTrainerAdapter,
    LegacyPredictorAdapter,
)
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


def test_pipeline_with_interfaces(
    sample_data_path, sample_description, sample_service_life, tmp_path
):
    """
    Test the full pipeline using the new interfaces.

    This test verifies that the pipeline still works correctly with the new interfaces.
    """
    # Create instances of the adapter classes
    data_loader = LegacyDataLoaderAdapter()
    data_preprocessor = LegacyDataPreprocessorAdapter()
    feature_engineer = LegacyFeatureEngineerAdapter()
    model_builder = LegacyModelBuilderAdapter()
    model_trainer = LegacyModelTrainerAdapter()
    model_evaluator = LegacyModelEvaluatorAdapter()
    model_serializer = LegacyModelSerializerAdapter()
    predictor = LegacyPredictorAdapter()

    # Load and preprocess data
    df = data_loader.load_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Preprocess data
    df = data_preprocessor.preprocess(df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Engineer features
    df = feature_engineer.engineer_features(df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "combined_features" in df.columns
    assert "service_life" in df.columns

    # Prepare training data
    X = pd.DataFrame(
        {
            "combined_features": df["combined_features"],
            "service_life": df["service_life"],
        }
    )

    y = df[
        [
            "Equipment_Category",
            "Uniformat_Class",
            "System_Type",
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # Build model
    model = model_builder.build_model()
    assert isinstance(model, Pipeline)

    # Train model
    trained_model = model_trainer.train(model, X, y)
    assert isinstance(trained_model, Pipeline)

    # Evaluate model
    metrics = model_evaluator.evaluate(trained_model, X, y)
    assert isinstance(metrics, dict)

    # Save model
    model_path = str(tmp_path / "model.pkl")
    model_serializer.save_model(trained_model, model_path)
    assert os.path.exists(model_path)

    # Load model
    loaded_model = model_serializer.load_model(model_path)
    assert isinstance(loaded_model, Pipeline)

    # Make a prediction
    # Create a test input similar to what would be created from sample_description and sample_service_life
    test_input = pd.DataFrame(
        {
            "combined_features": [sample_description],
            "service_life": [sample_service_life],
        }
    )

    prediction = predictor.predict(loaded_model, test_input)
    assert isinstance(prediction, pd.DataFrame)
    assert not prediction.empty

    print("Pipeline with interfaces test completed successfully!")


@pytest.mark.skip(
    reason="This test is a simplified version that doesn't require a full pipeline run"
)
def test_simplified_pipeline_with_interfaces(sample_data_path):
    """
    A simplified test of the pipeline using the new interfaces.

    This test only verifies the data loading, preprocessing, and feature engineering steps.
    """
    # Create instances of the adapter classes
    data_loader = LegacyDataLoaderAdapter()
    data_preprocessor = LegacyDataPreprocessorAdapter()
    feature_engineer = LegacyFeatureEngineerAdapter()

    # Load and preprocess data
    df = data_loader.load_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Preprocess data
    df = data_preprocessor.preprocess(df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    # Engineer features
    df = feature_engineer.engineer_features(df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "combined_features" in df.columns
    assert "service_life" in df.columns

    print("Simplified pipeline with interfaces test completed successfully!")
