"""
Feature Engineering Example

This example demonstrates how to use the new config-driven feature engineering approach.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add the parent directory to the path so we can import nexusml
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.feature_engineering import GenericFeatureEngineer
from nexusml.core.model import predict_with_enhanced_model, train_enhanced_model


def demonstrate_generic_feature_engineering():
    """
    Demonstrate how to use the GenericFeatureEngineer class directly.
    """
    print("\n=== Demonstrating GenericFeatureEngineer ===\n")

    # Load sample data
    print("Loading sample data...")
    df = load_and_preprocess_data()

    # Print original columns
    print("\nOriginal columns:")
    print(df.columns.tolist())

    # Apply generic feature engineering
    print("\nApplying generic feature engineering...")
    engineer = GenericFeatureEngineer()
    df_transformed = engineer.transform(df)

    # Print new columns
    print("\nNew columns after transformation:")
    print(df_transformed.columns.tolist())

    # Print sample of combined text
    print("\nSample of combined_text:")
    print(df_transformed["combined_text"].iloc[0][:200] + "...")

    # Print sample of hierarchical categories
    print("\nSample of hierarchical categories:")
    print(f"Equipment_Type: {df_transformed['Equipment_Type'].iloc[0]}")
    print(f"System_Subtype: {df_transformed['System_Subtype'].iloc[0]}")

    return df_transformed


def demonstrate_model_training_with_config():
    """
    Demonstrate how to train a model using the config-driven approach.
    """
    print("\n=== Demonstrating Model Training with Config ===\n")

    # Train model with config-driven feature engineering
    print("Training model with config-driven feature engineering...")
    model, df = train_enhanced_model(
        sampling_strategy="direct"
    )  # Use direct to speed up example

    # Make a prediction
    print("\nMaking a prediction...")
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = 20.0
    prediction = predict_with_enhanced_model(model, description, service_life)

    print("\nPrediction results:")
    for key, value in prediction.items():
        print(f"{key}: {value}")

    return model, df


if __name__ == "__main__":
    # Demonstrate using GenericFeatureEngineer directly
    df_transformed = demonstrate_generic_feature_engineering()

    # Demonstrate training a model with config-driven feature engineering
    model, df = demonstrate_model_training_with_config()

    print("\nExample completed successfully!")
