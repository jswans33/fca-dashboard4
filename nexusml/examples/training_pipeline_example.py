#!/usr/bin/env python
"""
Training Pipeline Example

This example demonstrates how to use the updated training pipeline entry point
with the pipeline orchestrator. It shows various configuration options and
error handling examples.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.src.utils.cli.training_args import TrainingArguments, setup_logging
from nexusml.train_model_pipeline_v2 import (
    create_orchestrator,
    make_sample_prediction_with_orchestrator,
    train_with_orchestrator,
)


def setup_example_data():
    """
    Set up example data for the training pipeline.

    Returns:
        Path to the example data file
    """
    # Create example data directory if it doesn't exist
    example_data_dir = Path("nexusml/examples/data")
    example_data_dir.mkdir(parents=True, exist_ok=True)

    # Path to the example data file
    data_path = example_data_dir / "example_training_data.csv"

    # Create example data if it doesn't exist
    if not data_path.exists():
        # Create a simple DataFrame with example data
        data = pd.DataFrame(
            {
                "equipment_tag": [f"EQ-{i:03d}" for i in range(1, 101)],
                "manufacturer": ["Trane", "Carrier", "York", "Daikin", "Lennox"] * 20,
                "model": [f"Model-{i:03d}" for i in range(1, 101)],
                "category_name": ["AHU", "Chiller", "Pump", "Fan", "Boiler"] * 20,
                "omniclass_code": [
                    "23-33 13 11",
                    "23-33 13 13",
                    "23-33 13 17",
                    "23-33 13 19",
                    "23-33 13 21",
                ]
                * 20,
                "uniformat_code": ["D3010", "D3020", "D3030", "D3040", "D3050"] * 20,
                "masterformat_code": [
                    "23 74 13",
                    "23 64 23",
                    "23 21 23",
                    "23 34 13",
                    "23 52 16",
                ]
                * 20,
                "mcaa_system_category": [
                    "HVAC",
                    "Plumbing",
                    "Mechanical",
                    "Controls",
                    "Electrical",
                ]
                * 20,
                "building_name": [
                    "Building A",
                    "Building B",
                    "Building C",
                    "Building D",
                    "Building E",
                ]
                * 20,
                "initial_cost": [10000 + i * 1000 for i in range(100)],
                "condition_score": [i % 5 + 1 for i in range(100)],
                "CategoryID": [i % 5 + 1 for i in range(100)],
                "OmniClassID": [i % 5 + 1 for i in range(100)],
                "UniFormatID": [i % 5 + 1 for i in range(100)],
                "MasterFormatID": [i % 5 + 1 for i in range(100)],
                "MCAAID": [i % 5 + 1 for i in range(100)],
                "LocationID": [i % 5 + 1 for i in range(100)],
                "description": [
                    "Air Handling Unit with cooling coil",
                    "Centrifugal Chiller for HVAC system",
                    "Centrifugal Pump for chilled water",
                    "Supply Fan for air distribution",
                    "Hot Water Boiler for heating",
                ]
                * 20,
                "service_life": [15, 20, 25, 30, 35] * 20,
            }
        )

        # Save the example data
        data.to_csv(data_path, index=False)
        print(f"Created example data file: {data_path}")

    return data_path


def basic_example(logger):
    """
    Basic example of using the training pipeline.

    Args:
        logger: Logger instance
    """
    logger.info("Running basic example")

    # Set up example data
    data_path = setup_example_data()

    # Create training arguments
    args = TrainingArguments(
        data_path=str(data_path),
        test_size=0.3,
        random_state=42,
        output_dir="nexusml/examples/output/models",
        model_name="example_model",
    )

    # Train the model
    try:
        model, metrics, _ = train_with_orchestrator(args, logger)

        logger.info("Model training completed successfully")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        return model

    except Exception as e:
        logger.error(f"Error in basic example: {e}")
        return None


def advanced_example(logger):
    """
    Advanced example with hyperparameter optimization and visualizations.

    Args:
        logger: Logger instance
    """
    logger.info("Running advanced example")

    # Set up example data
    data_path = setup_example_data()

    # Create training arguments
    args = TrainingArguments(
        data_path=str(data_path),
        test_size=0.2,
        random_state=123,
        optimize_hyperparameters=True,
        output_dir="nexusml/examples/output/models",
        model_name="advanced_model",
        visualize=True,
    )

    # Train the model
    try:
        model, metrics, viz_paths = train_with_orchestrator(args, logger)

        logger.info("Model training completed successfully")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        logger.info("Visualizations:")
        if viz_paths:
            for key, path in viz_paths.items():
                logger.info(f"  {key}: {path}")

        return model

    except Exception as e:
        logger.error(f"Error in advanced example: {e}")
        return None


def prediction_example(model, logger):
    """
    Example of making predictions with a trained model.

    Args:
        model: Trained model
        logger: Logger instance
    """
    logger.info("Running prediction example")

    # Create orchestrator
    orchestrator = create_orchestrator(logger)

    # Create sample data for prediction
    data = pd.DataFrame(
        {
            "equipment_tag": ["TEST-01", "TEST-02", "TEST-03"],
            "manufacturer": ["Trane", "Carrier", "York"],
            "model": ["Model-X", "Model-Y", "Model-Z"],
            "description": [
                "Air Handling Unit with cooling coil and variable frequency drive",
                "Water-cooled centrifugal chiller with high efficiency",
                "Vertical inline pump for condenser water system",
            ],
            "service_life": [20, 25, 30],
        }
    )

    # Make predictions
    try:
        predictions = orchestrator.predict(model=model, data=data)

        logger.info("Predictions:")
        for i, row in predictions.iterrows():
            logger.info(f"Item {i+1}:")
            logger.info(f"  Equipment Tag: {data.iloc[i]['equipment_tag']}")
            logger.info(f"  Description: {data.iloc[i]['description']}")
            for col in row.index:
                logger.info(f"  {col}: {row[col]}")

        return predictions

    except Exception as e:
        logger.error(f"Error in prediction example: {e}")
        return None


def error_handling_example(logger):
    """
    Example of error handling in the training pipeline.

    Args:
        logger: Logger instance
    """
    logger.info("Running error handling example")

    # Create training arguments with non-existent data path
    args = TrainingArguments(
        data_path="non_existent_file.csv",  # This will be caught by validation
        test_size=0.3,
        random_state=42,
    )

    # This should raise a ValueError
    try:
        train_with_orchestrator(args, logger)
    except ValueError as e:
        logger.info(f"Expected error caught: {e}")
        logger.info("Error handling worked correctly")


def feature_flags_example(logger):
    """
    Example of using feature flags for backward compatibility.

    Args:
        logger: Logger instance
    """
    logger.info("Running feature flags example")

    # Set up example data
    data_path = setup_example_data()

    # Example with orchestrator (new implementation)
    logger.info("Using orchestrator (new implementation)")
    args_new = TrainingArguments(
        data_path=str(data_path),
        use_orchestrator=True,
    )

    # Example with legacy implementation
    logger.info("Using legacy implementation")
    args_legacy = TrainingArguments(
        data_path=str(data_path),
        use_orchestrator=False,
    )

    # Note: In a real example, we would call the main function with these arguments
    # For this example, we'll just show the different configurations
    logger.info(f"New implementation args: {args_new.to_dict()}")
    logger.info(f"Legacy implementation args: {args_legacy.to_dict()}")


def main():
    """Main function to run the examples."""
    # Set up logging
    logger = setup_logging("INFO")
    logger.info("Starting training pipeline examples")

    try:
        # Run basic example
        logger.info("\n" + "=" * 50)
        logger.info("BASIC EXAMPLE")
        logger.info("=" * 50)
        model = basic_example(logger)

        if model:
            # Run prediction example
            logger.info("\n" + "=" * 50)
            logger.info("PREDICTION EXAMPLE")
            logger.info("=" * 50)
            prediction_example(model, logger)

        # Run advanced example
        logger.info("\n" + "=" * 50)
        logger.info("ADVANCED EXAMPLE")
        logger.info("=" * 50)
        advanced_example(logger)

        # Run error handling example
        logger.info("\n" + "=" * 50)
        logger.info("ERROR HANDLING EXAMPLE")
        logger.info("=" * 50)
        error_handling_example(logger)

        # Run feature flags example
        logger.info("\n" + "=" * 50)
        logger.info("FEATURE FLAGS EXAMPLE")
        logger.info("=" * 50)
        feature_flags_example(logger)

        logger.info("\n" + "=" * 50)
        logger.info("All examples completed")

    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()
