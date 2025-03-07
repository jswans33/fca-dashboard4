#!/usr/bin/env python
"""
Basic Usage Example for NexusML

This example demonstrates the basic usage of NexusML for training a model and making predictions.
It covers:
- Loading data
- Training a model
- Making predictions
- Evaluating results
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry


def main():
    """Main function to demonstrate basic usage of NexusML."""
    print("NexusML Basic Usage Example")
    print("===========================")

    # Step 1: Set up the component registry, DI container, and factory
    print("\nStep 1: Setting up the component registry, DI container, and factory")
    registry = ComponentRegistry()
    container = DIContainer()
    factory = PipelineFactory(registry, container)
    context = PipelineContext()
    orchestrator = PipelineOrchestrator(factory, context)

    # Step 2: Get configuration
    print("\nStep 2: Getting configuration")
    config_provider = ConfigurationProvider()
    config = config_provider.config
    print(f"Configuration loaded from: {config_provider._load_config.__name__}")

    # Step 3: Train a model
    print("\nStep 3: Training a model")
    data_path = "examples/sample_data.xlsx"  # Path to sample data

    try:
        model, metrics = orchestrator.train_model(
            data_path=data_path,
            test_size=0.3,
            random_state=42,
            optimize_hyperparameters=True,
            output_dir="outputs/models",
            model_name="equipment_classifier_example",
        )

        print("Model training completed successfully")
        print(f"Model saved to: {orchestrator.context.get('model_path')}")
        print(f"Metadata saved to: {orchestrator.context.get('metadata_path')}")
        print("Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error training model: {e}")
        return

    # Step 4: Make predictions
    print("\nStep 4: Making predictions")

    # Create sample data for prediction
    import pandas as pd

    prediction_data = pd.DataFrame(
        {
            "equipment_tag": ["AHU-01", "CHW-01", "P-01"],
            "manufacturer": ["Trane", "Carrier", "Armstrong"],
            "model": ["M-1000", "C-2000", "A-3000"],
            "description": [
                "Air Handling Unit with cooling coil",
                "Centrifugal Chiller for HVAC system",
                "Centrifugal Pump for chilled water",
            ],
        }
    )

    try:
        predictions = orchestrator.predict(
            model=model,
            data=prediction_data,
            output_path="outputs/predictions_example.csv",
        )

        print("Predictions completed successfully")
        print(f"Predictions saved to: {orchestrator.context.get('output_path')}")
        print("Sample predictions:")

        # Loop through predictions using integer index
        for i in range(len(predictions)):
            if i >= 3:  # Only show first 3 predictions
                break

            print(f"  Item {i+1}:")
            print(f"    Equipment Tag: {prediction_data.iloc[i]['equipment_tag']}")
            print(f"    Description: {prediction_data.iloc[i]['description']}")

            # Access prediction values safely
            category = (
                predictions.iloc[i].get("category_name", "N/A")
                if "category_name" in predictions.columns
                else "N/A"
            )
            system_type = (
                predictions.iloc[i].get("mcaa_system_category", "N/A")
                if "mcaa_system_category" in predictions.columns
                else "N/A"
            )

            print(f"    Predicted Category: {category}")
            print(f"    Predicted System Type: {system_type}")

    except Exception as e:
        print(f"Error making predictions: {e}")
        return

    # Step 5: Evaluate the model
    print("\nStep 5: Evaluating the model")

    try:
        results = orchestrator.evaluate(
            model=model,
            data_path=data_path,
            output_path="outputs/evaluation_results_example.json",
        )

        print("Evaluation completed successfully")
        print(f"Evaluation results saved to: outputs/evaluation_results_example.json")
        print("Metrics:")
        for key, value in results["metrics"].items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return

    # Step 6: Get execution summary
    print("\nStep 6: Getting execution summary")

    summary = orchestrator.get_execution_summary()
    print("Execution summary:")
    print(f"  Status: {summary['status']}")
    print("  Component execution times:")
    for component, time in summary["component_execution_times"].items():
        print(f"    {component}: {time:.2f} seconds")
    print(
        f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds"
    )


if __name__ == "__main__":
    main()
