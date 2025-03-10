"""
Pipeline Stages Example

This example demonstrates how to use the pipeline stages to create a complete
machine learning pipeline for equipment classification.
"""

import os
import pandas as pd
from pathlib import Path

from nexusml.config.manager import ConfigurationManager
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages import (
    ConfigurableDataLoadingStage,
    ConfigDrivenValidationStage,
    SimpleFeatureEngineeringStage,
    RandomSplittingStage,
    ConfigDrivenModelBuildingStage,
    StandardModelTrainingStage,
    ClassificationEvaluationStage,
    ModelCardSavingStage,
    StandardPredictionStage,
)


def main():
    """
    Run the pipeline stages example.
    """
    # Create a pipeline context
    context = PipelineContext()
    context.start()

    # Create a configuration manager
    config_manager = ConfigurationManager()

    # Define the pipeline stages
    stages = [
        ConfigurableDataLoadingStage(
            config={"loader_type": "csv"},
            config_manager=config_manager,
        ),
        ConfigDrivenValidationStage(
            config={"config_name": "production_data_config"},
            config_manager=config_manager,
        ),
        SimpleFeatureEngineeringStage(),
        RandomSplittingStage(
            config={
                "test_size": 0.3,
                "random_state": 42,
            }
        ),
        # Use ConfigDrivenModelBuildingStage instead of RandomForestModelBuildingStage
        # to handle text data properly
        ConfigDrivenModelBuildingStage(
            config={
                "model_type": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
            }
        ),
        StandardModelTrainingStage(),
        ClassificationEvaluationStage(),
        ModelCardSavingStage(
            config={
                "model_name": "Equipment Classifier",
                "model_version": "1.0.0",
                "model_description": "A random forest model for classifying equipment based on descriptions.",
                "model_authors": ["NexusML Team"],
                "model_license": "Proprietary",
            }
        ),
        StandardPredictionStage(),
    ]

    try:
        # Get the data path
        data_path = os.path.join(
            Path(__file__).resolve().parent.parent, "data", "training_data", "production_training_data.csv"
        )

        # Execute each stage
        for stage in stages:
            print(f"Executing stage: {stage.get_name()}")
            
            # Skip stages that require data we don't have yet
            if stage.get_name() == "ModelCardSaving" and not context.has("trained_model"):
                print("Skipping model saving stage (no trained model)")
                continue
                
            if stage.get_name() == "StandardPrediction" and not context.has("trained_model"):
                print("Skipping prediction stage (no trained model)")
                continue

            # Execute the stage
            if stage.get_name() == "ConfigurableDataLoading":
                # Pass the data path to the data loading stage
                stage.execute(context, data_path=data_path)
                
                # Print the column names for debugging
                if context.has("data"):
                    data = context.get("data")
                    print("\nAvailable columns in the loaded data:")
                    for col in data.columns:
                        print(f"  - {col}")
                    print()
            elif stage.get_name() == "RandomSplitting":
                # Pass target columns to the data splitting stage
                stage.execute(
                    context,
                    target_columns=[
                        "category_name",
                        "uniformat_code",
                        "mcaa_system_category",
                        "System_Type_ID",
                        "Equip_Name_ID",
                    ],
                )
            elif stage.get_name() == "ModelCardSaving":
                # Pass the output path to the model saving stage
                output_path = os.path.join(
                    Path(__file__).resolve().parent.parent,
                    "output",
                    "models",
                    "equipment_classifier.pkl",
                )
                # Create the output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Get the model and metadata
                model = context.get("trained_model")
                metadata = {
                    "evaluation_results": context.get("evaluation_results", {}),
                    "created_at": pd.Timestamp.now().isoformat(),
                }
                
                # Execute the stage
                stage.save_model(model, output_path, metadata)
            else:
                # Execute the stage normally
                stage.execute(context)

        # Print the execution summary
        print("\nExecution Summary:")
        summary = context.get_execution_summary()
        for key, value in summary.items():
            if key == "component_execution_times":
                print(f"Component Execution Times:")
                for component, time in value.items():
                    print(f"  {component}: {time:.2f} seconds")
            elif key in ["accessed_keys", "modified_keys"]:
                print(f"{key}: {', '.join(value)}")
            else:
                print(f"{key}: {value}")

        # Print evaluation results if available
        if context.has("evaluation_results"):
            print("\nEvaluation Results:")
            evaluation_results = context.get("evaluation_results")
            if "overall" in evaluation_results:
                print("Overall Metrics:")
                for metric, value in evaluation_results["overall"].items():
                    print(f"  {metric}: {value}")

        # End the pipeline execution
        context.end("completed")
        print("\nPipeline execution completed successfully.")

    except Exception as e:
        # Log the error and end the pipeline execution
        context.log("ERROR", f"Pipeline execution failed: {str(e)}")
        context.end("failed")
        print(f"\nPipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()