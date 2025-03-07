#!/usr/bin/env python
"""
Equipment Classification Prediction Script (V2)

This script loads a trained model and makes predictions on new equipment descriptions
using the pipeline orchestrator. It maintains backward compatibility with the original
prediction script through feature flags.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.cli.prediction_args import PredictionArgumentParser
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry


def create_orchestrator(logger: logging.Logger) -> PipelineOrchestrator:
    """
    Create a PipelineOrchestrator instance with all required components.

    Args:
        logger: Logger instance for logging messages.

    Returns:
        Configured PipelineOrchestrator instance.
    """
    # Create a component registry
    registry = ComponentRegistry()

    # Register default implementations
    # In a real application, we would register all implementations here
    # For now, we'll use the default implementations from the registry

    # Create a dependency injection container
    container = DIContainer()

    # Create a pipeline factory
    factory = PipelineFactory(registry, container)

    # Create a pipeline context
    context = PipelineContext()

    # Create a pipeline orchestrator
    orchestrator = PipelineOrchestrator(factory, context, logger)

    return orchestrator


def run_legacy_prediction(args, logger: logging.Logger) -> None:
    """
    Run the prediction using the legacy implementation.

    Args:
        args: Command-line arguments.
        logger: Logger instance for logging messages.

    Raises:
        SystemExit: If an error occurs during prediction.
    """
    logger.info("Using legacy prediction implementation")

    # Import the legacy implementation
    from nexusml.core.model import EquipmentClassifier

    try:
        # Load the model
        logger.info(f"Loading model from {args.model_path}")
        classifier = EquipmentClassifier()
        classifier.load_model(args.model_path)
        logger.info("Model loaded successfully")

        # Load input data
        logger.info(f"Loading input data from {args.input_file}")
        input_data = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(input_data)} items")
        logger.info(f"Input data columns: {input_data.columns.tolist()}")

        # Check if we have the fake data columns or the description column
        has_fake_data_columns = all(
            col in input_data.columns
            for col in ["equipment_tag", "manufacturer", "model"]
        )

        if (
            not has_fake_data_columns
            and args.description_column not in input_data.columns
        ):
            logger.error(
                f"Neither fake data columns nor description column '{args.description_column}' found in input file"
            )
            sys.exit(1)

        # Apply feature engineering to input data
        logger.info("Applying feature engineering to input data...")
        from nexusml.core.data_mapper import map_staging_to_model_input
        from nexusml.core.feature_engineering import GenericFeatureEngineer

        # First map staging data columns to model input format
        input_data = map_staging_to_model_input(input_data)
        logger.info(f"Columns after mapping: {input_data.columns.tolist()}")

        # Then apply feature engineering
        feature_engineer = GenericFeatureEngineer()
        processed_data = feature_engineer.transform(input_data)
        logger.info(
            f"Columns after feature engineering: {processed_data.columns.tolist()}"
        )

        # Make predictions
        logger.info("Making predictions...")
        results = []
        for i, row in processed_data.iterrows():
            # Get combined text from feature engineering
            if "combined_text" in processed_data.columns:
                description = row["combined_text"]
            else:
                # Fallback to creating a combined description
                description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"

            # Get service life from feature engineering
            service_life = 20.0
            if "service_life" in processed_data.columns:
                service_life = float(row.get("service_life", 20.0))
            elif "condition_score" in processed_data.columns:
                service_life = float(row.get("condition_score", 20.0))
            elif args.service_life_column in processed_data.columns:
                service_life = float(row.get(args.service_life_column, 20.0))

            # Get asset tag
            asset_tag = ""
            if "equipment_tag" in processed_data.columns:
                asset_tag = str(row.get("equipment_tag", ""))
            elif args.asset_tag_column in processed_data.columns:
                asset_tag = str(row.get(args.asset_tag_column, ""))

            # Debug the row data
            logger.info(f"Row data for prediction: {row.to_dict()}")

            # Make prediction with properly processed data
            # Instead of just passing the description, service_life, and asset_tag,
            # we need to pass the entire row to the model
            prediction = classifier.predict_from_row(row)

            # Add original description and service life to results
            prediction["original_description"] = description
            prediction["service_life"] = service_life
            if asset_tag:
                prediction["asset_tag"] = asset_tag

            results.append(prediction)

            # Print progress
            current_index = int(i)
            total_items = len(input_data)
            if (current_index + 1) % 10 == 0 or current_index == total_items - 1:
                logger.info(f"Processed {current_index + 1}/{total_items} items")

        # Convert results to DataFrame
        logger.info("Converting results to DataFrame")
        results_df = pd.DataFrame(results)

        # Create output directory if it doesn't exist
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        logger.info(f"Saving results to {args.output_file}")
        results_df.to_csv(args.output_file, index=False)

        # Print summary
        logger.info("Prediction completed successfully")
        logger.info(f"Results saved to: {args.output_file}")

        # Print sample of predictions
        logger.info("Sample predictions:")
        for idx, row in enumerate(results_df.head(3).to_dict("records")):
            logger.info(f"  Item {idx+1}:")
            logger.info(f"    Description: {row.get('original_description', 'N/A')}")
            logger.info(f"    Equipment Category: {row.get('category_name', 'N/A')}")
            logger.info(f"    System Type: {row.get('mcaa_system_category', 'N/A')}")
            logger.info(f"    Equipment Type: {row.get('Equipment_Type', 'N/A')}")

    except Exception as e:
        logger.error(f"Error in prediction script: {e}", exc_info=True)
        sys.exit(1)


def run_orchestrator_prediction(args, logger: logging.Logger) -> None:
    """
    Run the prediction using the pipeline orchestrator.

    Args:
        args: Command-line arguments.
        logger: Logger instance for logging messages.

    Raises:
        SystemExit: If an error occurs during prediction.
    """
    logger.info("Using pipeline orchestrator for prediction")

    try:
        # Create orchestrator
        orchestrator = create_orchestrator(logger)

        # Load input data
        logger.info(f"Loading input data from {args.input_file}")
        input_data = pd.read_csv(args.input_file)
        logger.info(f"Loaded {len(input_data)} items")
        logger.info(f"Input data columns: {input_data.columns.tolist()}")

        # Make predictions using the orchestrator
        predictions = orchestrator.predict(
            model_path=args.model_path,
            data=input_data,
            output_path=args.output_file,
            feature_config_path=args.feature_config_path,
            description_column=args.description_column,
            service_life_column=args.service_life_column,
            asset_tag_column=args.asset_tag_column,
        )

        # Print summary
        logger.info("Prediction completed successfully")
        logger.info(f"Results saved to: {args.output_file}")

        # Get execution summary
        summary = orchestrator.get_execution_summary()
        logger.info("Execution summary:")
        logger.info(f"  Status: {summary['status']}")
        logger.info("  Component execution times:")
        for component, time in summary["component_execution_times"].items():
            logger.info(f"    {component}: {time:.2f} seconds")
        logger.info(
            f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds"
        )

        # Print sample of predictions
        logger.info("Sample predictions:")
        for idx, row in enumerate(predictions.head(3).iterrows()):
            logger.info(f"  Item {idx+1}:")
            if "original_description" in row[1]:
                logger.info(
                    f"    Description: {row[1].get('original_description', 'N/A')}"
                )
            elif "combined_text" in row[1]:
                logger.info(f"    Description: {row[1].get('combined_text', 'N/A')}")
            logger.info(f"    Equipment Category: {row[1].get('category_name', 'N/A')}")
            logger.info(f"    System Type: {row[1].get('mcaa_system_category', 'N/A')}")
            logger.info(f"    Equipment Type: {row[1].get('Equipment_Type', 'N/A')}")

    except Exception as e:
        logger.error(f"Error in orchestrator prediction: {e}", exc_info=True)
        sys.exit(1)


def main() -> None:
    """
    Main function to run the prediction script.

    This function parses command-line arguments, sets up logging, and runs
    the appropriate prediction implementation based on the feature flag.
    """
    # Parse command-line arguments
    parser = PredictionArgumentParser()
    args = parser.parse_args()

    # Set up logging
    logger = parser.setup_logging(args)
    logger.info("Starting equipment classification prediction (V2)")

    try:
        # Validate arguments
        parser.validate_args(args)

        # Run the appropriate prediction implementation based on the feature flag
        if args.use_orchestrator:
            run_orchestrator_prediction(args, logger)
        else:
            run_legacy_prediction(args, logger)

    except Exception as e:
        logger.error(f"Error in prediction script: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
