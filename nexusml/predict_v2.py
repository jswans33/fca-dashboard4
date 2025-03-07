#!/usr/bin/env python
"""
Equipment Classification Prediction Script (Version 2)

This script loads a trained model and makes predictions on new equipment descriptions
using the pipeline orchestrator. It maintains backward compatibility with the original
prediction script while adding new capabilities.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.cli.prediction_args import PredictionArgumentParser
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.container import DIContainer
from nexusml.core.model import EquipmentClassifier  # For backward compatibility
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry


def create_orchestrator(logger: logging.Logger) -> PipelineOrchestrator:
    """
    Create a PipelineOrchestrator instance.

    Args:
        logger: Logger instance for logging messages.

    Returns:
        Configured PipelineOrchestrator instance.
    """
    # Create a component registry
    registry = ComponentRegistry()

    # Register default implementations
    # This would typically be done by the dependency injection system
    # but we're creating it manually here for simplicity
    # Import default implementations
    # In a real application, these would be imported from their respective modules
    from nexusml.core.pipeline.components.data_loader import StandardDataLoader
    from nexusml.core.pipeline.components.data_preprocessor import StandardPreprocessor
    from nexusml.core.pipeline.components.feature_engineer import (
        StandardFeatureEngineer,
    )
    from nexusml.core.pipeline.components.model_builder import RandomForestModelBuilder
    from nexusml.core.pipeline.components.model_evaluator import StandardModelEvaluator
    from nexusml.core.pipeline.components.model_serializer import PickleModelSerializer
    from nexusml.core.pipeline.components.model_trainer import StandardModelTrainer
    from nexusml.core.pipeline.components.predictor import StandardPredictor
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

    # Register the components
    registry.register(DataLoader, "standard", StandardDataLoader)
    registry.register(DataPreprocessor, "standard", StandardPreprocessor)
    registry.register(FeatureEngineer, "standard", StandardFeatureEngineer)
    registry.register(ModelBuilder, "standard", RandomForestModelBuilder)
    registry.register(ModelTrainer, "standard", StandardModelTrainer)
    registry.register(ModelEvaluator, "standard", StandardModelEvaluator)
    registry.register(ModelSerializer, "standard", PickleModelSerializer)
    registry.register(Predictor, "standard", StandardPredictor)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "standard")
    registry.set_default_implementation(DataPreprocessor, "standard")
    registry.set_default_implementation(FeatureEngineer, "standard")
    registry.set_default_implementation(ModelBuilder, "standard")
    registry.set_default_implementation(ModelTrainer, "standard")
    registry.set_default_implementation(ModelEvaluator, "standard")
    registry.set_default_implementation(ModelSerializer, "standard")
    registry.set_default_implementation(Predictor, "standard")

    # Create a dependency injection container
    container = DIContainer()

    # Create a pipeline factory
    factory = PipelineFactory(registry, container)

    # Create a pipeline context
    context = PipelineContext()

    # Create a pipeline orchestrator
    orchestrator = PipelineOrchestrator(factory, context, logger)

    return orchestrator


def predict_with_orchestrator(
    args: Dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    """
    Make predictions using the pipeline orchestrator.

    Args:
        args: Dictionary of command-line arguments.
        logger: Logger instance for logging messages.

    Returns:
        DataFrame containing the prediction results.

    Raises:
        Exception: If an error occurs during prediction.
    """
    logger.info("Starting prediction using pipeline orchestrator")

    try:
        # Create orchestrator
        orchestrator = create_orchestrator(logger)

        # Make predictions
        predictions = orchestrator.predict(
            model_path=args["model_path"],
            data_path=args["input_file"],
            output_path=args["output_file"],
            feature_config_path=args.get("feature_config_path"),
        )

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

        return predictions

    except Exception as e:
        logger.error(f"Error in prediction with orchestrator: {e}", exc_info=True)
        raise


def predict_legacy(args: Dict[str, Any], logger: logging.Logger) -> pd.DataFrame:
    """
    Make predictions using the legacy approach for backward compatibility.

    Args:
        args: Dictionary of command-line arguments.
        logger: Logger instance for logging messages.

    Returns:
        DataFrame containing the prediction results.

    Raises:
        Exception: If an error occurs during prediction.
    """
    logger.info("Starting prediction using legacy approach")

    try:
        # Load the model
        logger.info(f"Loading model from {args['model_path']}")
        classifier = EquipmentClassifier()
        classifier.load_model(args["model_path"])
        logger.info("Model loaded successfully")

        # Load input data
        logger.info(f"Loading input data from {args['input_file']}")
        input_data = pd.read_csv(args["input_file"])
        logger.info(f"Loaded {len(input_data)} items")
        logger.info(f"Input data columns: {input_data.columns.tolist()}")

        # Check if we have the fake data columns or the description column
        has_fake_data_columns = all(
            col in input_data.columns
            for col in ["equipment_tag", "manufacturer", "model"]
        )

        if (
            not has_fake_data_columns
            and args["description_column"] not in input_data.columns
        ):
            logger.error(
                f"Neither fake data columns nor description column '{args['description_column']}' found in input file"
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
            elif args["service_life_column"] in processed_data.columns:
                service_life = float(row.get(args["service_life_column"], 20.0))

            # Get asset tag
            asset_tag = ""
            if "equipment_tag" in processed_data.columns:
                asset_tag = str(row.get("equipment_tag", ""))
            elif args["asset_tag_column"] in processed_data.columns:
                asset_tag = str(row.get(args["asset_tag_column"], ""))

            # Debug the row data
            logger.debug(f"Row data for prediction: {row.to_dict()}")

            # Make prediction with properly processed data
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
        output_path = Path(args["output_file"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        logger.info(f"Saving results to {args['output_file']}")
        results_df.to_csv(args["output_file"], index=False)

        return results_df

    except Exception as e:
        logger.error(f"Error in legacy prediction: {e}", exc_info=True)
        raise


def main() -> None:
    """Main function to run the prediction script."""
    # Parse command-line arguments
    parser = PredictionArgumentParser()
    args = parser.parse_args()
    args_dict = vars(args)

    # Set up logging
    logger = parser.setup_logging(args)
    logger.info("Starting equipment classification prediction")

    try:
        # Validate arguments
        parser.validate_args(args)

        # Determine whether to use the orchestrator or legacy approach
        if args.use_orchestrator:
            logger.info("Using pipeline orchestrator for prediction")
            results_df = predict_with_orchestrator(args_dict, logger)
        else:
            logger.info("Using legacy approach for prediction")
            results_df = predict_legacy(args_dict, logger)

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


if __name__ == "__main__":
    main()
