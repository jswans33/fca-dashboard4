#!/usr/bin/env python
"""
Prediction Pipeline Example

This example demonstrates how to use the updated prediction pipeline entry point
with both the legacy implementation and the new orchestrator-based implementation.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.cli.prediction_args import PredictionArgumentParser
from nexusml.predict_v2 import create_orchestrator, run_legacy_prediction, run_orchestrator_prediction


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "prediction_example.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("prediction_pipeline_example")


def create_sample_data():
    """
    Create sample data for prediction.

    Returns:
        Path to the sample data file.
    """
    logger = logging.getLogger("prediction_pipeline_example")
    logger.info("Creating sample data for prediction")

    # Create sample data directory if it doesn't exist
    sample_dir = Path("examples/data")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create sample data
    sample_data = pd.DataFrame(
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

    # Save sample data
    sample_path = sample_dir / "sample_prediction_data.csv"
    sample_data.to_csv(sample_path, index=False)
    logger.info(f"Sample data saved to {sample_path}")

    return sample_path


def legacy_prediction_example(logger):
    """
    Example of using the legacy prediction implementation.

    Args:
        logger: Logger instance for logging messages.
    """
    logger.info("Legacy Prediction Example")

    # Create sample data
    sample_path = create_sample_data()

    # Create output directory if it doesn't exist
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up arguments
    args = type(
        "Args",
        (),
        {
            "model_path": "outputs/models/equipment_classifier_latest.pkl",
            "input_file": str(sample_path),
            "output_file": str(output_dir / "legacy_prediction_results.csv"),
            "log_level": "INFO",
            "description_column": "description",
            "service_life_column": "service_life",
            "asset_tag_column": "equipment_tag",
            "feature_config_path": None,
            "use_orchestrator": False,
        },
    )()

    # Check if model exists, if not, skip this example
    if not Path(args.model_path).exists():
        logger.warning(f"Model file not found: {args.model_path}")
        logger.warning("Skipping legacy prediction example")
        return

    try:
        # Run legacy prediction
        logger.info(f"Running legacy prediction with input file: {args.input_file}")
        run_legacy_prediction(args, logger)

        # Check if output file was created
        if Path(args.output_file).exists():
            logger.info(f"Legacy prediction results saved to: {args.output_file}")

            # Load and display results
            results = pd.read_csv(args.output_file)
            logger.info(f"Legacy prediction results ({len(results)} rows):")
            for i, row in results.head(3).iterrows():
                logger.info(f"  Item {i+1}:")
                logger.info(f"    Description: {row.get('original_description', 'N/A')}")
                logger.info(f"    Equipment Category: {row.get('category_name', 'N/A')}")
                logger.info(f"    System Type: {row.get('mcaa_system_category', 'N/A')}")
        else:
            logger.warning(f"Output file not created: {args.output_file}")

    except Exception as e:
        logger.error(f"Error in legacy prediction example: {e}", exc_info=True)


def orchestrator_prediction_example(logger):
    """
    Example of using the orchestrator-based prediction implementation.

    Args:
        logger: Logger instance for logging messages.
    """
    logger.info("Orchestrator Prediction Example")

    # Create sample data
    sample_path = create_sample_data()

    # Create output directory if it doesn't exist
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up arguments
    args = type(
        "Args",
        (),
        {
            "model_path": "outputs/models/equipment_classifier_latest.pkl",
            "input_file": str(sample_path),
            "output_file": str(output_dir / "orchestrator_prediction_results.csv"),
            "log_level": "INFO",
            "description_column": "description",
            "service_life_column": "service_life",
            "asset_tag_column": "equipment_tag",
            "feature_config_path": "nexusml/config/feature_config.yml",
            "use_orchestrator": True,
        },
    )()

    # Check if model exists, if not, skip this example
    if not Path(args.model_path).exists():
        logger.warning(f"Model file not found: {args.model_path}")
        logger.warning("Skipping orchestrator prediction example")
        return

    # Check if feature config exists, if not, skip this example
    if args.feature_config_path and not Path(args.feature_config_path).exists():
        logger.warning(f"Feature config file not found: {args.feature_config_path}")
        logger.warning("Skipping orchestrator prediction example")
        return

    try:
        # Run orchestrator prediction
        logger.info(f"Running orchestrator prediction with input file: {args.input_file}")
        run_orchestrator_prediction(args, logger)

        # Check if output file was created
        if Path(args.output_file).exists():
            logger.info(f"Orchestrator prediction results saved to: {args.output_file}")

            # Load and display results
            results = pd.read_csv(args.output_file)
            logger.info(f"Orchestrator prediction results ({len(results)} rows):")
            for i, row in results.head(3).iterrows():
                logger.info(f"  Item {i+1}:")
                if "original_description" in row:
                    logger.info(f"    Description: {row.get('original_description', 'N/A')}")
                elif "combined_text" in row:
                    logger.info(f"    Description: {row.get('combined_text', 'N/A')}")
                logger.info(f"    Equipment Category: {row.get('category_name', 'N/A')}")
                logger.info(f"    System Type: {row.get('mcaa_system_category', 'N/A')}")
        else:
            logger.warning(f"Output file not created: {args.output_file}")

    except Exception as e:
        logger.error(f"Error in orchestrator prediction example: {e}", exc_info=True)


def command_line_example(logger):
    """
    Example of using the prediction pipeline from the command line.

    Args:
        logger: Logger instance for logging messages.
    """
    logger.info("Command Line Example")

    # Create sample data
    sample_path = create_sample_data()

    # Create output directory if it doesn't exist
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Legacy command
    legacy_cmd = (
        f"python -m nexusml.predict_v2 "
        f"--input-file={sample_path} "
        f"--output-file={output_dir / 'cli_legacy_results.csv'} "
        f"--description-column=description "
        f"--service-life-column=service_life "
        f"--asset-tag-column=equipment_tag"
    )

    # Orchestrator command
    orchestrator_cmd = (
        f"python -m nexusml.predict_v2 "
        f"--input-file={sample_path} "
        f"--output-file={output_dir / 'cli_orchestrator_results.csv'} "
        f"--description-column=description "
        f"--service-life-column=service_life "
        f"--asset-tag-column=equipment_tag "
        f"--use-orchestrator"
    )

    logger.info("To run the prediction pipeline from the command line:")
    logger.info(f"  Legacy mode: {legacy_cmd}")
    logger.info(f"  Orchestrator mode: {orchestrator_cmd}")


def error_handling_example(logger):
    """
    Example of error handling in the prediction pipeline.

    Args:
        logger: Logger instance for logging messages.
    """
    logger.info("Error Handling Example")

    # Set up arguments with nonexistent files
    args = type(
        "Args",
        (),
        {
            "model_path": "nonexistent_model.pkl",
            "input_file": "nonexistent_input.csv",
            "output_file": "nonexistent_output.csv",
            "log_level": "INFO",
            "description_column": "description",
            "service_life_column": "service_life",
            "asset_tag_column": "equipment_tag",
            "feature_config_path": None,
            "use_orchestrator": False,
        },
    )()

    try:
        # Try to validate arguments (should fail)
        parser = PredictionArgumentParser()
        parser.validate_args(args)
    except ValueError as e:
        logger.info(f"Expected error caught: {e}")
        logger.info("Error handling worked correctly")


def main():
    """Main function to run the example."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Prediction Pipeline Example")

    # Run examples
    legacy_prediction_example(logger)
    orchestrator_prediction_example(logger)
    command_line_example(logger)
    error_handling_example(logger)

    logger.info("Prediction Pipeline Example completed")


if __name__ == "__main__":
    main()
