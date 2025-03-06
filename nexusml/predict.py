#!/usr/bin/env python
"""
Equipment Classification Prediction Script

This script loads a trained model and makes predictions on new equipment descriptions.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.model import EquipmentClassifier


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "prediction.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("equipment_prediction")


def main():
    """Main function to run the prediction script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Make equipment classification predictions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/models/equipment_classifier_latest.pkl",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input CSV file with equipment descriptions",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="prediction_results.csv",
        help="Path to save the prediction results",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--description-column",
        type=str,
        default="Description",
        help="Column name containing equipment descriptions",
    )
    parser.add_argument(
        "--service-life-column",
        type=str,
        default="Service Life",
        help="Column name containing service life values",
    )
    parser.add_argument(
        "--asset-tag-column",
        type=str,
        default="Asset Tag",
        help="Column name containing asset tags",
    )
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting equipment classification prediction")

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

        # Make predictions
        logger.info("Making predictions...")
        results = []
        for i, row in input_data.iterrows():
            # For fake data, use the combined columns for description
            if (
                "equipment_tag" in input_data.columns
                and "manufacturer" in input_data.columns
                and "model" in input_data.columns
            ):
                # Create a combined description from multiple columns
                description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"
            else:
                # Use the specified description column
                description = str(row[args.description_column])

            # Get service life if column exists, otherwise use default
            service_life = 20.0
            if "condition_score" in input_data.columns:
                service_life = float(row.get("condition_score", 20.0))
            elif args.service_life_column in input_data.columns:
                service_life = float(row.get(args.service_life_column, 20.0))

            # Get asset tag if column exists, otherwise use empty string
            asset_tag = ""
            if "equipment_tag" in input_data.columns:
                asset_tag = str(row.get("equipment_tag", ""))
            elif args.asset_tag_column in input_data.columns:
                asset_tag = str(row.get(args.asset_tag_column, ""))

            # Make prediction
            prediction = classifier.predict(description, service_life, asset_tag)

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


if __name__ == "__main__":
    main()
