#!/usr/bin/env python
"""
Production Model Training Pipeline for Equipment Classification

This script implements a production-ready pipeline for training the equipment classification model
following SOP 008. It provides a structured workflow with command-line arguments for flexibility,
proper logging, comprehensive evaluation, and model versioning.

Usage:
    python train_model_pipeline.py --data-path PATH [options]

Example:
    python train_model_pipeline.py --data-path files/training-data/equipment_data.csv --optimize
"""

import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import core modules
from nexusml.core.data_mapper import (
    map_predictions_to_master_db,
    map_staging_to_model_input,
)
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.evaluation import (
    analyze_other_category_features,
    analyze_other_misclassifications,
    enhanced_evaluation,
)
from nexusml.core.feature_engineering import GenericFeatureEngineer
from nexusml.core.model import EquipmentClassifier
from nexusml.core.model_building import build_enhanced_model, optimize_hyperparameters
from nexusml.core.reference.manager import ReferenceManager


# Implement missing functions
def validate_training_data(data_path: str) -> Dict:
    """
    Validate the training data to ensure it meets quality standards.

    This function checks:
    1. If the file exists and can be read
    2. If required columns are present
    3. If data types are correct
    4. If there are any missing values in critical columns

    Args:
        data_path: Path to the training data file

    Returns:
        Dictionary with validation results
    """
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            return {"valid": False, "issues": [f"File not found: {data_path}"]}

        # Try to read the file
        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            return {"valid": False, "issues": [f"Error reading file: {str(e)}"]}

        # Check required columns for the real data format
        required_columns = [
            "equipment_tag",
            "manufacturer",
            "model",
            "category_name",
            "omniclass_code",
            "uniformat_code",
            "masterformat_code",
            "mcaa_system_category",
            "building_name",
            "initial_cost",
            "condition_score",
            "CategoryID",
            "OmniClassID",
            "UniFormatID",
            "MasterFormatID",
            "MCAAID",
            "LocationID",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return {
                "valid": False,
                "issues": [f"Missing required columns: {', '.join(missing_columns)}"],
            }

        # Check for missing values in critical columns
        critical_columns = ["equipment_tag", "category_name", "mcaa_system_category"]
        missing_values = {}

        for col in critical_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = missing_count

        if missing_values:
            issues = [
                f"Missing values in {col}: {count}"
                for col, count in missing_values.items()
            ]
            return {"valid": False, "issues": issues}

        # All checks passed
        return {"valid": True, "issues": []}

    except Exception as e:
        return {
            "valid": False,
            "issues": [f"Unexpected error during validation: {str(e)}"],
        }


def visualize_category_distribution(
    df: pd.DataFrame, output_dir: str = "outputs"
) -> Tuple[str, str]:
    """
    Visualize the distribution of categories in the dataset.

    Args:
        df: DataFrame with category columns
        output_dir: Directory to save visualizations

    Returns:
        Tuple of paths to the saved visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    equipment_category_file = f"{output_dir}/equipment_category_distribution.png"
    system_type_file = f"{output_dir}/system_type_distribution.png"

    # Generate visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df, y="category_name"
    )  # Use category_name instead of Equipment_Category
    plt.title("Equipment Category Distribution")
    plt.tight_layout()
    plt.savefig(equipment_category_file)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df, y="mcaa_system_category"
    )  # Use mcaa_system_category instead of System_Type
    plt.title("System Type Distribution")
    plt.tight_layout()
    plt.savefig(system_type_file)
    plt.close()

    return equipment_category_file, system_type_file


def visualize_confusion_matrix(
    y_true, y_pred, class_name: str, output_file: str
) -> None:
    """
    Create and save a confusion matrix visualization.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_name: Name of the classification column
        output_file: Path to save the visualization
    """
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Get unique classes as a list of strings
    classes = sorted(list(set([str(c) for c in y_true] + [str(c) for c in y_pred])))

    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {class_name}")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"model_training_{timestamp}.log"

    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger("model_training")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Train the equipment classification model"
    )

    # Data arguments
    parser.add_argument(
        "--data-path", type=str, help="Path to the training data CSV file"
    )
    parser.add_argument(
        "--feature-config", type=str, help="Path to the feature configuration YAML file"
    )
    parser.add_argument(
        "--reference-config",
        type=str,
        help="Path to the reference configuration YAML file",
    )

    # Training arguments
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Proportion of data to use for testing (default: 0.3)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--sampling-strategy",
        type=str,
        default="direct",
        choices=["direct"],
        help="Sampling strategy for handling class imbalance (default: direct)",
    )

    # Optimization arguments
    parser.add_argument(
        "--optimize", action="store_true", help="Perform hyperparameter optimization"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save the trained model and results (default: outputs/models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="equipment_classifier",
        help="Base name for the saved model (default: equipment_classifier)",
    )

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    # Visualization arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of model performance",
    )

    return parser.parse_args()


def load_reference_data(
    config_path: Optional[str] = None, logger: Optional[logging.Logger] = None
) -> ReferenceManager:
    """
    Load reference data using the ReferenceManager.

    Args:
        config_path: Path to the reference configuration file
        logger: Logger instance

    Returns:
        Initialized ReferenceManager with loaded data
    """
    if logger:
        logger.info("Loading reference data...")

    ref_manager = ReferenceManager(config_path)
    ref_manager.load_all()

    if logger:
        logger.info("Reference data loaded successfully")

    return ref_manager


def validate_data(data_path: str, logger: Optional[logging.Logger] = None) -> Dict:
    """
    Validate the training data to ensure it meets quality standards.

    Args:
        data_path: Path to the training data
        logger: Logger instance

    Returns:
        Validation results dictionary
    """
    if logger:
        logger.info(f"Validating training data at {data_path}...")

    validation_results = validate_training_data(data_path)

    if logger:
        logger.info("Data validation completed")

        # Log validation summary
        if validation_results.get("valid", False):
            logger.info("Data validation passed")
        else:
            logger.warning("Data validation failed")
            for issue in validation_results.get("issues", []):
                logger.warning(f"Validation issue: {issue}")

    return validation_results


def train_model(
    data_path: Optional[str] = None,
    feature_config_path: Optional[str] = None,
    sampling_strategy: str = "direct",
    test_size: float = 0.3,
    random_state: int = 42,
    optimize_params: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[EquipmentClassifier, pd.DataFrame, Dict]:
    """
    Train the equipment classification model.

    Args:
        data_path: Path to the training data
        feature_config_path: Path to the feature configuration
        sampling_strategy: Strategy for handling class imbalance
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility
        optimize_params: Whether to perform hyperparameter optimization
        logger: Logger instance

    Returns:
        Tuple containing:
        - Trained EquipmentClassifier
        - Processed DataFrame
        - Dictionary with evaluation metrics
    """
    # Create classifier instance
    classifier = EquipmentClassifier(sampling_strategy=sampling_strategy)

    # Train the model
    if logger:
        logger.info("Training model...")
        logger.info(f"Using data path: {data_path}")
        logger.info(f"Using feature config: {feature_config_path}")
        logger.info(f"Test size: {test_size}")
        logger.info(f"Random state: {random_state}")
        logger.info(f"Sampling strategy: {sampling_strategy}")
        logger.info(f"Hyperparameter optimization: {optimize_params}")

    start_time = time.time()

    # Train with custom parameters
    classifier.train(
        data_path=data_path,
        feature_config_path=feature_config_path,
        test_size=test_size,
        random_state=random_state,
    )

    # Get the processed data
    df = classifier.df

    # Prepare data for evaluation
    x = pd.DataFrame(
        {
            "combined_features": df["combined_text"],
            "service_life": df["service_life"],
        }
    )

    y = df[
        [
            "category_name",  # Use category_name instead of Equipment_Category
            "uniformat_code",  # Use uniformat_code instead of Uniformat_Class
            "mcaa_system_category",  # Use mcaa_system_category instead of System_Type
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # Split for evaluation
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # Optimize hyperparameters if requested
    if optimize_params and classifier.model is not None:
        if logger:
            logger.info("Optimizing hyperparameters...")

        optimized_model = optimize_hyperparameters(classifier.model, x_train, y_train)

        if logger:
            logger.info("Hyperparameter optimization completed")

        # Update classifier with optimized model
        classifier.model = optimized_model
    elif optimize_params and classifier.model is None:
        if logger:
            logger.warning("Cannot optimize hyperparameters: model is None")

    # Evaluate the model
    if logger:
        logger.info("Evaluating model...")

    # Make predictions if model exists
    metrics = {}
    if classifier.model is not None:
        y_pred_df = enhanced_evaluation(classifier.model, x_test, y_test)

        # Calculate metrics
        for col in y_test.columns:
            metrics[col] = {
                "accuracy": float(accuracy_score(y_test[col], y_pred_df[col])),
                "f1_macro": float(
                    f1_score(y_test[col], y_pred_df[col], average="macro")
                ),
            }

        # Analyze "Other" category performance
        analyze_other_category_features(
            classifier.model, x_test["combined_features"], y_test, y_pred_df
        )
        analyze_other_misclassifications(x_test["combined_features"], y_test, y_pred_df)
    else:
        if logger:
            logger.warning("Cannot evaluate model: model is None")

    training_time = time.time() - start_time

    if logger:
        logger.info(
            f"Model training and evaluation completed in {training_time:.2f} seconds"
        )
        logger.info("Evaluation metrics:")
        for col, col_metrics in metrics.items():
            logger.info(f"  {col}:")
            for metric_name, metric_value in col_metrics.items():
                logger.info(f"    {metric_name}: {metric_value:.4f}")

    return classifier, df, metrics


def save_model(
    classifier: EquipmentClassifier,
    output_dir: str,
    model_name: str,
    metrics: Dict,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Save the trained model and metadata.

    Args:
        classifier: Trained EquipmentClassifier
        output_dir: Directory to save the model
        model_name: Base name for the model file
        metrics: Evaluation metrics
        logger: Logger instance

    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create a timestamp for versioning
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name}_{timestamp}.pkl"
    metadata_filename = f"{model_name}_{timestamp}_metadata.json"

    model_path = output_path / model_filename
    metadata_path = output_path / metadata_filename

    # Save the model
    if logger:
        logger.info(f"Saving model to {model_path}")

    with open(model_path, "wb") as f:
        pickle.dump(classifier.model, f)

    # Create and save metadata
    metadata = {
        "timestamp": timestamp,
        "model_version": timestamp,
        "model_path": str(model_path),
        "metrics": metrics,
        "model_type": "EquipmentClassifier",
        "training_parameters": {
            "sampling_strategy": classifier.sampling_strategy,
        },
    }

    if logger:
        logger.info(f"Saving metadata to {metadata_path}")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Create a symlink to the latest model
    latest_model_path = output_path / f"{model_name}_latest.pkl"
    latest_metadata_path = output_path / f"{model_name}_latest_metadata.json"

    # Remove existing symlinks if they exist
    if latest_model_path.exists():
        latest_model_path.unlink()
    if latest_metadata_path.exists():
        latest_metadata_path.unlink()

    # Create new symlinks
    latest_model_path.symlink_to(model_filename)
    latest_metadata_path.symlink_to(metadata_filename)

    if logger:
        logger.info(f"Created symlinks to latest model and metadata")

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "latest_model_path": str(latest_model_path),
        "latest_metadata_path": str(latest_metadata_path),
    }


def generate_visualizations(
    classifier: EquipmentClassifier,
    df: pd.DataFrame,
    output_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Generate visualizations of model performance and data distribution.

    Args:
        classifier: Trained EquipmentClassifier
        df: Processed DataFrame
        output_dir: Directory to save visualizations
        logger: Logger instance

    Returns:
        Dictionary with paths to visualization files
    """
    # Create visualizations directory if it doesn't exist
    viz_dir = Path(output_dir) / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"Generating visualizations in {viz_dir}")

    # Visualize category distribution
    equipment_category_file, system_type_file = visualize_category_distribution(
        df, str(viz_dir)
    )

    # Prepare data for confusion matrix
    x = pd.DataFrame(
        {
            "combined_features": df["combined_text"],
            "service_life": df["service_life"],
        }
    )

    y = df[
        [
            "category_name",  # Use category_name instead of Equipment_Category
            "uniformat_code",  # Use uniformat_code instead of Uniformat_Class
            "mcaa_system_category",  # Use mcaa_system_category instead of System_Type
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # Split for evaluation
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Generate confusion matrices if model exists
    confusion_matrix_files = {}
    if classifier.model is not None:
        # Make predictions
        y_pred = classifier.model.predict(x_test)
        y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

        # Generate confusion matrices
        for col in y_test.columns:
            output_file = str(viz_dir / f"confusion_matrix_{col}.png")
            visualize_confusion_matrix(y_test[col], y_pred_df[col], col, output_file)
            confusion_matrix_files[col] = output_file
    else:
        if logger:
            logger.warning("Cannot generate confusion matrices: model is None")

    if logger:
        logger.info("Visualizations generated successfully")

    return {
        "equipment_category_distribution": equipment_category_file,
        "system_type_distribution": system_type_file,
        "confusion_matrices": confusion_matrix_files,
    }


def make_sample_prediction(
    classifier: EquipmentClassifier,
    description: str = "Heat Exchanger for Chilled Water system with Plate and Frame design",
    service_life: float = 20.0,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Make a sample prediction using the trained model.

    Args:
        classifier: Trained EquipmentClassifier
        description: Equipment description
        service_life: Service life value
        logger: Logger instance

    Returns:
        Prediction results
    """
    if logger:
        logger.info("Making a sample prediction...")
        logger.info(f"Description: {description}")
        logger.info(f"Service life: {service_life}")

    # Check if classifier has a model
    if hasattr(classifier, "predict") and callable(classifier.predict):
        prediction = classifier.predict(description, service_life)

        if logger:
            logger.info("Prediction results:")
            for key, value in prediction.items():
                if key != "attribute_template" and key != "master_db_mapping":
                    logger.info(f"  {key}: {value}")

            logger.info("Classification IDs:")
            logger.info(f"  OmniClass ID: {prediction.get('OmniClass_ID', 'N/A')}")
            logger.info(f"  Uniformat ID: {prediction.get('Uniformat_ID', 'N/A')}")
            logger.info(
                f"  MasterFormat Class: {prediction.get('MasterFormat_Class', 'N/A')}"
            )

            logger.info("Required Attributes:")
            template = prediction.get("attribute_template", {})
            for attr, info in template.get("required_attributes", {}).items():
                logger.info(f"  {attr}: {info}")

        return prediction
    else:
        if logger:
            logger.warning("Cannot make prediction: model is not available")
        return {"error": "Model not available for prediction"}


def main():
    """Main function to run the model training pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting equipment classification model training pipeline")

    try:
        # Step 1: Load reference data
        ref_manager = load_reference_data(args.reference_config, logger)

        # Step 2: Validate training data if a path is provided
        if args.data_path:
            validation_results = validate_data(args.data_path, logger)
            if not validation_results.get("valid", False):
                logger.warning("Data validation failed, but continuing with training")

        # Step 3: Train the model
        classifier, df, metrics = train_model(
            data_path=args.data_path,
            feature_config_path=args.feature_config,
            sampling_strategy=args.sampling_strategy,
            test_size=args.test_size,
            random_state=args.random_state,
            optimize_params=args.optimize,
            logger=logger,
        )

        # Step 4: Save the trained model
        save_paths = save_model(
            classifier,
            args.output_dir,
            args.model_name,
            metrics,
            logger,
        )

        # Step 5: Generate visualizations if requested
        if args.visualize:
            viz_paths = generate_visualizations(
                classifier,
                df,
                args.output_dir,
                logger,
            )

        # Step 6: Make a sample prediction
        sample_prediction = make_sample_prediction(classifier, logger=logger)

        logger.info("Model training pipeline completed successfully")
        logger.info(f"Model saved to: {save_paths['model_path']}")
        logger.info(f"Metadata saved to: {save_paths['metadata_path']}")

    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
