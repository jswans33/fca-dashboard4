"""
Common Utilities for NexusML Examples

This module provides shared functionality for example scripts to reduce code duplication
and ensure consistent behavior across examples.
"""

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import pandas as pd

from nexusml.config import get_data_path, get_output_dir
from nexusml.core.model import predict_with_enhanced_model, train_enhanced_model
from nexusml.utils.logging import get_logger

# Initialize logger
logger = get_logger(__name__)


def run_training_and_prediction(
    data_path: Union[str, Path, None] = None,
    description: str = "Heat Exchanger for Chilled Water system",
    service_life: float = 20.0,
    output_dir: Union[str, Path, None] = None,
    save_results: bool = True,
) -> Tuple[Any, pd.DataFrame, Dict[str, str]]:
    """
    Run a standard training and prediction workflow.

    Args:
        data_path: Path to training data CSV file (if None, uses default from config)
        description: Equipment description for prediction
        service_life: Service life value for prediction (in years)
        output_dir: Directory to save outputs (if None, uses default from config)
        save_results: Whether to save results to file

    Returns:
        Tuple: (trained model, training dataframe, prediction results)
    """
    # Use config for default paths
    if data_path is None:
        data_path = get_data_path("training_data")
        logger.info(f"Using default training data path: {data_path}")

    if output_dir is None:
        output_dir = get_output_dir()
        logger.info(f"Using default output directory: {output_dir}")

    # Convert Path objects to strings
    if isinstance(data_path, Path):
        data_path = str(data_path)

    if isinstance(output_dir, Path):
        output_dir = str(output_dir)

    # Create output directory if it doesn't exist
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Training
    logger.info(f"Training model using data from: {data_path}")
    model, df = train_enhanced_model(data_path)

    # Prediction
    logger.info(
        f"Making prediction for: {description} (service life: {service_life} years)"
    )
    prediction = predict_with_enhanced_model(model, description, service_life)

    # Save results if requested
    if save_results and output_dir is not None:
        prediction_file = os.path.join(output_dir, "example_prediction.txt")
        logger.info(f"Saving prediction results to: {prediction_file}")

        with open(prediction_file, "w") as f:
            f.write("Enhanced Prediction Results\n")
            f.write("==========================\n\n")
            f.write("Input:\n")
            f.write(f"  Description: {description}\n")
            f.write(f"  Service Life: {service_life} years\n\n")
            f.write("Prediction:\n")
            for key, value in prediction.items():
                f.write(f"  {key}: {value}\n")

    return model, df, prediction


def visualize_results(
    df: pd.DataFrame,
    model: Any,
    output_dir: Union[str, Path, None] = None,
    show_plots: bool = False,
) -> Dict[str, str]:
    """
    Generate visualizations for model results.

    Args:
        df: Training dataframe
        model: Trained model
        output_dir: Directory to save visualizations (if None, uses default from config)
        show_plots: Whether to display plots (in addition to saving them)

    Returns:
        Dict[str, str]: Paths to generated visualization files
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning(
            "Matplotlib and/or seaborn not available. Skipping visualizations."
        )
        return {}

    if output_dir is None:
        output_dir = get_output_dir()

    # Convert Path object to string if needed
    if isinstance(output_dir, Path):
        output_dir = str(output_dir)

    # If output_dir is still None, return empty dict
    if output_dir is None:
        logger.warning("Output directory is None. Skipping visualizations.")
        return {}

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    visualization_files = {}

    # Equipment Category Distribution
    equipment_category_file = os.path.join(
        output_dir, "equipment_category_distribution.png"
    )
    visualization_files["equipment_category"] = equipment_category_file

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="Equipment_Category")
    plt.title("Equipment Category Distribution")
    plt.tight_layout()
    plt.savefig(equipment_category_file)

    # System Type Distribution
    system_type_file = os.path.join(output_dir, "system_type_distribution.png")
    visualization_files["system_type"] = system_type_file

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="System_Type")
    plt.title("System Type Distribution")
    plt.tight_layout()
    plt.savefig(system_type_file)

    if not show_plots:
        plt.close("all")

    logger.info(f"Visualizations saved to: {output_dir}")
    return visualization_files
