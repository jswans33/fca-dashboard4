#!/usr/bin/env python
"""
Training Arguments Module

This module defines the command-line arguments for the training pipeline
and provides utilities for parsing and validating them.
"""

import argparse
import datetime
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union


@dataclass
class TrainingArguments:
    """
    Training arguments for the equipment classification model.

    This class encapsulates all the arguments needed for training the model,
    including data paths, training parameters, and output settings.
    """

    # Data arguments
    data_path: str
    feature_config_path: Optional[str] = None
    reference_config_path: Optional[str] = None

    # Training arguments
    test_size: float = 0.3
    random_state: int = 42
    sampling_strategy: str = "direct"
    optimize_hyperparameters: bool = False

    # Output arguments
    output_dir: str = "outputs/models"
    model_name: str = "equipment_classifier"
    log_level: str = "INFO"
    visualize: bool = False

    # Feature flags
    use_orchestrator: bool = True

    def __post_init__(self):
        """Validate arguments after initialization."""
        # Validate data_path
        if self.data_path and not os.path.exists(self.data_path):
            raise ValueError(f"Data path does not exist: {self.data_path}")

        # Validate feature_config_path
        if self.feature_config_path and not os.path.exists(self.feature_config_path):
            raise ValueError(
                f"Feature config path does not exist: {self.feature_config_path}"
            )

        # Validate reference_config_path
        if self.reference_config_path and not os.path.exists(
            self.reference_config_path
        ):
            raise ValueError(
                f"Reference config path does not exist: {self.reference_config_path}"
            )

        # Validate test_size
        if not 0 < self.test_size < 1:
            raise ValueError(f"Test size must be between 0 and 1, got {self.test_size}")

        # Validate sampling_strategy
        valid_strategies = ["direct"]
        if self.sampling_strategy not in valid_strategies:
            raise ValueError(
                f"Sampling strategy must be one of {valid_strategies}, got {self.sampling_strategy}"
            )

        # Validate log_level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"Log level must be one of {valid_log_levels}, got {self.log_level}"
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def to_dict(self) -> Dict:
        """
        Convert arguments to a dictionary.

        Returns:
            Dictionary representation of the arguments.
        """
        return asdict(self)


def parse_args() -> TrainingArguments:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments as a TrainingArguments object.
    """
    parser = argparse.ArgumentParser(
        description="Train the equipment classification model"
    )

    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training data CSV file",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        help="Path to the feature configuration YAML file",
        dest="feature_config_path",
    )
    parser.add_argument(
        "--reference-config",
        type=str,
        help="Path to the reference configuration YAML file",
        dest="reference_config_path",
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
        "--optimize",
        action="store_true",
        help="Perform hyperparameter optimization",
        dest="optimize_hyperparameters",
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

    # Feature flags
    parser.add_argument(
        "--legacy",
        action="store_false",
        help="Use legacy implementation instead of orchestrator",
        dest="use_orchestrator",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create TrainingArguments object
    return TrainingArguments(
        data_path=args.data_path,
        feature_config_path=args.feature_config_path,
        reference_config_path=args.reference_config_path,
        test_size=args.test_size,
        random_state=args.random_state,
        sampling_strategy=args.sampling_strategy,
        optimize_hyperparameters=args.optimize_hyperparameters,
        output_dir=args.output_dir,
        model_name=args.model_name,
        log_level=args.log_level,
        visualize=args.visualize,
        use_orchestrator=args.use_orchestrator,
    )


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

    # Get the logger
    logger = logging.getLogger("model_training")

    # Set the logger level
    logger.setLevel(numeric_level)

    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()

    # Add handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
