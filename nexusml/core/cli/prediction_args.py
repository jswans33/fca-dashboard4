"""
Prediction Pipeline Argument Parsing Module

This module provides argument parsing functionality for the prediction pipeline,
using argparse for command-line arguments with validation and documentation.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class PredictionArgumentParser:
    """
    Argument parser for the prediction pipeline.

    This class encapsulates the logic for parsing and validating command-line
    arguments for the prediction pipeline.
    """

    def __init__(self) -> None:
        """Initialize a new PredictionArgumentParser."""
        self.parser = argparse.ArgumentParser(
            description="Make equipment classification predictions"
        )
        self._configure_parser()

    def _configure_parser(self) -> None:
        """Configure the argument parser with all required arguments."""
        # Model arguments
        self.parser.add_argument(
            "--model-path",
            type=str,
            default="outputs/models/equipment_classifier_latest.pkl",
            help="Path to the trained model file",
        )

        # Input/output arguments
        self.parser.add_argument(
            "--input-file",
            type=str,
            required=True,
            help="Path to the input CSV file with equipment descriptions",
        )
        self.parser.add_argument(
            "--output-file",
            type=str,
            default="prediction_results.csv",
            help="Path to save the prediction results",
        )

        # Logging arguments
        self.parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level (default: INFO)",
        )

        # Column mapping arguments
        self.parser.add_argument(
            "--description-column",
            type=str,
            default="Description",
            help="Column name containing equipment descriptions",
        )
        self.parser.add_argument(
            "--service-life-column",
            type=str,
            default="Service Life",
            help="Column name containing service life values",
        )
        self.parser.add_argument(
            "--asset-tag-column",
            type=str,
            default="Asset Tag",
            help="Column name containing asset tags",
        )

        # Feature engineering arguments
        self.parser.add_argument(
            "--feature-config-path",
            type=str,
            default=None,
            help="Path to the feature engineering configuration file",
        )

        # Architecture selection arguments
        self.parser.add_argument(
            "--use-orchestrator",
            action="store_true",
            help="Use the new pipeline orchestrator (default: False)",
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Args:
            args: List of command-line arguments to parse. If None, uses sys.argv.

        Returns:
            Parsed arguments as a Namespace object.
        """
        return self.parser.parse_args(args)

    def parse_args_to_dict(self, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Parse command-line arguments and convert to a dictionary.

        Args:
            args: List of command-line arguments to parse. If None, uses sys.argv.

        Returns:
            Dictionary of parsed arguments.
        """
        namespace = self.parse_args(args)
        return vars(namespace)

    def validate_args(self, args: argparse.Namespace) -> None:
        """
        Validate parsed arguments.

        Args:
            args: Parsed arguments to validate.

        Raises:
            ValueError: If any arguments are invalid.
        """
        # Validate model path
        if not Path(args.model_path).exists():
            raise ValueError(f"Model file not found: {args.model_path}")

        # Validate input file
        if not Path(args.input_file).exists():
            raise ValueError(f"Input file not found: {args.input_file}")

        # Validate feature config path if provided
        if args.feature_config_path and not Path(args.feature_config_path).exists():
            raise ValueError(
                f"Feature config file not found: {args.feature_config_path}"
            )

        # Validate log level
        try:
            numeric_level = getattr(logging, args.log_level.upper())
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {args.log_level}")
        except (AttributeError, ValueError):
            raise ValueError(f"Invalid log level: {args.log_level}")

    def setup_logging(self, args: argparse.Namespace) -> logging.Logger:
        """
        Set up logging based on the parsed arguments.

        Args:
            args: Parsed arguments containing logging configuration.

        Returns:
            Configured logger instance.
        """
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Set up logging
        numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "prediction.log"),
                logging.StreamHandler(),
            ],
        )

        return logging.getLogger("equipment_prediction")
