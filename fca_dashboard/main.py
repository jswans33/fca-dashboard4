"""
Main entry point for the FCA Dashboard ETL pipeline.

This module provides the main functionality to run the ETL pipeline,
including command-line argument parsing and pipeline execution.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from fca_dashboard.config.settings import get_settings
from fca_dashboard.utils.error_handler import (
    ConfigurationError,
    DataExtractionError,
    ErrorHandler,
)
from fca_dashboard.utils.logging_config import configure_logging, get_logger
from fca_dashboard.utils.path_util import get_logs_path, resolve_path


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="FCA Dashboard ETL Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config" / "settings.yml"),
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument("--excel-file", type=str, help="Path to Excel file to process")

    parser.add_argument("--table-name", type=str, help="Name of the table to process")

    return parser.parse_args()


def run_etl_pipeline(args: argparse.Namespace, log: Any) -> int:
    """
    Run the ETL pipeline with the given arguments.

    Args:
        args: Command line arguments
        log: Logger instance

    Returns:
        Exit code (0 for success, non-zero for failure)

    Raises:
        ConfigurationError: If there is an error in the configuration
        DataExtractionError: If there is an error extracting data
        FileNotFoundError: If a required file is not found
    """
    # Resolve the configuration file path
    config_path = resolve_path(args.config)
    log.info(f"Loading configuration from {config_path}")

    # Load settings
    try:
        settings = get_settings(str(config_path))
    except yaml.YAMLError as yaml_err:
        raise ConfigurationError(f"YAML configuration error: {yaml_err}") from yaml_err

    # Log startup information
    log.info("FCA Dashboard ETL Pipeline starting")
    log.info(f"Python version: {sys.version}")
    log.info(f"Current working directory: {Path.cwd()}")

    # TODO: Implement ETL pipeline execution (See GitHub issue #123)
    # Steps include:
    # 1. Extract data from Excel or database source (See GitHub issue #124)
    #    - Read source data using appropriate extractor strategy
    #    - Validate source data structure
    # 2. Transform data (cleaning, normalization, enrichment) (See GitHub issue #125)
    #    - Apply business rules and transformations
    #    - Map source fields to destination schema
    # 3. Load data into destination database or output format (See GitHub issue #126)
    #    - Batch insert/update operations
    #    - Validate data integrity after loading
    log.info("ETL Pipeline execution would start here")

    log.info(f"Database URL: {settings.get('databases.sqlite.url')}")

    if args.excel_file:
        try:
            excel_path = resolve_path(args.excel_file)
            log.info(f"Would process Excel file: {excel_path}")
        except FileNotFoundError:
            raise DataExtractionError(f"Excel file not found: {args.excel_file}") from None

    if args.table_name:
        log.info(f"Would process table: {args.table_name}")

    # Log successful completion
    log.info("ETL Pipeline completed successfully")
    return 0


def main() -> int:
    """
    Main entry point for the ETL pipeline.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    log_file = get_logs_path("fca_dashboard.log")
    configure_logging(level=args.log_level, log_file=str(log_file), rotation="10 MB", retention="1 month")

    # Get a logger for this module
    log = get_logger("main")

    # Create an error handler
    error_handler = ErrorHandler("main")

    # Run the ETL pipeline with error handling
    try:
        return run_etl_pipeline(args, log)
    except Exception as e:
        return error_handler.handle_error(e)


if __name__ == "__main__":
    sys.exit(main())
