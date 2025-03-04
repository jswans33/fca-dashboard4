"""
Main entry point for the FCA Dashboard ETL pipeline.

This module provides the main functionality to run the ETL pipeline,
including command-line argument parsing and pipeline execution.
"""

import argparse
import sys
from pathlib import Path

import yaml

from fca_dashboard.config.settings import get_settings
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

    try:
        # Resolve the configuration file path
        config_path = resolve_path(args.config)
        log.info(f"Loading configuration from {config_path}")

        # Load settings
        settings = get_settings(str(config_path))

        # Log startup information
        log.info("FCA Dashboard ETL Pipeline starting")
        log.info(f"Python version: {sys.version}")
        log.info(f"Current working directory: {Path.cwd()}")

        # TODO: Implement ETL pipeline execution
        # Steps include:
        # 1. Extract data from Excel or database source
        #    - Read source data using appropriate extractor strategy
        #    - Validate source data structure
        # 2. Transform data (cleaning, normalization, enrichment)
        #    - Apply business rules and transformations
        #    - Map source fields to destination schema
        # 3. Load data into destination database or output format
        #    - Batch insert/update operations
        #    - Validate data integrity after loading
        log.info("ETL Pipeline execution would start here")

        log.info(f"Database URL: {settings.get('databases.sqlite.url')}")

        if args.excel_file:
            excel_path = resolve_path(args.excel_file)
            log.info(f"Would process Excel file: {excel_path}")

        if args.table_name:
            log.info(f"Would process table: {args.table_name}")

        # Log successful completion
        log.info("ETL Pipeline completed successfully")
        return 0

    except FileNotFoundError as fnf:
        log.error(f"File not found: {fnf}")
        return 1
    except yaml.YAMLError as yaml_err:
        log.error(f"YAML configuration error: {yaml_err}")
        return 1
    except Exception as e:
        log.exception(f"Unexpected error in ETL Pipeline: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
