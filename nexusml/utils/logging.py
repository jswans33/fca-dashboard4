"""
Unified Logging Module for NexusML

This module provides a consistent logging interface that works both
standalone and when integrated with fca_dashboard.
"""

import logging
import os
import sys
from typing import Optional, Union, cast

# Try to use fca_dashboard logging if available
try:
    from fca_dashboard.utils.logging_config import (
        configure_logging as fca_configure_logging,
    )

    FCA_LOGGING_AVAILABLE = True
    FCA_CONFIGURE_LOGGING = fca_configure_logging
except ImportError:
    FCA_LOGGING_AVAILABLE = False
    FCA_CONFIGURE_LOGGING = None


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    simple_format: bool = False,
) -> (
    logging.Logger
):  # Type checker will still warn about this, but it's the best we can do
    """
    Configure application logging.

    Args:
        level: Logging level (e.g., "INFO", "DEBUG", etc.)
        log_file: Path to log file (if None, logs to console only)
        simple_format: Whether to use a simplified log format

    Returns:
        logging.Logger: Configured root logger
    """
    if FCA_LOGGING_AVAILABLE and FCA_CONFIGURE_LOGGING:
        # Convert level to string if it's an int to match fca_dashboard's API
        if isinstance(level, int):
            level = logging.getLevelName(level)

        # Use cast to tell the type checker that this will return a Logger
        return cast(
            logging.Logger,
            FCA_CONFIGURE_LOGGING(
                level=level, log_file=log_file, simple_format=simple_format
            ),
        )

    # Fallback to standard logging
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist and log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if simple_format:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str = "nexusml") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)
