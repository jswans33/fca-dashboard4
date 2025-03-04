"""
Logging configuration module for the FCA Dashboard application.

This module provides functionality to configure logging for the application
using Loguru, which offers improved formatting, better exception handling,
and simplified configuration compared to the standard logging module.
"""

import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger  # type: ignore


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 month",
    format_string: Optional[str] = None,
) -> None:
    """
    Configure application logging with console and optional file output using Loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is configured.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 month", "1 year")
        format_string: Custom format string for log messages
    """
    # Remove default handlers
    logger.remove()

    # Default format string if none provided
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(sys.stderr, level=level.upper(), format=format_string, colorize=True)

    # Add file handler if log_file is provided
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_dir = log_path.parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        # Add rotating file handler
        logger.add(
            log_file,
            level=level.upper(),
            format=format_string,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str = "fca_dashboard") -> Any:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name, typically the module name

    Returns:
        Loguru logger instance
    """
    return logger.bind(name=name)
