"""
Logging configuration module for the FCA Dashboard application.

This module provides functionality to configure logging for the application
using Loguru, which offers improved formatting, better exception handling,
and simplified configuration compared to the standard logging module.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from loguru import logger  # type: ignore

# Define a Record type for type hints
Record = Dict[str, Any]
# Define a FormatFunction type for type hints
FormatFunction = Callable[[Record], str]


def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 month",
    format_string: Optional[Union[str, Callable[[Record], str]]] = None,
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
        # Define a custom format function that safely handles extra[name]
        def safe_format(record: Record) -> str:
            # Add the name from extra if available, otherwise use empty string
            name = record["extra"].get("name", "")
            name_part = f"<cyan>{name}</cyan> | " if name else ""

            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                f"{name_part}"
                "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ).format_map(record)

        format_string = safe_format

    # Add console handler
    logger.add(sys.stderr, level=level.upper(), format=format_string, colorize=True)  # type: ignore

    # Add file handler if log_file is provided
    if log_file:
        # Create the log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add rotating file handler
        logger.add(  # type: ignore[arg-type]
            str(log_path),
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
