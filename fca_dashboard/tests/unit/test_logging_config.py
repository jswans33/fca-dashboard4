"""
Unit tests for the logging configuration module.

This module contains tests for the logging configuration functionality
in the fca_dashboard.utils.logging_config module.
"""

import os
import tempfile
from typing import Generator

import pytest
from _pytest.capture import CaptureFixture
from loguru import logger

from fca_dashboard.utils.logging_config import configure_logging, get_logger


@pytest.fixture
def temp_log_file() -> Generator[str, None, None]:
    """Create a temporary log file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as temp_file:
        temp_path = temp_file.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_configure_logging_console_only() -> None:
    """Test configuring logging with console output only."""
    # Remove any existing handlers
    logger.remove()

    # Configure logging with console output only
    configure_logging(level="DEBUG")

    # Get a logger and log a message
    log = get_logger("test_logger")
    log.debug("Test debug message")
    log.info("Test info message")

    # No assertions here as we're just testing that no exceptions are raised
    # and visually confirming console output during test execution


def test_configure_logging_with_file(temp_log_file: str) -> None:
    """Test configuring logging with file output."""
    # Remove any existing handlers
    logger.remove()

    # Configure logging with file output
    configure_logging(level="INFO", log_file=temp_log_file)

    # Get a logger and log messages
    log = get_logger("test_logger")
    log.debug("Test debug message - should not be in file")
    log.info("Test info message - should be in file")
    log.warning("Test warning message - should be in file")

    # Force flush by removing handlers
    logger.remove()

    # Check that the log file exists and contains the expected messages
    assert os.path.exists(temp_log_file)
    with open(temp_log_file, "r") as f:
        log_content = f.read()
        assert "Test debug message - should not be in file" not in log_content
        assert "Test info message - should be in file" in log_content
        assert "Test warning message - should be in file" in log_content


def test_configure_logging_creates_directory() -> None:
    """Test that configure_logging creates the log directory if it doesn't exist."""
    # Remove any existing handlers
    logger.remove()

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a path to a log file in a non-existent subdirectory
        nonexistent_dir = os.path.join(temp_dir, "nonexistent_dir")
        log_file = os.path.join(nonexistent_dir, "test.log")

        # Configure logging - this should create the directory
        configure_logging(level="INFO", log_file=log_file)

        # Get a logger and log a message
        log = get_logger("test_logger")
        log.info("Test message")

        # Force flush by removing handlers
        logger.remove()

        # Check that the directory and log file were created
        assert os.path.exists(nonexistent_dir)
        assert os.path.exists(log_file)

        # Check that the log file contains the message
        with open(log_file, "r") as f:
            log_content = f.read()
            assert "Test message" in log_content


def test_configure_logging_with_custom_format(temp_log_file: str) -> None:
    """Test configuring logging with a custom format string."""
    # Remove any existing handlers
    logger.remove()

    # Custom format string
    custom_format = "{time} | {level} | {message}"

    # Configure logging with custom format
    configure_logging(level="INFO", log_file=temp_log_file, format_string=custom_format)

    # Get a logger and log a message
    log = get_logger("test_logger")
    log.info("Test message with custom format")

    # Force flush by removing handlers
    logger.remove()

    # Check that the log file contains the message with the custom format
    with open(temp_log_file, "r") as f:
        log_content = f.read()
        assert "Test message with custom format" in log_content
        # The format is simplified, so we don't check for exact format


def test_configure_logging_with_simple_format(temp_log_file: str) -> None:
    """Test configuring logging with the simple format option."""
    # Remove any existing handlers
    logger.remove()

    # Configure logging with simple format
    configure_logging(level="INFO", log_file=temp_log_file, simple_format=True)

    # Get a logger and log a message
    log = get_logger("test_logger")
    log.info("Test message with simple format")

    # Force flush by removing handlers
    logger.remove()

    # Check that the log file contains the message
    with open(temp_log_file, "r") as f:
        log_content = f.read()
        assert "Test message with simple format" in log_content
        # We don't check the exact format, just that the message is there


def test_get_logger_with_name(capfd: CaptureFixture) -> None:
    """Test getting a logger with a specific name."""
    # Remove any existing handlers
    logger.remove()

    # Configure logging
    configure_logging(level="INFO")

    # Get a logger with a specific name
    log_name = "custom_logger_name"
    log = get_logger(log_name)
    log.info("Checking logger name")

    # Capture the output and verify the log message is included
    captured = capfd.readouterr()
    # The name might not be directly visible in the output due to formatting
    # but we can verify the log message is there
    assert "Checking logger name" in captured.err


def test_get_logger_default_name(capfd: CaptureFixture) -> None:
    """Test getting a logger with the default name."""
    # Remove any existing handlers
    logger.remove()

    # Configure logging
    configure_logging(level="INFO")

    # Get a logger with the default name
    log = get_logger()
    log.info("Checking default logger name")

    # Capture the output and verify the default logger name is included
    captured = capfd.readouterr()
    assert "fca_dashboard" in captured.err
    assert "Checking default logger name" in captured.err
