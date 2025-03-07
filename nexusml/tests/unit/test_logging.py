#!/usr/bin/env python
"""
Tests for the logging module in nexusml.utils.logging
"""

import logging
import os
import tempfile
from unittest import mock

import pytest

from nexusml.utils.logging import configure_logging, get_logger


class TestLogging:
    """Tests for the logging module."""

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    @mock.patch("logging.FileHandler")
    def test_configure_logging_with_file(self, mock_file_handler):
        """Test configure_logging function with a log file."""
        # Mock the file handler
        mock_handler = mock.MagicMock()
        mock_file_handler.return_value = mock_handler

        # Configure logging with a file
        log_file = "test.log"
        logger = configure_logging(level="INFO", log_file=log_file)

        # Check that the logger was configured
        assert logger is not None
        assert logger.level == logging.INFO

        # Check that the file handler was created
        mock_file_handler.assert_called_once_with(log_file)

        # Check that the handler was added to the logger
        assert mock_handler in logger.handlers

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_without_file(self):
        """Test configure_logging function without a log file."""
        # Configure logging
        logger = configure_logging(level="DEBUG")

        # Check that the logger was configured
        assert logger is not None
        assert logger.level == logging.DEBUG

        # Check that the logger has a console handler
        assert any(
            isinstance(handler, logging.StreamHandler) for handler in logger.handlers
        )

        # Clean up
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    @mock.patch("logging.FileHandler")
    def test_configure_logging_with_simple_format(self, mock_file_handler):
        """Test configure_logging function with simple format."""
        # Mock the file handler
        mock_handler = mock.MagicMock()
        mock_file_handler.return_value = mock_handler

        # Configure logging with simple format
        log_file = "test_simple.log"
        logger = configure_logging(level="INFO", log_file=log_file, simple_format=True)

        # Check that the logger was configured
        assert logger is not None

        # Check that the file handler was created
        mock_file_handler.assert_called_once_with(log_file)

        # Check that the handler was added to the logger
        assert mock_handler in logger.handlers

        # Check that the formatter is using simple format
        # Get the formatter from the call to setFormatter
        formatter_call = mock_handler.setFormatter.call_args
        assert formatter_call is not None
        formatter = formatter_call[0][0]
        assert formatter._fmt == "%(message)s"

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_with_int_level(self):
        """Test configure_logging function with integer level."""
        # Configure logging with integer level
        logger = configure_logging(level=logging.WARNING)

        # Check that the logger was configured with the correct level
        assert logger is not None
        assert logger.level == logging.WARNING

        # Clean up
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", True)
    @mock.patch("nexusml.utils.logging.FCA_CONFIGURE_LOGGING")
    def test_configure_logging_with_fca(self, mock_fca_configure):
        """Test configure_logging function when FCA logging is available."""
        # Mock the FCA configure_logging function
        mock_logger = mock.MagicMock()
        mock_fca_configure.return_value = mock_logger

        # Configure logging
        logger = configure_logging(
            level="INFO", log_file="test.log", simple_format=True
        )

        # Check that FCA configure_logging was called
        mock_fca_configure.assert_called_once_with(
            level="INFO", log_file="test.log", simple_format=True
        )

        # Check that the logger returned is the one from FCA
        assert logger is mock_logger

    def test_get_logger(self):
        """Test get_logger function."""
        # Get a logger
        logger = get_logger("test_get_logger")

        # Check that the logger was created with the correct name
        assert logger.name == "test_get_logger"

    def test_logger_methods(self):
        """Test standard logger methods."""
        # Create a mock logger to avoid file operations
        mock_logger = mock.MagicMock(spec=logging.Logger)

        # Test various logging methods
        mock_logger.debug("Debug message")
        mock_logger.info("Info message")
        mock_logger.warning("Warning message")
        mock_logger.error("Error message")

        # Verify that the logging methods were called
        mock_logger.debug.assert_called_once_with("Debug message")
        mock_logger.info.assert_called_once_with("Info message")
        mock_logger.warning.assert_called_once_with("Warning message")
        mock_logger.error.assert_called_once_with("Error message")


if __name__ == "__main__":
    pytest.main()
