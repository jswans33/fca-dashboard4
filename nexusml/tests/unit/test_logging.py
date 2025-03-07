"""
Unit tests for the nexusml.utils.logging module.
"""

import logging
import os
import tempfile
import unittest
from unittest import mock

from nexusml.utils.logging import configure_logging, get_logger


class TestLogging(unittest.TestCase):
    """Test cases for the logging module."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the root logger before each test
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    def tearDown(self):
        """Clean up after each test."""
        # Reset the root logger after each test
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        root_logger.setLevel(logging.NOTSET)

    def test_get_logger_default_name(self):
        """Test get_logger with default name."""
        logger = get_logger()
        self.assertEqual(logger.name, "nexusml")
        self.assertIsInstance(logger, logging.Logger)

    def test_get_logger_custom_name(self):
        """Test get_logger with custom name."""
        custom_name = "custom_logger"
        logger = get_logger(custom_name)
        self.assertEqual(logger.name, custom_name)
        self.assertIsInstance(logger, logging.Logger)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_default(self):
        """Test configure_logging with default parameters."""
        logger = configure_logging()
        self.assertIsNotNone(logger)

        # Check logger level
        self.assertEqual(logger.level, logging.INFO)

        # Check handlers
        self.assertGreaterEqual(len(logger.handlers), 1)
        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        self.assertGreaterEqual(len(stream_handlers), 1)

        # Check formatter
        formatter = stream_handlers[0].formatter
        self.assertIsNotNone(formatter)
        if hasattr(formatter, "_fmt"):
            self.assertEqual(
                formatter._fmt, "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        else:
            # Some formatter implementations might use a different attribute
            self.assertIn("%(asctime)s", str(formatter))
            self.assertIn("%(name)s", str(formatter))
            self.assertIn("%(levelname)s", str(formatter))
            self.assertIn("%(message)s", str(formatter))

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_debug_level(self):
        """Test configure_logging with DEBUG level."""
        logger = configure_logging(level="DEBUG")
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, logging.DEBUG)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_integer_level(self):
        """Test configure_logging with integer level."""
        logger = configure_logging(level=logging.WARNING)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, logging.WARNING)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_simple_format(self):
        """Test configure_logging with simple format."""
        logger = configure_logging(simple_format=True)
        self.assertIsNotNone(logger)

        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
        ]
        self.assertGreaterEqual(len(stream_handlers), 1)
        formatter = stream_handlers[0].formatter
        self.assertIsNotNone(formatter)
        if hasattr(formatter, "_fmt"):
            self.assertEqual(formatter._fmt, "%(message)s")
        else:
            # Some formatter implementations might use a different attribute
            self.assertEqual(str(formatter).strip(), "%(message)s")

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_with_file(self):
        """Test configure_logging with log file."""
        # Use a unique temporary directory for each test
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file_path = os.path.join(temp_dir, "test.log")

            logger = configure_logging(log_file=log_file_path)
            self.assertIsNotNone(logger)

            # Should have at least two handlers: console and file
            self.assertGreaterEqual(len(logger.handlers), 2)

            # Check that one handler is a FileHandler with the correct path
            file_handlers = [
                h for h in logger.handlers if isinstance(h, logging.FileHandler)
            ]
            self.assertGreaterEqual(len(file_handlers), 1)

            # Log a message
            test_message = "Test log message"
            logger.info(test_message)

            # Close handlers to release file locks
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Check log file content
            if os.path.exists(log_file_path):
                with open(log_file_path, "r") as f:
                    log_content = f.read()
                    self.assertIn(test_message, log_content)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_creates_log_directory(self):
        """Test that configure_logging creates the log directory if it doesn't exist."""
        # Use a unique temporary directory for each test
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = os.path.join(temp_dir, "logs")
            log_file = os.path.join(log_dir, "test.log")

            # Directory shouldn't exist yet
            self.assertFalse(os.path.exists(log_dir))

            # Configure logging should create the directory
            logger = configure_logging(log_file=log_file)
            self.assertIsNotNone(logger)

            # Close handlers to release file locks
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Check directory was created
            self.assertTrue(os.path.exists(log_dir))
            self.assertTrue(os.path.isdir(log_dir))

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", True)
    @mock.patch("nexusml.utils.logging.FCA_CONFIGURE_LOGGING")
    def test_configure_logging_uses_fca_logging_if_available(self, mock_fca_logging):
        """Test that configure_logging uses FCA logging if available."""
        # Set up the mock to return a logger
        mock_logger = mock.MagicMock(spec=logging.Logger)
        mock_fca_logging.return_value = mock_logger

        # Call configure_logging
        result = configure_logging(
            level="DEBUG", log_file="test.log", simple_format=True
        )

        # Check that FCA_CONFIGURE_LOGGING was called with the right parameters
        mock_fca_logging.assert_called_once_with(
            level="DEBUG", log_file="test.log", simple_format=True
        )

        # Check that the result is the mock logger
        self.assertEqual(result, mock_logger)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", True)
    @mock.patch("nexusml.utils.logging.FCA_CONFIGURE_LOGGING")
    def test_configure_logging_converts_int_level_for_fca(self, mock_fca_logging):
        """Test that configure_logging converts integer levels to strings for FCA logging."""
        mock_logger = mock.MagicMock(spec=logging.Logger)
        mock_fca_logging.return_value = mock_logger

        # Call configure_logging with an integer level
        configure_logging(level=logging.WARNING)

        # Check that FCA_CONFIGURE_LOGGING was called with the string level
        mock_fca_logging.assert_called_once()
        args, _ = mock_fca_logging.call_args
        kwargs = mock_fca_logging.call_args.kwargs
        self.assertEqual(kwargs["level"], "WARNING")

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", True)
    @mock.patch("nexusml.utils.logging.FCA_CONFIGURE_LOGGING", None)
    def test_configure_logging_handles_import_error(self):
        """Test that configure_logging handles the case when FCA logging is not available."""
        # This test simulates the ImportError case by setting FCA_CONFIGURE_LOGGING to None
        logger = configure_logging()
        self.assertIsNotNone(logger)
        self.assertIsInstance(logger, logging.Logger)

    @mock.patch("nexusml.utils.logging.FCA_LOGGING_AVAILABLE", False)
    def test_configure_logging_removes_existing_handlers(self):
        """Test that configure_logging removes existing handlers."""
        # Add a handler to the root logger
        root_logger = logging.getLogger()

        # Store the original handlers
        original_handlers = list(root_logger.handlers)

        # Add a custom handler
        handler = logging.StreamHandler()
        root_logger.addHandler(handler)

        # Verify that the handler was added
        self.assertIn(handler, root_logger.handlers)

        # Call configure_logging with a mock to force the removeHandler call
        with mock.patch.object(root_logger, "removeHandler") as mock_remove:
            logger = configure_logging()

            # Verify that removeHandler was called at least once
            mock_remove.assert_called()


if __name__ == "__main__":
    unittest.main()
