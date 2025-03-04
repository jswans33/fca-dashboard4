"""
Unit tests for the error handling module.

This module contains tests for the error handling functionality
in the fca_dashboard.utils.error_handler module.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from fca_dashboard.utils.error_handler import (
    ConfigurationError,
    DataExtractionError,
    DataLoadingError,
    DataTransformationError,
    ErrorHandler,
    FCADashboardError,
    ValidationError,
)


def test_error_handler_initialization() -> None:
    """Test that the ErrorHandler initializes correctly."""
    # Mock the get_logger function to verify it's called with the correct name
    with patch("fca_dashboard.utils.error_handler.get_logger") as mock_get_logger:
        handler = ErrorHandler("test_handler")
        mock_get_logger.assert_called_once_with("test_handler")


def test_handle_file_not_found_error() -> None:
    """Test handling of FileNotFoundError."""
    handler = ErrorHandler()
    error = FileNotFoundError("test.txt")
    exit_code = handler.handle_error(error)
    assert exit_code == 1


def test_handle_configuration_error() -> None:
    """Test handling of ConfigurationError."""
    handler = ErrorHandler()
    error = ConfigurationError("Invalid configuration")
    exit_code = handler.handle_error(error)
    assert exit_code == 2


def test_handle_data_extraction_error() -> None:
    """Test handling of DataExtractionError."""
    handler = ErrorHandler()
    error = DataExtractionError("Failed to extract data")
    exit_code = handler.handle_error(error)
    assert exit_code == 3


def test_handle_data_transformation_error() -> None:
    """Test handling of DataTransformationError."""
    handler = ErrorHandler()
    error = DataTransformationError("Failed to transform data")
    exit_code = handler.handle_error(error)
    assert exit_code == 4


def test_handle_data_loading_error() -> None:
    """Test handling of DataLoadingError."""
    handler = ErrorHandler()
    error = DataLoadingError("Failed to load data")
    exit_code = handler.handle_error(error)
    assert exit_code == 5


def test_handle_validation_error() -> None:
    """Test handling of ValidationError."""
    handler = ErrorHandler()
    error = ValidationError("Data validation failed")
    exit_code = handler.handle_error(error)
    assert exit_code == 6


def test_handle_generic_error() -> None:
    """Test handling of a generic Exception."""
    handler = ErrorHandler()
    error = Exception("Generic error")
    exit_code = handler.handle_error(error)
    assert exit_code == 99


def test_with_error_handling_decorator_success() -> None:
    """Test that the with_error_handling decorator works for successful functions."""
    handler = ErrorHandler()

    @handler.with_error_handling
    def successful_function() -> str:
        return "success"

    result = successful_function()
    assert result == "success"


def test_with_error_handling_decorator_error() -> None:
    """Test that the with_error_handling decorator handles errors correctly."""
    handler = ErrorHandler()
    mock_handle_error = MagicMock(return_value=42)
    handler.handle_error = mock_handle_error

    # Test with a function that returns int
    @handler.with_error_handling
    def failing_function_with_int_return() -> int:
        raise ValueError("Test error")

    # When the function returns int, the decorator should return the exit code
    result = failing_function_with_int_return()
    assert result == 42
    mock_handle_error.assert_called_once()


def test_with_error_handling_decorator_pytest_behavior() -> None:
    """Test that the with_error_handling decorator raises exceptions in pytest environment."""
    handler = ErrorHandler()
    mock_handle_error = MagicMock(return_value=42)
    handler.handle_error = mock_handle_error

    # Test with a function that returns None
    @handler.with_error_handling
    def failing_function() -> None:
        raise ValueError("Test error")

    # In pytest environment, the decorator should re-raise the exception
    with patch("sys.exit") as mock_exit:
        with pytest.raises(ValueError, match="Test error"):
            failing_function()
        
        # Verify sys.exit was NOT called
        mock_exit.assert_not_called()
        # Ensure handle_error was called correctly
        mock_handle_error.assert_called_once()


def test_with_error_handling_decorator_exit() -> None:
    """Test that the with_error_handling decorator calls sys.exit for non-int return types."""
    handler = ErrorHandler()
    mock_handle_error = MagicMock(return_value=42)
    handler.handle_error = mock_handle_error

    # Test with a function that returns None
    @handler.with_error_handling
    def failing_function() -> None:
        raise ValueError("Test error")

    # We need to patch both sys.modules and sys.exit
    with patch.dict("sys.modules", {"pytest": None}):  # Simulate non-pytest environment
        with patch("sys.exit") as mock_exit:
            failing_function()
            mock_exit.assert_called_once_with(42)


def test_custom_exception_inheritance() -> None:
    """Test that custom exceptions inherit correctly."""
    assert issubclass(ConfigurationError, FCADashboardError)
    assert issubclass(DataExtractionError, FCADashboardError)
    assert issubclass(DataTransformationError, FCADashboardError)
    assert issubclass(DataLoadingError, FCADashboardError)
    assert issubclass(ValidationError, FCADashboardError)
    assert issubclass(FCADashboardError, Exception)


def test_custom_exception_message() -> None:
    """Test that custom exceptions store the message correctly."""
    message = "Test error message"
    error = FCADashboardError(message)
    assert error.message == message
    assert str(error) == message


def test_error_handler_with_main_function() -> None:
    """Test integration of ErrorHandler with a main-like function."""
    handler = ErrorHandler()

    # Mock a main-like function that raises different exceptions
    def mock_main(exception_type: Exception) -> int:
        try:
            raise exception_type
        except Exception as e:
            return handler.handle_error(e)

    # Test with different exception types
    assert mock_main(FileNotFoundError("test.txt")) == 1
    assert mock_main(ConfigurationError("Invalid config")) == 2
    assert mock_main(Exception("Generic error")) == 99
