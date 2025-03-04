"""
Error handling module for the FCA Dashboard application.

This module provides a centralized error handling mechanism for the application,
including custom exceptions and an error handler class that integrates with
the logging system.
"""

import sys
from typing import Any, Callable, Dict, Type, TypeVar, cast

from fca_dashboard.utils.logging_config import get_logger

# Type variable for function return type
T = TypeVar("T")


class FCADashboardError(Exception):
    """Base exception class for all FCA Dashboard application errors."""

    def __init__(self, message: str, *args: Any) -> None:
        """
        Initialize the exception with a message and optional arguments.

        Args:
            message: Error message
            *args: Additional arguments to pass to the Exception constructor
        """
        self.message = message
        super().__init__(message, *args)


class ConfigurationError(FCADashboardError):
    """Exception raised for errors in the configuration."""

    pass


class DataExtractionError(FCADashboardError):
    """Exception raised for errors during data extraction."""

    pass


class DataTransformationError(FCADashboardError):
    """Exception raised for errors during data transformation."""

    pass


class DataLoadingError(FCADashboardError):
    """Exception raised for errors during data loading."""

    pass


class ValidationError(FCADashboardError):
    """Exception raised for data validation errors."""

    pass


class ErrorHandler:
    """
    Centralized error handler for the FCA Dashboard application.

    This class provides methods for handling errors in a consistent way
    throughout the application, including logging and appropriate responses.
    """

    def __init__(self, logger_name: str = "error_handler") -> None:
        """
        Initialize the error handler with a logger.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = get_logger(logger_name)
        self.error_mapping: Dict[Type[Exception], Callable[[Exception], int]] = {
            FileNotFoundError: self._handle_file_not_found,
            ConfigurationError: self._handle_configuration_error,
            DataExtractionError: self._handle_data_extraction_error,
            DataTransformationError: self._handle_data_transformation_error,
            DataLoadingError: self._handle_data_loading_error,
            ValidationError: self._handle_validation_error,
        }

    def handle_error(self, error: Exception) -> int:
        """
        Handle an exception by logging it and returning an appropriate exit code.

        Args:
            error: The exception to handle

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        # Find the most specific handler for this error type
        for error_type, handler in self.error_mapping.items():
            if isinstance(error, error_type):
                return handler(error)

        # If no specific handler is found, use the generic handler
        return self._handle_generic_error(error)

    def _handle_file_not_found(self, error: FileNotFoundError) -> int:
        """
        Handle a FileNotFoundError.

        Args:
            error: The FileNotFoundError to handle

        Returns:
            Exit code (1 for file not found)
        """
        self.logger.error(f"File not found: {error}")
        return 1

    def _handle_configuration_error(self, error: ConfigurationError) -> int:
        """
        Handle a ConfigurationError.

        Args:
            error: The ConfigurationError to handle

        Returns:
            Exit code (2 for configuration error)
        """
        self.logger.error(f"Configuration error: {error.message}")
        return 2

    def _handle_data_extraction_error(self, error: DataExtractionError) -> int:
        """
        Handle a DataExtractionError.

        Args:
            error: The DataExtractionError to handle

        Returns:
            Exit code (3 for data extraction error)
        """
        self.logger.error(f"Data extraction error: {error.message}")
        return 3

    def _handle_data_transformation_error(self, error: DataTransformationError) -> int:
        """
        Handle a DataTransformationError.

        Args:
            error: The DataTransformationError to handle

        Returns:
            Exit code (4 for data transformation error)
        """
        self.logger.error(f"Data transformation error: {error.message}")
        return 4

    def _handle_data_loading_error(self, error: DataLoadingError) -> int:
        """
        Handle a DataLoadingError.

        Args:
            error: The DataLoadingError to handle

        Returns:
            Exit code (5 for data loading error)
        """
        self.logger.error(f"Data loading error: {error.message}")
        return 5

    def _handle_validation_error(self, error: ValidationError) -> int:
        """
        Handle a ValidationError.

        Args:
            error: The ValidationError to handle

        Returns:
            Exit code (6 for validation error)
        """
        self.logger.error(f"Validation error: {error.message}")
        return 6

    def _handle_generic_error(self, error: Exception) -> int:
        """
        Handle a generic exception.

        Args:
            error: The exception to handle

        Returns:
            Exit code (99 for generic error)
        """
        self.logger.exception(f"Unexpected error: {error}")
        return 99

    def with_error_handling(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to wrap a function with error handling.

        Args:
            func: The function to wrap

        Returns:
            Wrapped function with error handling
        """
        from typing import get_type_hints

        # Get the return type annotation of the function
        return_type = get_type_hints(func).get('return')
        returns_int = return_type is int

        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the function name where the error occurred for easier debugging
                self.logger.error(f"Error occurred in function '{func.__name__}'")
                # Call handle_error to get the exit code and log the error
                exit_code = self.handle_error(e)
                
                # For functions returning int, always return the exit code
                if returns_int:
                    return cast(T, exit_code)
                
                # For other return types, check if we're in a pytest environment
                if "pytest" in sys.modules and sys.modules["pytest"] is not None:
                    # In pytest environment, re-raise the exception for pytest to catch
                    raise
                
                # Not in pytest environment, exit the program
                sys.exit(exit_code)

        return wrapper
