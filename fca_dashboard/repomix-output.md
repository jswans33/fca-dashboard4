This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

## Additional Info

# Directory Structure
```
.repomixignore
config/__init__.py
config/settings.py
config/settings.yml
main.py
repomix.config.json
tests/conftest.py
tests/unit/test_error_handler.py
tests/unit/test_logging_config.py
tests/unit/test_main.py
tests/unit/test_path_util.py
tests/unit/test_settings.py
utils/error_handler.py
utils/logging_config.py
utils/loguru_stubs.pyi
utils/path_util.py
```

# Files

## File: .repomixignore
```
# Add patterns to ignore here, one per line
# Example:
# *.log
# tmp/
```

## File: config/__init__.py
```python
"""
Configuration package for the FCA Dashboard application.

This package contains modules for loading and managing application configuration.
"""
```

## File: config/settings.py
```python
"""
Configuration module for loading and accessing application settings.

This module provides functionality to load settings from YAML configuration files
and access them in a structured way throughout the application.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Settings:
    """
    Settings class for loading and accessing application configuration.

    This class provides methods to load settings from YAML files and access
    them through a simple interface.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Settings with optional config path.

        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
        """
        self.config: Dict[str, Any] = {}
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "settings.yml"
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        config_path = self.config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Cache for settings instances
_settings_cache: Dict[str, Settings] = {}
# Create a default settings instance
settings = Settings()


def get_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """
    Get a Settings instance, with caching for repeated calls.

    Args:
        config_path: Optional path to a configuration file

    Returns:
        A Settings instance
    """
    if config_path is None:
        return settings

    # Convert to string for dictionary key
    cache_key = str(config_path)

    # Return cached instance if available
    if cache_key in _settings_cache:
        return _settings_cache[cache_key]

    # Create new instance and cache it
    new_settings = Settings(config_path)
    _settings_cache[cache_key] = new_settings
    return new_settings
```

## File: config/settings.yml
```yaml
# Database settings
databases:
  sqlite:
    url: "sqlite:///fca_dashboard.db"
  postgresql:
    url: "postgresql://user:password@localhost/fca_dashboard"
    
# Pipeline settings
pipeline_settings:
  batch_size: 5000
  log_level: "INFO"
  
# Table mappings
tables:
  equipment:
    mapping_type: "direct"
    column_mappings:
      tag: "Tag"
      name: "Name"
      description: "Description"
```

## File: main.py
```python
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
        raise ConfigurationError(f"YAML configuration error: {yaml_err}")

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
        try:
            excel_path = resolve_path(args.excel_file)
            log.info(f"Would process Excel file: {excel_path}")
        except FileNotFoundError:
            raise DataExtractionError(f"Excel file not found: {args.excel_file}")

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
```

## File: repomix.config.json
```json
{
  "output": {
    "filePath": "repomix-output.md",
    "style": "markdown",
    "parsableStyle": false,
    "fileSummary": true,
    "directoryStructure": true,
    "removeComments": false,
    "removeEmptyLines": false,
    "compress": false,
    "topFilesLength": 5,
    "showLineNumbers": false,
    "copyToClipboard": false
  },
  "include": [],
  "ignore": {
    "useGitignore": true,
    "useDefaultPatterns": true,
    "customPatterns": []
  },
  "security": {
    "enableSecurityCheck": true
  },
  "tokenCount": {
    "encoding": "o200k_base"
  }
}
```

## File: tests/conftest.py
```python
"""
Pytest configuration file.

This file contains shared fixtures and configuration for pytest.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
# This ensures that the tests can import modules from the project
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

## File: tests/unit/test_error_handler.py
```python
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
```

## File: tests/unit/test_logging_config.py
```python
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
```

## File: tests/unit/test_main.py
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from fca_dashboard.main import main, parse_args, run_etl_pipeline
from fca_dashboard.utils.error_handler import ConfigurationError, DataExtractionError


@patch("sys.argv", ["main.py", "--config", "config/settings.yml"])
def test_main_runs_successfully() -> None:
    """Test that the main function runs successfully with default arguments."""
    exit_code = main()
    assert exit_code == 0


def test_parse_args_defaults() -> None:
    """Test that parse_args returns expected defaults."""
    with patch("sys.argv", ["main.py"]):
        args = parse_args()
        assert "settings.yml" in args.config
        assert args.log_level == "INFO"
        assert args.excel_file is None
        assert args.table_name is None


def test_parse_args_custom_values() -> None:
    """Test that parse_args handles custom arguments correctly."""
    with patch(
        "sys.argv",
        [
            "main.py",
            "--config",
            "custom_config.yml",
            "--log-level",
            "DEBUG",
            "--excel-file",
            "data.xlsx",
            "--table-name",
            "equipment",
        ],
    ):
        args = parse_args()
        assert args.config == "custom_config.yml"
        assert args.log_level == "DEBUG"
        assert args.excel_file == "data.xlsx"
        assert args.table_name == "equipment"


@patch("fca_dashboard.main.get_settings")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.resolve_path")
def test_main_with_excel_file_and_table(
    mock_resolve_path: MagicMock,
    mock_get_logger: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_settings: MagicMock,
) -> None:
    """Test main function with excel_file and table_name arguments."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_settings = MagicMock()
    mock_get_settings.return_value = mock_settings
    mock_settings.get.return_value = "sqlite:///test.db"
    mock_resolve_path.side_effect = lambda x: Path(f"/resolved/{x}")

    # Run with excel file and table name
    with patch(
        "sys.argv",
        ["main.py", "--config", "config/settings.yml", "--excel-file", "data.xlsx", "--table-name", "equipment"],
    ):
        exit_code = main()

    # Verify
    assert exit_code == 0
    assert mock_logger.info.call_count >= 5  # Multiple info logs
    # Check that excel file and table name were logged
    mock_resolve_path.assert_any_call("data.xlsx")
    # Check that the log message contains the excel file path (exact format may vary by OS)
    excel_log_found = False
    for call_args in mock_logger.info.call_args_list:
        if "Would process Excel file:" in call_args[0][0] and "data.xlsx" in call_args[0][0]:
            excel_log_found = True
            break
    assert excel_log_found, "Excel file log message not found"
    mock_logger.info.assert_any_call("Would process table: equipment")


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
@patch("fca_dashboard.main.ErrorHandler")
def test_main_file_not_found_error(
    mock_error_handler_class: MagicMock,
    mock_resolve_path: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_logger: MagicMock,
) -> None:
    """Test main function handling FileNotFoundError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("nonexistent_file.yml")
    
    # Setup error handler mock
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 1
    mock_error_handler_class.return_value = mock_error_handler

    # Simulate FileNotFoundError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=FileNotFoundError("File not found")),
        patch("sys.argv", ["main.py", "--config", "nonexistent_file.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 1
    mock_error_handler.handle_error.assert_called_once()
    # Verify the error passed to handle_error is a FileNotFoundError
    args, _ = mock_error_handler.handle_error.call_args
    assert isinstance(args[0], FileNotFoundError)


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
@patch("fca_dashboard.main.ErrorHandler")
def test_main_yaml_error(
    mock_error_handler_class: MagicMock,
    mock_resolve_path: MagicMock,
    mock_configure_logging: MagicMock,
    mock_get_logger: MagicMock,
) -> None:
    """Test main function handling YAMLError."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("invalid_yaml.yml")
    
    # Setup error handler mock
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 2  # ConfigurationError code
    mock_error_handler_class.return_value = mock_error_handler

    # Simulate YAMLError when trying to load settings
    with (
        patch("fca_dashboard.main.get_settings", side_effect=yaml.YAMLError("Invalid YAML")),
        patch("sys.argv", ["main.py", "--config", "invalid_yaml.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 2  # ConfigurationError code
    mock_error_handler.handle_error.assert_called_once()
    # Verify the error passed to handle_error is a ConfigurationError
    args, _ = mock_error_handler.handle_error.call_args
    assert isinstance(args[0], ConfigurationError)
    assert "YAML configuration error" in str(args[0])


@patch("fca_dashboard.main.get_logger")
@patch("fca_dashboard.main.configure_logging")
@patch("fca_dashboard.main.resolve_path")
def test_main_unexpected_error(
    mock_resolve_path: MagicMock, mock_configure_logging: MagicMock, mock_get_logger: MagicMock
) -> None:
    """Test main function handling unexpected exceptions."""
    # Setup mocks
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_resolve_path.return_value = Path("config.yml")

    # Simulate unexpected exception
    with (
        patch("fca_dashboard.main.get_settings", side_effect=Exception("Unexpected error")),
        patch("sys.argv", ["main.py", "--config", "config.yml"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 99  # Generic error code from ErrorHandler
    # The error is now handled by ErrorHandler, not directly in main


def test_run_etl_pipeline_success() -> None:
    """Test that run_etl_pipeline runs successfully with valid arguments."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "config/settings.yml"
    mock_args.excel_file = None
    mock_args.table_name = None
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", return_value=Path("config/settings.yml")),
        patch("fca_dashboard.main.get_settings", return_value={"databases.sqlite.url": "sqlite:///test.db"}),
    ):
        exit_code = run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert exit_code == 0
    assert mock_log.info.call_count >= 4  # Multiple info logs


def test_run_etl_pipeline_configuration_error() -> None:
    """Test that run_etl_pipeline raises ConfigurationError for YAML errors."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "invalid_config.yml"
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", return_value=Path("invalid_config.yml")),
        patch("fca_dashboard.main.get_settings", side_effect=yaml.YAMLError("Invalid YAML")),
        pytest.raises(ConfigurationError) as exc_info,
    ):
        run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert "YAML configuration error" in str(exc_info.value)


def test_run_etl_pipeline_excel_file_not_found() -> None:
    """Test that run_etl_pipeline raises DataExtractionError for missing Excel files."""
    # Setup mocks
    mock_args = MagicMock()
    mock_args.config = "config/settings.yml"
    mock_args.excel_file = "nonexistent.xlsx"
    mock_args.table_name = None
    mock_log = MagicMock()

    with (
        patch("fca_dashboard.main.resolve_path", side_effect=[
            Path("config/settings.yml"),  # First call for config file
            FileNotFoundError("Excel file not found"),  # Second call for Excel file
        ]),
        patch("fca_dashboard.main.get_settings", return_value={"databases.sqlite.url": "sqlite:///test.db"}),
        pytest.raises(DataExtractionError) as exc_info,
    ):
        run_etl_pipeline(mock_args, mock_log)

    # Verify
    assert "Excel file not found" in str(exc_info.value)


def test_main_with_error_handler() -> None:
    """Test that main uses ErrorHandler to handle exceptions."""
    # Setup mocks
    mock_error_handler = MagicMock()
    mock_error_handler.handle_error.return_value = 42

    with (
        patch("fca_dashboard.main.ErrorHandler", return_value=mock_error_handler),
        patch("fca_dashboard.main.run_etl_pipeline", side_effect=Exception("Test error")),
        patch("fca_dashboard.main.configure_logging"),
        patch("fca_dashboard.main.get_logger"),
        patch("sys.argv", ["main.py"]),
    ):
        exit_code = main()

    # Verify
    assert exit_code == 42
    mock_error_handler.handle_error.assert_called_once()
```

## File: tests/unit/test_path_util.py
```python
"""
Unit tests for the path utility module.

This module contains tests for the path utility functions in the
fca_dashboard.utils.path_util module.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fca_dashboard.utils.path_util import get_config_path, get_logs_path, get_root_dir, resolve_path


def test_get_root_dir() -> None:
    """Test that get_root_dir returns a Path object to the project root."""
    root_dir = get_root_dir()
    assert isinstance(root_dir, Path)
    # Check that the directory exists
    assert root_dir.exists()
    # Check that it contains expected project files/directories
    assert (root_dir / "fca_dashboard").exists()
    assert (root_dir / "setup.py").exists() or (root_dir / "pyproject.toml").exists()


def test_get_config_path_default() -> None:
    """Test get_config_path with default filename."""
    config_path = get_config_path()
    assert isinstance(config_path, Path)
    assert config_path.name == "settings.yml"
    # Use os.path.join to handle platform-specific path separators
    assert os.path.join("fca_dashboard", "config") in str(config_path)


def test_get_config_path_custom() -> None:
    """Test get_config_path with custom filename."""
    custom_filename = "custom_settings.yml"
    config_path = get_config_path(custom_filename)
    assert isinstance(config_path, Path)
    assert config_path.name == custom_filename
    # Use os.path.join to handle platform-specific path separators
    assert os.path.join("fca_dashboard", "config") in str(config_path)


@patch("fca_dashboard.utils.path_util.logger")
def test_get_config_path_nonexistent(mock_logger: Any) -> None:
    """Test get_config_path with a nonexistent file."""
    nonexistent_file = "nonexistent_file.yml"
    config_path = get_config_path(nonexistent_file)
    assert isinstance(config_path, Path)
    assert config_path.name == nonexistent_file
    # Check that a warning was logged
    mock_logger.warning.assert_called_once()


def test_get_logs_path_default() -> None:
    """Test get_logs_path with default filename."""
    logs_path = get_logs_path()
    assert isinstance(logs_path, Path)
    assert logs_path.name == "fca_dashboard.log"
    assert "logs" in str(logs_path)
    # Check that the logs directory exists
    assert logs_path.parent.exists()


def test_get_logs_path_custom() -> None:
    """Test get_logs_path with custom filename."""
    custom_filename = "custom.log"
    logs_path = get_logs_path(custom_filename)
    assert isinstance(logs_path, Path)
    assert logs_path.name == custom_filename
    assert "logs" in str(logs_path)


def test_resolve_path_absolute() -> None:
    """Test resolve_path with an absolute path."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Test with absolute path
        resolved_path = resolve_path(temp_path)
        assert resolved_path == temp_path
    finally:
        # Clean up
        os.unlink(temp_path)


def test_resolve_path_existing_relative() -> None:
    """Test resolve_path with an existing relative path."""
    # Create a temporary file in the current directory
    with tempfile.NamedTemporaryFile(dir=".", delete=False) as temp_file:
        temp_name = Path(temp_file.name).name

    try:
        # Test with relative path that exists
        resolved_path = resolve_path(temp_name)
        assert resolved_path.is_absolute()
        assert resolved_path.name == temp_name
    finally:
        # Clean up
        os.unlink(temp_name)


@patch("fca_dashboard.utils.path_util.logger")
def test_resolve_path_nonexistent(mock_logger: Any) -> None:
    """Test resolve_path with a nonexistent path."""
    nonexistent_path = "nonexistent_file.txt"
    resolved_path = resolve_path(nonexistent_path)
    assert isinstance(resolved_path, Path)
    assert resolved_path.name == nonexistent_path
    # Check that a debug message was logged
    mock_logger.debug.assert_called()


def test_resolve_path_with_base_dir() -> None:
    """Test resolve_path with a base directory."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file in the temporary directory
        temp_file_path = Path(temp_dir) / "test_file.txt"
        with open(temp_file_path, "w") as f:
            f.write("test")

        # Test resolving the file relative to the base directory
        resolved_path = resolve_path("test_file.txt", base_dir=Path(temp_dir))
        assert resolved_path.is_absolute()
        assert resolved_path.name == "test_file.txt"
        assert resolved_path.parent == Path(temp_dir).resolve()


def test_resolve_path_with_fca_dashboard_subdir() -> None:
    """Test resolve_path with a path in the fca_dashboard subdirectory."""
    # Mock a base directory with an fca_dashboard subdirectory
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        fca_dir = base_dir / "fca_dashboard"
        fca_dir.mkdir()

        # Create a file in the fca_dashboard subdirectory
        test_file = fca_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test")

        # Test resolving the file
        resolved_path = resolve_path("test_file.txt", base_dir=base_dir)
        assert resolved_path.is_absolute()
        assert resolved_path.name == "test_file.txt"
        assert resolved_path.parent == fca_dir.resolve()
```

## File: tests/unit/test_settings.py
```python
"""
Unit tests for the Settings module.

This module contains tests for the Settings class and related functionality
in the fca_dashboard.config.settings module.
"""

import os
import tempfile
from typing import Generator

import pytest

from fca_dashboard.config.settings import Settings, get_settings


@pytest.fixture
def temp_settings_file() -> Generator[str, None, None]:
    """Create a temporary settings file for testing."""
    config_content = """
database:
  host: localhost
  port: 5432
  user: test_user
  password: secret
app:
  name: test_app
  debug: true
"""
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
        temp_file.write(config_content.encode("utf-8"))
        temp_path = temp_file.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_settings_load_valid_file(temp_settings_file: str) -> None:
    """Test loading settings from a valid file."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("database.host") == "localhost"
    assert settings.get("database.port") == 5432
    assert settings.get("app.name") == "test_app"
    assert settings.get("app.debug") is True


def test_settings_load_missing_file() -> None:
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Settings(config_path="nonexistent_file.yml")


def test_settings_get_nonexistent_key(temp_settings_file: str) -> None:
    """Test getting a non-existent key returns the default value."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("nonexistent.key") is None
    assert settings.get("nonexistent.key", default="fallback") == "fallback"


def test_settings_get_nested_keys(temp_settings_file: str) -> None:
    """Test getting nested keys from the configuration."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("database.user") == "test_user"
    assert settings.get("database.password") == "secret"


def test_get_settings_caching(temp_settings_file: str) -> None:
    """Test that get_settings caches instances for the same config path."""
    settings1 = get_settings(temp_settings_file)
    settings2 = get_settings(temp_settings_file)

    # Should be the same instance
    assert settings1 is settings2

    # Modify the first instance and check that the second reflects the change
    settings1.config["test_key"] = "test_value"
    assert settings2.config["test_key"] == "test_value"


def test_get_settings_default() -> None:
    """Test that get_settings returns the default instance when no path is provided."""
    settings = get_settings()
    assert isinstance(settings, Settings)

    # Should return the same default instance on subsequent calls
    settings2 = get_settings()
    assert settings is settings2
```

## File: utils/error_handler.py
```python
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
```

## File: utils/logging_config.py
```python
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
```

## File: utils/loguru_stubs.pyi
```
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

class Logger:
    def remove(self, handler_id: Optional[int] = None) -> None: ...
    def add(
        self,
        sink: Union[TextIO, str, Callable, Dict[str, Any]],
        *,
        level: Optional[Union[str, int]] = None,
        format: Optional[Union[str, Callable[[Dict[str, Any]], str]]] = None,
        filter: Optional[Union[str, Callable, Dict[str, Any]]] = None,
        colorize: Optional[bool] = None,
        serialize: Optional[bool] = None,
        backtrace: Optional[bool] = None,
        diagnose: Optional[bool] = None,
        enqueue: Optional[bool] = None,
        catch: Optional[bool] = None,
        rotation: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        retention: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        compression: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
        delay: Optional[bool] = None,
        mode: Optional[str] = None,
        buffering: Optional[int] = None,
        encoding: Optional[str] = None,
        **kwargs: Any,
    ) -> int: ...
    def bind(self, **kwargs: Any) -> "Logger": ...
    def opt(
        self,
        *,
        exception: Optional[Union[bool, Tuple[Any, ...], Dict[str, Any]]] = None,
        record: Optional[bool] = None,
        lazy: Optional[bool] = None,
        colors: Optional[bool] = None,
        raw: Optional[bool] = None,
        capture: Optional[bool] = None,
        depth: Optional[int] = None,
        ansi: Optional[bool] = None,
    ) -> "Logger": ...
    def trace(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def debug(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def info(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def success(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def warning(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def error(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def critical(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def exception(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def log(self, level: Union[int, str], __message: Any, *args: Any, **kwargs: Any) -> None: ...
    def level(self, name: str, no: int = 0, color: Optional[str] = None, icon: Optional[str] = None) -> "Logger": ...
    def disable(self, name: str) -> None: ...
    def enable(self, name: str) -> None: ...
    def configure(
        self,
        *,
        handlers: List[Dict[str, Any]] = [],
        levels: List[Dict[str, Any]] = [],
        extra: Dict[str, Any] = {},
        patcher: Optional[Callable] = None,
        activation: List[Tuple[str, bool]] = [],
    ) -> None: ...
    def patch(self, patcher: Callable) -> "Logger": ...
    def complete(self) -> None: ...
    @property
    def catch(self) -> Callable: ...

logger: Logger
```

## File: utils/path_util.py
```python
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def get_root_dir() -> Path:
    """Return the project's root directory (assuming this module is within project)."""
    return Path(__file__).resolve().parents[2]


def get_config_path(filename: str = "settings.yml") -> Path:
    """Get absolute path to the config file, ensuring it exists."""
    config_path = get_root_dir() / "fca_dashboard" / "config" / filename
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
    return config_path


def get_logs_path(filename: str = "fca_dashboard.log") -> Path:
    """Get absolute path to the log file, ensuring the logs directory exists."""
    logs_dir = get_root_dir() / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)
    return logs_dir / filename


def resolve_path(path: Union[str, Path], base_dir: Union[Path, None] = None) -> Path:
    """
    Resolve a path relative to the base directory or project root.

    Args:
        path: The path to resolve.
        base_dir: Optional base directory; defaults to the project's root directory.

    Returns:
        Resolved Path object.
    """
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    if path_obj.exists():
        return path_obj.resolve()

    if base_dir is None:
        base_dir = get_root_dir()

    candidate_paths = [
        base_dir / path_obj,
        base_dir / "fca_dashboard" / path_obj,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            logger.debug(f"Resolved path '{path}' to '{candidate.resolve()}'")
            return candidate.resolve()

    logger.debug(f"Could not resolve path: {path_obj}. Returning unresolved path.")
    return path_obj
```
