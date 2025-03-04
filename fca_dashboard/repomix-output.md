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
tests/unit/test_date_utils.py
tests/unit/test_error_handler.py
tests/unit/test_json_util.py
tests/unit/test_logging_config.py
tests/unit/test_main.py
tests/unit/test_number_utils.py
tests/unit/test_path_util.py
tests/unit/test_settings.py
tests/unit/test_string_utils.py
tests/unit/test_validation_utils.py
utils/__init__.py
utils/date_utils.py
utils/error_handler.py
utils/json_utils.py
utils/logging_config.py
utils/loguru_stubs.pyi
utils/number_utils.py
utils/path_util.py
utils/string_utils.py
utils/validation_utils.py
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

import os
import re
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
            
        # Process environment variable substitutions
        self._process_env_vars(self.config)
        
    def _process_env_vars(self, config_section: Any) -> None:
        """
        Recursively process environment variable substitutions in the config.
        
        Args:
            config_section: A section of the configuration to process
        """
        if isinstance(config_section, dict):
            for key, value in config_section.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[key] = self._substitute_env_vars(value)
        elif isinstance(config_section, list):
            for i, value in enumerate(config_section):
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[i] = self._substitute_env_vars(value)
    
    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: The string value to process
            
        Returns:
            The string with environment variables substituted
        """
        # Match ${VAR_NAME} pattern
        pattern = r'\${([A-Za-z0-9_]+)}'
        
        def replace_env_var(match):
            env_var_name = match.group(1)
            env_var_value = os.environ.get(env_var_name)
            if env_var_value is None:
                # If environment variable is not set, keep the original placeholder
                return match.group(0)
            return env_var_value
            
        return re.sub(pattern, replace_env_var, value)

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
# Consider thread-safety if accessed from multiple threads
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
    url: "${POSTGRES_URL}"
    
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
        raise ConfigurationError(f"YAML configuration error: {yaml_err}") from yaml_err

    # Log startup information
    log.info("FCA Dashboard ETL Pipeline starting")
    log.info(f"Python version: {sys.version}")
    log.info(f"Current working directory: {Path.cwd()}")

    # TODO: Implement ETL pipeline execution (See GitHub issue #123)
    # Steps include:
    # 1. Extract data from Excel or database source (See GitHub issue #124)
    #    - Read source data using appropriate extractor strategy
    #    - Validate source data structure
    # 2. Transform data (cleaning, normalization, enrichment) (See GitHub issue #125)
    #    - Apply business rules and transformations
    #    - Map source fields to destination schema
    # 3. Load data into destination database or output format (See GitHub issue #126)
    #    - Batch insert/update operations
    #    - Validate data integrity after loading
    log.info("ETL Pipeline execution would start here")

    log.info(f"Database URL: {settings.get('databases.sqlite.url')}")

    if args.excel_file:
        try:
            excel_path = resolve_path(args.excel_file)
            log.info(f"Would process Excel file: {excel_path}")
        except FileNotFoundError:
            raise DataExtractionError(f"Excel file not found: {args.excel_file}") from None

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
    "topFilesLength": 15,
    "showLineNumbers": false,
    "copyToClipboard": true
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

## File: tests/unit/test_date_utils.py
```python
"""Tests for date and time utility functions."""
import datetime

import pytest
from freezegun import freeze_time

from fca_dashboard.utils.date_utils import format_date, parse_date, time_since


class TestFormatDate:
    """Tests for the format_date function."""

    def test_format_date_default(self):
        """Test formatting a date with default format."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date) == "May 15, 2023"

    def test_format_date_custom_format(self):
        """Test formatting a date with a custom format."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date, "%Y-%m-%d") == "2023-05-15"

    def test_format_date_with_time(self):
        """Test formatting a date with time."""
        date = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert format_date(date, "%b %d, %Y %H:%M") == "May 15, 2023 14:30"

    def test_format_date_none(self):
        """Test formatting None date."""
        assert format_date(None) == ""

    def test_format_date_with_default_value(self):
        """Test formatting None date with a default value."""
        assert format_date(None, default="N/A") == "N/A"


class TestTimeSince:
    """Tests for the time_since function."""

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_seconds(self):
        """Test time since for seconds."""
        date = datetime.datetime(2023, 5, 15, 14, 29, 30)
        assert time_since(date) == "30 seconds ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_minute(self):
        """Test time since for a minute."""
        date = datetime.datetime(2023, 5, 15, 14, 29, 0)
        assert time_since(date) == "1 minute ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_minutes(self):
        """Test time since for minutes."""
        date = datetime.datetime(2023, 5, 15, 14, 25, 0)
        assert time_since(date) == "5 minutes ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_hour(self):
        """Test time since for an hour."""
        date = datetime.datetime(2023, 5, 15, 13, 30, 0)
        assert time_since(date) == "1 hour ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_hours(self):
        """Test time since for hours."""
        date = datetime.datetime(2023, 5, 15, 10, 30, 0)
        assert time_since(date) == "4 hours ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_day(self):
        """Test time since for a day."""
        date = datetime.datetime(2023, 5, 14, 14, 30, 0)
        assert time_since(date) == "1 day ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_days(self):
        """Test time since for days."""
        date = datetime.datetime(2023, 5, 10, 14, 30, 0)
        assert time_since(date) == "5 days ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_month(self):
        """Test time since for a month."""
        date = datetime.datetime(2023, 4, 15, 14, 30, 0)
        assert time_since(date) == "1 month ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_months(self):
        """Test time since for months."""
        date = datetime.datetime(2023, 1, 15, 14, 30, 0)
        assert time_since(date) == "4 months ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_year(self):
        """Test time since for a year."""
        date = datetime.datetime(2022, 5, 15, 14, 30, 0)
        assert time_since(date) == "1 year ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_years(self):
        """Test time since for years."""
        date = datetime.datetime(2020, 5, 15, 14, 30, 0)
        assert time_since(date) == "3 years ago"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_future(self):
        """Test time since for a future date."""
        date = datetime.datetime(2023, 5, 16, 14, 30, 0)
        assert time_since(date) == "in 1 day"

    @freeze_time("2023-05-15 14:30:00")
    def test_time_since_none(self):
        """Test time since for None date."""
        assert time_since(None) == ""


class TestParseDate:
    """Tests for the parse_date function."""

    def test_parse_date_iso_format(self):
        """Test parsing a date in ISO format."""
        assert parse_date("2023-05-15") == datetime.datetime(2023, 5, 15, 0, 0, 0)

    def test_parse_date_with_time(self):
        """Test parsing a date with time."""
        assert parse_date("2023-05-15 14:30:00") == datetime.datetime(2023, 5, 15, 14, 30, 0)

    def test_parse_date_custom_format(self):
        """Test parsing a date with a custom format."""
        assert parse_date("15/05/2023", format="%d/%m/%Y") == datetime.datetime(2023, 5, 15, 0, 0, 0)

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_yesterday(self):
        """Test parsing 'yesterday'."""
        expected = datetime.datetime(2023, 5, 14, 0, 0, 0)
        assert parse_date("yesterday") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_today(self):
        """Test parsing 'today'."""
        expected = datetime.datetime(2023, 5, 15, 0, 0, 0)
        assert parse_date("today") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_tomorrow(self):
        """Test parsing 'tomorrow'."""
        expected = datetime.datetime(2023, 5, 16, 0, 0, 0)
        assert parse_date("tomorrow") == expected

    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_days_ago(self):
        """Test parsing 'X days ago'."""
        expected = datetime.datetime(2023, 5, 10, 0, 0, 0)
        assert parse_date("5 days ago") == expected
        
    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_invalid_days_ago_format(self):
        """Test parsing an invalid 'X days ago' format."""
        # This should fall through to the dateutil parser and raise ValueError
        with pytest.raises(ValueError):
            parse_date("invalid days ago")
            
    @freeze_time("2023-05-15 14:30:00")
    def test_parse_date_empty_days_ago_format(self):
        """Test parsing an empty 'days ago' format."""
        # This should fall through to the dateutil parser and raise ValueError
        with pytest.raises(ValueError):
            parse_date(" days ago")

    def test_parse_date_datetime_object(self):
        """Test parsing a datetime object."""
        dt = datetime.datetime(2023, 5, 15, 14, 30, 0)
        assert parse_date(dt) is dt

    def test_parse_date_invalid(self):
        """Test parsing an invalid date."""
        with pytest.raises(ValueError):
            parse_date("not a date")

    def test_parse_date_none(self):
        """Test parsing None."""
        assert parse_date(None) is None

    def test_parse_date_empty(self):
        """Test parsing an empty string."""
        assert parse_date("") is None
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

## File: tests/unit/test_json_util.py
```python
import os
import tempfile

import pytest

from fca_dashboard.utils.json_utils import (
    json_deserialize,
    json_is_valid,
    json_load,
    json_save,
    json_serialize,
    pretty_print_json,
    safe_get,
    safe_get_nested,
)


def test_json_load_and_save():
    data = {"key": "value", "number": 42}
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        path = tmp.name
        json_save(data, path)

    loaded_data = json_load(path)
    assert loaded_data == data

    os.unlink(path)


def test_json_serialize():
    data = {"name": "Alice", "age": 30}
    json_str = json_serialize(data)
    assert json_str == '{"name": "Alice", "age": 30}'


def test_json_deserialize_valid():
    json_str = '{"valid": true, "value": 10}'
    result = json_deserialize(json_str)
    assert result == {"valid": True, "value": 10}


def test_json_deserialize_invalid():
    json_str = '{invalid json}'
    default = {"default": True}
    result = json_deserialize(json_str, default=default)
    assert result == default


def test_json_is_valid():
    assert json_is_valid('{"valid": true}') is True
    assert json_is_valid('{invalid json}') is False


def test_pretty_print_json():
    data = {"key": "value"}
    expected = '{\n  "key": "value"\n}'
    assert pretty_print_json(data) == expected


def test_safe_get():
    data = {"a": 1, "b": None}
    assert safe_get(data, "a") == 1
    assert safe_get(data, "b", default="default") is None
    assert safe_get(data, "missing", default="default") == "default"


def test_safe_get_nested():
    data = {"a": {"b": {"c": 42}}}
    assert safe_get_nested(data, "a", "b", "c") == 42
    assert safe_get_nested(data, "a", "x", default="missing") == "missing"
    assert safe_get_nested(data, "a", "b", "c", "d", default=None) is None


def test_json_load_file_not_found():
    with pytest.raises(FileNotFoundError):
        json_load("nonexistent_file.json")


def test_json_load_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("{invalid json}")
        path = tmp.name

    with pytest.raises(Exception):
        json_load(path)

    os.unlink(path)


def test_json_save_and_load_unicode():
    data = {"message": "こんにちは世界"}
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        path = tmp.name
        json_save(data, path)

    loaded_data = json_load(path)
    assert loaded_data == data

    os.unlink(path)
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

## File: tests/unit/test_number_utils.py
```python
"""Unit tests for number utilities."""
import re
from decimal import Decimal

import pytest

from fca_dashboard.utils.number_utils import format_currency, random_number, round_to


class TestFormatCurrency:
    """Test cases for currency formatting function."""

    def test_integer_values(self):
        """Test formatting integer values as currency."""
        assert format_currency(1234) == "$1,234.00"
        assert format_currency(0) == "$0.00"
        assert format_currency(-1234) == "-$1,234.00"

    def test_float_values(self):
        """Test formatting float values as currency."""
        assert format_currency(1234.56) == "$1,234.56"
        assert format_currency(1234.5) == "$1,234.50"
        assert format_currency(0.99) == "$0.99"
        assert format_currency(-1234.56) == "-$1,234.56"

    def test_decimal_values(self):
        """Test formatting Decimal values as currency."""
        assert format_currency(Decimal("1234.56")) == "$1,234.56"
        assert format_currency(Decimal("1234.5")) == "$1,234.50"
        assert format_currency(Decimal("0.99")) == "$0.99"
        assert format_currency(Decimal("-1234.56")) == "-$1,234.56"

    def test_custom_currency_symbol(self):
        """Test formatting with custom currency symbols."""
        assert format_currency(1234.56, symbol="€") == "€1,234.56"
        assert format_currency(1234.56, symbol="£") == "£1,234.56"
        assert format_currency(1234.56, symbol="¥") == "¥1,234.56"
        assert format_currency(1234.56, symbol="") == "1,234.56"

    def test_custom_decimal_places(self):
        """Test formatting with custom decimal places."""
        assert format_currency(1234.56, decimal_places=0) == "$1,235"
        assert format_currency(1234.56, decimal_places=1) == "$1,234.6"
        assert format_currency(1234.56, decimal_places=3) == "$1,234.560"
        assert format_currency(1234.56789, decimal_places=4) == "$1,234.5679"

    def test_custom_thousands_separator(self):
        """Test formatting with custom thousands separator."""
        assert format_currency(1234567.89, thousands_sep=".") == "$1.234.567.89"
        assert format_currency(1234567.89, thousands_sep=" ") == "$1 234 567.89"
        assert format_currency(1234567.89, thousands_sep="") == "$1234567.89"

    def test_custom_decimal_separator(self):
        """Test formatting with custom decimal separator."""
        assert format_currency(1234.56, decimal_sep=",") == "$1,234,56"
        assert format_currency(1234.56, decimal_sep=" ") == "$1,234 56"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert format_currency(None) == ""
        assert format_currency(None, default="N/A") == "N/A"

    def test_non_numeric_input(self):
        """Test that non-numeric inputs are handled correctly."""
        with pytest.raises(TypeError):
            format_currency("not a number")
        with pytest.raises(TypeError):
            format_currency([])


class TestRoundTo:
    """Test cases for number rounding function."""

    def test_round_to_zero_places(self):
        """Test rounding to zero decimal places."""
        assert round_to(1.4, 0) == 1
        assert round_to(1.5, 0) == 2
        assert round_to(-1.5, 0) == -2
        assert round_to(0, 0) == 0

    def test_round_to_positive_places(self):
        """Test rounding to positive decimal places."""
        assert round_to(1.234, 2) == 1.23
        assert round_to(1.235, 2) == 1.24
        assert round_to(-1.235, 2) == -1.24
        assert round_to(1.2, 2) == 1.20

    def test_round_to_negative_places(self):
        """Test rounding to negative decimal places (tens, hundreds, etc.)."""
        assert round_to(123, -1) == 120
        assert round_to(125, -1) == 130
        assert round_to(1234, -2) == 1200
        assert round_to(1250, -2) == 1300
        assert round_to(-1250, -2) == -1300

    def test_round_decimal_type(self):
        """Test rounding Decimal objects."""
        assert round_to(Decimal("1.234"), 2) == Decimal("1.23")
        assert round_to(Decimal("1.235"), 2) == Decimal("1.24")
        assert round_to(Decimal("-1.235"), 2) == Decimal("-1.24")

    def test_return_type(self):
        """Test that the return type matches the input type."""
        assert isinstance(round_to(1.5, 0), int)
        assert isinstance(round_to(1.5, 1), float)
        assert isinstance(round_to(Decimal("1.5"), 1), Decimal)

    def test_none_input(self):
        """Test that None input is handled correctly."""
        with pytest.raises(TypeError):
            round_to(None, 2)

    def test_non_numeric_input(self):
        """Test that non-numeric inputs are handled correctly."""
        with pytest.raises(TypeError):
            round_to("not a number", 2)
        with pytest.raises(TypeError):
            round_to([], 2)


class TestRandomNumber:
    """Test cases for random number generation function."""

    def test_within_range(self):
        """Test that generated numbers are within the specified range."""
        for _ in range(100):  # Run multiple times to increase confidence
            num = random_number(1, 10)
            assert 1 <= num <= 10

    def test_min_equals_max(self):
        """Test when min equals max."""
        assert random_number(5, 5) == 5

    def test_negative_range(self):
        """Test with negative numbers in the range."""
        for _ in range(100):
            num = random_number(-10, -1)
            assert -10 <= num <= -1

    def test_mixed_range(self):
        """Test with a range that includes both negative and positive numbers."""
        for _ in range(100):
            num = random_number(-5, 5)
            assert -5 <= num <= 5

    def test_large_range(self):
        """Test with a large range."""
        for _ in range(10):
            num = random_number(-1000000, 1000000)
            assert -1000000 <= num <= 1000000

    def test_distribution(self):
        """Test that the distribution is roughly uniform."""
        # Generate a large number of random values between 1 and 10
        results = [random_number(1, 10) for _ in range(1000)]
        
        # Count occurrences of each value
        counts = {}
        for num in range(1, 11):
            counts[num] = results.count(num)
        
        # Check that each number appears roughly the expected number of times
        # (100 times each, with some tolerance for randomness)
        for num, count in counts.items():
            assert 70 <= count <= 130, f"Number {num} appeared {count} times, expected roughly 100"

    def test_invalid_range(self):
        """Test with invalid range (min > max)."""
        with pytest.raises(ValueError):
            random_number(10, 1)

    def test_non_integer_input(self):
        """Test with non-integer inputs."""
        with pytest.raises(TypeError):
            random_number(1.5, 10)
        with pytest.raises(TypeError):
            random_number(1, 10.5)
        with pytest.raises(TypeError):
            random_number("1", 10)
        with pytest.raises(TypeError):
            random_number(1, "10")
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
    # Check that a warning message was logged (changed from debug to warning)
    mock_logger.warning.assert_called()


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


@pytest.fixture
def temp_settings_file_with_env_vars() -> Generator[str, None, None]:
    """Create a temporary settings file with environment variable placeholders."""
    config_content = """
database:
  host: localhost
  port: 5432
  user: ${TEST_DB_USER}
  password: ${TEST_DB_PASSWORD}
app:
  name: test_app
  debug: true
  environments: ["dev", "${TEST_ENV}", "prod"]
  secrets: 
    - key1: value1
    - key2: ${TEST_SECRET}
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


def test_environment_variable_substitution(temp_settings_file_with_env_vars: str) -> None:
    """Test that environment variables are substituted in the configuration."""
    # Set environment variables for testing
    os.environ["TEST_DB_USER"] = "env_user"
    os.environ["TEST_DB_PASSWORD"] = "env_password"
    
    try:
        # Load settings with environment variables
        settings = Settings(config_path=temp_settings_file_with_env_vars)
        
        # Check that environment variables were substituted
        assert settings.get("database.user") == "env_user"
        assert settings.get("database.password") == "env_password"
        
        # Check that non-environment variable settings are still loaded correctly
        assert settings.get("database.host") == "localhost"
        assert settings.get("app.name") == "test_app"
    finally:
        # Clean up environment variables
        del os.environ["TEST_DB_USER"]
        del os.environ["TEST_DB_PASSWORD"]


def test_missing_environment_variable(temp_settings_file_with_env_vars: str) -> None:
    """Test that missing environment variables keep the original placeholder."""
    # Ensure environment variables are not set
    if "TEST_DB_USER" in os.environ:
        del os.environ["TEST_DB_USER"]
    if "TEST_DB_PASSWORD" in os.environ:
        del os.environ["TEST_DB_PASSWORD"]
    
    # Load settings with missing environment variables
    settings = Settings(config_path=temp_settings_file_with_env_vars)
    
    # Check that placeholders are preserved
    assert settings.get("database.user") == "${TEST_DB_USER}"
    assert settings.get("database.password") == "${TEST_DB_PASSWORD}"


def test_environment_variable_substitution_in_lists(temp_settings_file_with_env_vars: str) -> None:
    """Test that environment variables are substituted in lists and nested structures."""
    # Set environment variables for testing
    os.environ["TEST_ENV"] = "staging"
    os.environ["TEST_SECRET"] = "secret_value"
    
    try:
        # Load settings with environment variables
        settings = Settings(config_path=temp_settings_file_with_env_vars)
        
        # Check that environment variables in lists are substituted
        environments = settings.get("app.environments")
        assert isinstance(environments, list)
        assert environments[0] == "dev"
        assert environments[1] == "staging"  # Substituted from ${TEST_ENV}
        assert environments[2] == "prod"
        
        # Check that environment variables in nested structures are substituted
        secrets = settings.get("app.secrets")
        assert isinstance(secrets, list)
        assert secrets[0]["key1"] == "value1"
        assert secrets[1]["key2"] == "secret_value"  # Substituted from ${TEST_SECRET}
    finally:
        # Clean up environment variables
        del os.environ["TEST_ENV"]
        del os.environ["TEST_SECRET"]
```

## File: tests/unit/test_string_utils.py
```python
"""Tests for string utility functions."""
import pytest

from fca_dashboard.utils.string_utils import capitalize, is_empty, slugify, truncate


class TestCapitalize:
    """Tests for the capitalize function."""

    def test_capitalize_lowercase(self):
        """Test capitalizing a lowercase string."""
        assert capitalize("hello") == "Hello"

    def test_capitalize_already_capitalized(self):
        """Test capitalizing an already capitalized string."""
        assert capitalize("Hello") == "Hello"

    def test_capitalize_empty_string(self):
        """Test capitalizing an empty string."""
        assert capitalize("") == ""

    def test_capitalize_single_char(self):
        """Test capitalizing a single character."""
        assert capitalize("a") == "A"

    def test_capitalize_with_spaces(self):
        """Test capitalizing a string with leading spaces."""
        assert capitalize("  hello") == "  Hello"

    def test_capitalize_with_numbers(self):
        """Test capitalizing a string starting with numbers."""
        assert capitalize("123abc") == "123abc"


class TestSlugify:
    """Tests for the slugify function."""

    def test_slugify_simple_string(self):
        """Test slugifying a simple string."""
        assert slugify("Hello World") == "hello-world"

    def test_slugify_with_special_chars(self):
        """Test slugifying a string with special characters."""
        assert slugify("Hello, World!") == "hello-world"

    def test_slugify_with_multiple_spaces(self):
        """Test slugifying a string with multiple spaces."""
        assert slugify("Hello   World") == "hello-world"

    def test_slugify_with_dashes(self):
        """Test slugifying a string that already has dashes."""
        assert slugify("Hello-World") == "hello-world"

    def test_slugify_with_underscores(self):
        """Test slugifying a string with underscores."""
        assert slugify("Hello_World") == "hello-world"

    def test_slugify_empty_string(self):
        """Test slugifying an empty string."""
        assert slugify("") == ""

    def test_slugify_with_accents(self):
        """Test slugifying a string with accented characters."""
        assert slugify("Héllö Wörld") == "hello-world"


class TestTruncate:
    """Tests for the truncate function."""

    def test_truncate_short_string(self):
        """Test truncating a string shorter than the limit."""
        assert truncate("Hello", 10) == "Hello"

    def test_truncate_exact_length(self):
        """Test truncating a string of exact length."""
        assert truncate("Hello", 5) == "Hello"

    def test_truncate_long_string(self):
        """Test truncating a string longer than the limit."""
        assert truncate("Hello World", 5) == "Hello..."

    def test_truncate_with_custom_suffix(self):
        """Test truncating with a custom suffix."""
        assert truncate("Hello World", 5, suffix="...more") == "Hello...more"

    def test_truncate_empty_string(self):
        """Test truncating an empty string."""
        assert truncate("", 5) == ""

    def test_truncate_with_zero_length(self):
        """Test truncating with zero length."""
        assert truncate("Hello", 0) == "..."


class TestIsEmpty:
    """Tests for the is_empty function."""

    def test_is_empty_with_empty_string(self):
        """Test checking if an empty string is empty."""
        assert is_empty("") is True

    def test_is_empty_with_whitespace(self):
        """Test checking if a whitespace string is empty."""
        assert is_empty("   ") is True
        assert is_empty("\t\n") is True

    def test_is_empty_with_text(self):
        """Test checking if a non-empty string is empty."""
        assert is_empty("Hello") is False

    def test_is_empty_with_whitespace_and_text(self):
        """Test checking if a string with whitespace and text is empty."""
        assert is_empty("  Hello  ") is False

    def test_is_empty_with_none(self):
        """Test checking if None is empty."""
        with pytest.raises(TypeError):
            is_empty(None)
```

## File: tests/unit/test_validation_utils.py
```python
"""Unit tests for validation utilities."""
import pytest

from fca_dashboard.utils.validation_utils import is_valid_email, is_valid_phone, is_valid_url


class TestEmailValidation:
    """Test cases for email validation function."""

    def test_valid_emails(self):
        """Test that valid email addresses are correctly identified."""
        valid_emails = [
            "user@example.com",
            "user.name@example.com",
            "user+tag@example.com",
            "user-name@example.co.uk",
            "user_name@example-domain.com",
            "123456@example.com",
            "user@subdomain.example.com",
        ]
        for email in valid_emails:
            assert is_valid_email(email), f"Email should be valid: {email}"

    def test_invalid_emails(self):
        """Test that invalid email addresses are correctly rejected."""
        invalid_emails = [
            "",  # Empty string
            "user",  # Missing @ and domain
            "user@",  # Missing domain
            "@example.com",  # Missing username
            "user@.com",  # Missing domain name
            "user@example",  # Missing TLD
            "user@example..com",  # Double dot
            "user@example.com.",  # Trailing dot
            "user name@example.com",  # Space in username
            "user@exam ple.com",  # Space in domain
            "user@-example.com",  # Domain starts with hyphen
            "user@example-.com",  # Domain ends with hyphen
        ]
        for email in invalid_emails:
            assert not is_valid_email(email), f"Email should be invalid: {email}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_email(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_email(123), "Integer should be invalid"
        assert not is_valid_email(True), "Boolean should be invalid"
        assert not is_valid_email([]), "List should be invalid"


class TestPhoneValidation:
    """Test cases for phone number validation function."""

    def test_valid_phone_numbers(self):
        """Test that valid phone numbers are correctly identified."""
        valid_phones = [
            "1234567890",  # Simple 10-digit
            "123-456-7890",  # Hyphenated
            "(123) 456-7890",  # Parentheses
            "+1 123-456-7890",  # International format
            "123.456.7890",  # Dots
            "123 456 7890",  # Spaces
            "+12345678901",  # International without separators
        ]
        for phone in valid_phones:
            assert is_valid_phone(phone), f"Phone should be valid: {phone}"

    def test_invalid_phone_numbers(self):
        """Test that invalid phone numbers are correctly rejected."""
        invalid_phones = [
            "",  # Empty string
            "123",  # Too short
            "123456",  # Too short
            "abcdefghij",  # Letters
            "123-abc-7890",  # Mixed letters and numbers
            "123-456-789",  # Too short with separators
            "123-456-78901",  # Too long with separators
            "(123)456-7890",  # Missing space after parentheses
            "123 - 456 - 7890",  # Spaces around hyphens
        ]
        for phone in invalid_phones:
            assert not is_valid_phone(phone), f"Phone should be invalid: {phone}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_phone(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_phone(123), "Integer should be invalid"
        assert not is_valid_phone(True), "Boolean should be invalid"
        assert not is_valid_phone([]), "List should be invalid"


class TestUrlValidation:
    """Test cases for URL validation function."""

    def test_valid_urls(self):
        """Test that valid URLs are correctly identified."""
        valid_urls = [
            "http://example.com",
            "https://example.com",
            "http://www.example.com",
            "https://example.com/path",
            "https://example.com/path?query=value",
            "https://example.com/path#fragment",
            "https://example.com:8080",
            "https://subdomain.example.com",
            "https://example-domain.com",
            "https://example.co.uk",
            "http://localhost",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:8080",
        ]
        for url in valid_urls:
            assert is_valid_url(url), f"URL should be valid: {url}"

    def test_invalid_urls(self):
        """Test that invalid URLs are correctly rejected."""
        invalid_urls = [
            "",  # Empty string
            "example.com",  # Missing protocol
            "http://",  # Missing domain
            "http:/example.com",  # Missing slash
            "http://example",  # Missing TLD
            "http://.com",  # Missing domain name
            "http://example..com",  # Double dot
            "http://example.com.",  # Trailing dot
            "http://exam ple.com",  # Space in domain
            "http://-example.com",  # Domain starts with hyphen
            "http://example-.com",  # Domain ends with hyphen
            "htp://example.com",  # Typo in protocol
            "http:example.com",  # Missing slashes
            "http//example.com",  # Missing colon
        ]
        for url in invalid_urls:
            assert not is_valid_url(url), f"URL should be invalid: {url}"

    def test_none_input(self):
        """Test that None input is handled correctly."""
        assert not is_valid_url(None), "None should be invalid"

    def test_non_string_input(self):
        """Test that non-string inputs are handled correctly."""
        assert not is_valid_url(123), "Integer should be invalid"
        assert not is_valid_url(True), "Boolean should be invalid"
        assert not is_valid_url([]), "List should be invalid"
```

## File: utils/__init__.py
```python
"""Utility modules for the FCA Dashboard application."""

from fca_dashboard.utils.date_utils import *  # noqa
from fca_dashboard.utils.error_handler import *  # noqa
from fca_dashboard.utils.json_utils import *  # noqa
from fca_dashboard.utils.logging_config import *  # noqa
from fca_dashboard.utils.number_utils import (  # noqa
    format_currency,
    random_number,
    round_to,
)
from fca_dashboard.utils.path_util import *  # noqa
from fca_dashboard.utils.string_utils import *  # noqa
from fca_dashboard.utils.validation_utils import (  # noqa
    is_valid_email,
    is_valid_phone,
    is_valid_url,
)
```

## File: utils/date_utils.py
```python
"""
Date and time utility functions for common operations.

This module provides a collection of utility functions for date and time manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import datetime
from typing import Optional, Union

from dateutil import parser


def format_date(
    date: Optional[datetime.datetime], 
    format_str: str = "%b %d, %Y", 
    default: str = ""
) -> str:
    """
    Format a datetime object into a readable string.

    Args:
        date: The datetime object to format.
        format_str: The format string to use (default: "%b %d, %Y").
        default: The default value to return if date is None.

    Returns:
        A formatted date string or the default value if date is None.

    Examples:
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0))
        'May 15, 2023'
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0), "%Y-%m-%d")
        '2023-05-15'
    """
    if date is None:
        return default
    
    return date.strftime(format_str)


def time_since(date: Optional[datetime.datetime], default: str = "") -> str:
    """
    Calculate the relative time between the given date and now.

    Args:
        date: The datetime to calculate the time since.
        default: The default value to return if date is None.

    Returns:
        A human-readable string representing the time difference (e.g., "2 hours ago").

    Examples:
        >>> # Assuming current time is 2023-05-15 14:30:00
        >>> time_since(datetime.datetime(2023, 5, 15, 13, 30, 0))
        '1 hour ago'
        >>> time_since(datetime.datetime(2023, 5, 14, 14, 30, 0))
        '1 day ago'
    """
    if date is None:
        return default
    
    now = datetime.datetime.now()
    diff = now - date
    
    # Handle future dates
    if diff.total_seconds() < 0:
        diff = -diff
        is_future = True
    else:
        is_future = False
    
    seconds = int(diff.total_seconds())
    minutes = seconds // 60
    hours = minutes // 60
    days = diff.days
    months = days // 30  # Approximate
    years = days // 365  # Approximate
    
    if years > 0:
        time_str = f"{years} year{'s' if years != 1 else ''}"
    elif months > 0:
        time_str = f"{months} month{'s' if months != 1 else ''}"
    elif days > 0:
        time_str = f"{days} day{'s' if days != 1 else ''}"
    elif hours > 0:
        time_str = f"{hours} hour{'s' if hours != 1 else ''}"
    elif minutes > 0:
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
    
    return f"in {time_str}" if is_future else f"{time_str} ago"


def parse_date(
    date_str: Optional[Union[str, datetime.datetime]], 
    format: Optional[str] = None
) -> Optional[datetime.datetime]:
    """
    Convert a string into a datetime object.

    Args:
        date_str: The string to parse or a datetime object to return as-is.
        format: Optional format string for parsing (if None, tries to infer format).

    Returns:
        A datetime object or None if the input is None or empty.

    Raises:
        ValueError: If the string cannot be parsed as a date.

    Examples:
        >>> parse_date("2023-05-15")
        datetime.datetime(2023, 5, 15, 0, 0)
        >>> parse_date("15/05/2023", format="%d/%m/%Y")
        datetime.datetime(2023, 5, 15, 0, 0)
    """
    if date_str is None or (isinstance(date_str, str) and not date_str.strip()):
        return None
    
    if isinstance(date_str, datetime.datetime):
        return date_str
    
    if format:
        return datetime.datetime.strptime(date_str, format)
    
    # Handle common natural language date expressions
    if isinstance(date_str, str):
        date_str = date_str.lower().strip()
        now = datetime.datetime.now()
        
        if date_str == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "yesterday":
            return (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "tomorrow":
            return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str.endswith(" days ago"):
            try:
                days = int(date_str.split(" ")[0])
                return (now - datetime.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            except (ValueError, IndexError):
                pass
    
    # Try to parse using dateutil's flexible parser
    try:
        return parser.parse(date_str)
    except (ValueError, parser.ParserError) as err:
        raise ValueError(f"Could not parse date string: {date_str}") from err
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
```

## File: utils/json_utils.py
```python
"""
JSON utility functions for common JSON data operations.

This module provides utility functions for JSON serialization, deserialization,
validation, formatting, and safe access following CLEAN principles:
- Clear: Functions have descriptive names and clear docstrings.
- Logical: Each function has a single, well-defined purpose.
- Efficient: Optimized for typical JSON-related tasks.
- Adaptable: Allow optional parameters for flexibility.
"""

import json
from typing import Any, Dict, Optional, TypeVar, Union

T = TypeVar("T")


def json_load(file_path: str, encoding: str = "utf-8") -> Any:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.
        encoding: File encoding (default utf-8).

    Returns:
        Parsed JSON data.

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = json_load("data.json")
        >>> print(data)
        {'name': 'Bob'}
    """
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def json_save(data: Any, file_path: str, encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Save data as JSON to a file.

    Args:
        data: Data to serialize.
        file_path: Path to save the JSON file.
        encoding: File encoding (default utf-8).
        indent: Indentation spaces for formatting (default 2).

    Returns:
        None

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = {"name": "Bob"}
        >>> json_save(data, "data.json")
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def json_serialize(data: Any, indent: Optional[int] = None) -> str:
    """
    Serialize data to a JSON string.

    Args:
        data: Data to serialize.
        indent: Optional indentation for formatting.

    Returns:
        JSON-formatted string.

    Example:
        >>> json_serialize({"key": "value"})
        '{"key": "value"}'
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)


def json_deserialize(json_str: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Deserialize a JSON string into a Python object.

    Args:
        json_str: JSON-formatted string.
        default: Value to return if deserialization fails.

    Returns:
        Python data object or default.

    Example:
        >>> json_deserialize('{"name": "Bob"}')
        {'name': 'Bob'}
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default


def json_is_valid(json_str: str) -> bool:
    """
    Check if a string is valid JSON.

    Args:
        json_str: String to validate.

    Returns:
        True if valid JSON, False otherwise.

    Example:
        >>> json_is_valid('{"valid": true}')
        True
        >>> json_is_valid('{invalid json}')
        False
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def pretty_print_json(data: Any) -> str:
    """
    Pretty-print JSON data with indentation.

    Args:
        data: JSON data (Python object).

    Returns:
        Pretty-printed JSON string.

    Example:
        >>> pretty_print_json({"key": "value"})
        '{\n  "key": "value"\n}'
    """
    return json.dumps(data, ensure_ascii=False, indent=2)


def safe_get(data: Dict, key: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely get a value from a dictionary.

    Args:
        data: Dictionary to extract value from.
        key: Key to look up.
        default: Default value if key is missing.

    Returns:
        Value associated with key or default.

    Example:
        >>> safe_get({"a": 1}, "a")
        1
        >>> safe_get({"a": 1}, "b", 0)
        0
    """
    return data.get(key, default)


def safe_get_nested(data: Dict, *keys: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely retrieve a nested value from a dictionary.

    Args:
        data: Nested dictionary.
        *keys: Sequence of keys for nested lookup.
        default: Default value if key path is missing.

    Returns:
        Nested value or default.

    Example:
        >>> safe_get_nested({"a": {"b": 2}}, "a", "b")
        2
        >>> safe_get_nested({"a": {"b": 2}}, "a", "c", default="missing")
        'missing'
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
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
    simple_format: bool = False,
) -> None:
    """
    Configure application logging with console and optional file output using Loguru.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, only console logging is configured.
        rotation: When to rotate the log file (e.g., "10 MB", "1 day")
        retention: How long to keep log files (e.g., "1 month", "1 year")
        format_string: Custom format string for log messages
        simple_format: Use a simplified format for production environments
    """
    # Remove default handlers
    logger.remove()

    # Default format string if none provided
    if format_string is None:
        if simple_format:
            # Simple format for production environments
            def simple_format_fn(record: Record) -> str:
                return "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            
            format_string = simple_format_fn
        else:
            # Detailed format for development environments
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

## File: utils/number_utils.py
```python
"""
Number utility functions for common numeric operations.

This module provides a collection of utility functions for number formatting,
rounding, and random number generation that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union, overload

# Type aliases for numeric types
NumericType = Union[int, float, Decimal]


def format_currency(
    value: Optional[NumericType],
    symbol: str = "$",
    decimal_places: int = 2,
    thousands_sep: str = ",",
    decimal_sep: str = ".",
    default: str = "",
) -> str:
    """
    Format a number as a currency string.

    Args:
        value: The numeric value to format.
        symbol: Currency symbol to prepend (default: "$").
        decimal_places: Number of decimal places to show (default: 2).
        thousands_sep: Character to use as thousands separator (default: ",").
        decimal_sep: Character to use as decimal separator (default: ".").
        default: Value to return if input is None (default: "").

    Returns:
        Formatted currency string.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.56, symbol="€", decimal_sep=",")
        '€1,234,56'
    """
    if value is None:
        return default

    # Validate input type
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Handle negative values
    is_negative = value < 0
    abs_value = abs(value)

    # Round to specified decimal places
    if isinstance(value, Decimal):
        rounded_value = abs_value.quantize(Decimal(f"0.{'0' * decimal_places}"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = round(abs_value, decimal_places)

    # Convert to string and split into integer and decimal parts
    str_value = str(rounded_value)
    if "." in str_value:
        int_part, dec_part = str_value.split(".")
    else:
        int_part, dec_part = str_value, ""

    # Format integer part with thousands separator
    formatted_int = ""
    for i, char in enumerate(reversed(int_part)):
        if i > 0 and i % 3 == 0:
            formatted_int = thousands_sep + formatted_int
        formatted_int = char + formatted_int

    # Format decimal part
    if decimal_places > 0:
        # Pad with zeros if needed
        dec_part = dec_part.ljust(decimal_places, "0")
        # Truncate if too long
        dec_part = dec_part[:decimal_places]
        formatted_value = formatted_int + decimal_sep + dec_part
    else:
        formatted_value = formatted_int

    # Add currency symbol and handle negative values
    if is_negative:
        return f"-{symbol}{formatted_value}"
    else:
        return f"{symbol}{formatted_value}"


def round_to(value: NumericType, places: int) -> NumericType:
    """
    Round a number to a specified number of decimal places.

    This function handles both positive and negative decimal places:
    - Positive places round to that many decimal places
    - Zero places round to the nearest integer
    - Negative places round to tens, hundreds, etc.

    Args:
        value: The numeric value to round.
        places: Number of decimal places to round to.

    Returns:
        Rounded value of the same type as the input.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> round_to(1.234, 2)
        1.23
        >>> round_to(1.235, 2)
        1.24
        >>> round_to(123, -1)
        120
    """
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Handle Decimal type
    if isinstance(value, Decimal):
        if places >= 0:
            return value.quantize(Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP)
        else:
            # For negative places (tens, hundreds, etc.)
            return value.quantize(Decimal(f"1{'0' * abs(places)}"), rounding=ROUND_HALF_UP)

    # Handle int and float types
    factor = 10 ** places
    if places >= 0:
        result = round(value * factor) / factor
        # Convert to int if places is 0
        return int(result) if places == 0 else result
    else:
        # For negative places (tens, hundreds, etc.)
        return round(value / factor) * factor


def random_number(min_value: int, max_value: int) -> int:
    """
    Generate a random integer within a specified range.

    Args:
        min_value: The minimum value (inclusive).
        max_value: The maximum value (inclusive).

    Returns:
        A random integer between min_value and max_value (inclusive).

    Raises:
        ValueError: If min_value is greater than max_value.
        TypeError: If min_value or max_value is not an integer.

    Examples:
        >>> # Returns a random number between 1 and 10
        >>> random_number(1, 10)
        7
        >>> # Returns a random number between -10 and 10
        >>> random_number(-10, 10)
        -3
    """
    # Validate input types
    if not isinstance(min_value, int):
        raise TypeError(f"min_value must be an integer, got {type(min_value).__name__}")
    if not isinstance(max_value, int):
        raise TypeError(f"max_value must be an integer, got {type(max_value).__name__}")

    # Validate range
    if min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")

    return random.randint(min_value, max_value)
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

    logger.warning(f"Failed to resolve path '{path}'. Returning as is.")
    return path_obj
```

## File: utils/string_utils.py
```python
"""
String utility functions for common text operations.

This module provides a collection of utility functions for string manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import re
import unicodedata
from typing import Optional


def capitalize(text: str) -> str:
    """
    Capitalize the first letter of a string, preserving leading whitespace.

    Args:
        text: The string to capitalize.

    Returns:
        A string with the first non-space character capitalized.

    Examples:
        >>> capitalize("hello")
        'Hello'
        >>> capitalize("  hello")
        '  Hello'
        >>> capitalize("123abc")
        '123abc'
        >>> capitalize("")
        ''
    """
    if not text:
        return ""
    
    # If the string starts with non-alphabetic characters (except whitespace),
    # return it unchanged
    if text.strip() and not text.strip()[0].isalpha():
        return text
    
    # Preserve leading whitespace and capitalize the first non-space character
    leading_spaces = len(text) - len(text.lstrip())
    return text[:leading_spaces] + text[leading_spaces:].capitalize()


def slugify(text: str) -> str:
    """
    Convert text into a URL-friendly slug.

    This function:
    1. Converts to lowercase
    2. Removes accents/diacritics
    3. Replaces spaces and special characters with hyphens

    Args:
        text: The string to convert to a slug.

    Returns:
        URL-friendly slug.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Héllo, Wörld!")
        'hello-world'
    """
    if not text:
        return ""

    # Normalize and remove accents
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    
    # Replace multiple hyphens with a single hyphen
    text = re.sub(r"-+", "-", text)
    
    return text


def truncate(text: str, length: int, suffix: str = "...") -> str:
    """
    Limit the length of a string and add a suffix if truncated.

    Args:
        text: The string to truncate.
        length: Maximum allowed length before truncation.
        suffix: String appended after truncation (default "...").

    Returns:
        Truncated string.

    Examples:
        >>> truncate("Hello World", 5)
        'Hello...'
        >>> truncate("Hello", 10)
        'Hello'
    """
    if not text:
        return ""

    if length <= 0:
        return suffix

    return text if len(text) <= length else text[:length] + suffix


def is_empty(text: Optional[str]) -> bool:
    """
    Check if a string is empty or contains only whitespace.

    Args:
        text: The string to check.

    Returns:
        True if empty or whitespace, False otherwise.

    Raises:
        TypeError: if text is None.

    Examples:
        >>> is_empty("   ")
        True
        >>> is_empty("Hello")
        False
    """
    if text is None:
        raise TypeError("Cannot check emptiness of None")

    return not bool(text.strip())
```

## File: utils/validation_utils.py
```python
"""
Validation utilities for common data formats.

This module provides functions to validate common data formats such as
email addresses, phone numbers, and URLs.
"""
import re
from typing import Any


def is_valid_email(email: Any) -> bool:
    """
    Validate if the input is a properly formatted email address.

    Args:
        email: The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    # RFC 5322 compliant email regex pattern with additional validations
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
    
    # Basic pattern match
    if not re.match(pattern, email):
        return False
    
    # Additional validations
    if '..' in email:  # No consecutive dots
        return False
    if email.endswith('.'):  # No trailing dot
        return False
    if ' ' in email:  # No spaces
        return False
    
    # Check domain part
    domain = email.split('@')[1]
    if domain.startswith('-') or domain.endswith('-'):  # No leading/trailing hyphens in domain
        return False
    
    # Check for hyphens at the end of domain parts
    domain_parts = domain.split('.')
    for part in domain_parts:
        if part.endswith('-'):
            return False
    
    return True


def is_valid_phone(phone: Any) -> bool:
    """
    Validate if the input is a properly formatted phone number.

    Accepts various formats including:
    - 10 digits: 1234567890
    - Hyphenated: 123-456-7890
    - Parentheses: (123) 456-7890
    - International: +1 123-456-7890
    - Dots: 123.456.7890
    - Spaces: 123 456 7890

    Args:
        phone: The phone number to validate.

    Returns:
        bool: True if the phone number is valid, False otherwise.
    """
    if not isinstance(phone, str):
        return False

    # Check for specific invalid formats first
    if phone == "":
        return False
    
    # Check for spaces around hyphens
    if " - " in phone:
        return False
    
    # Check for missing space after parentheses in format like (123)456-7890
    if re.search(r'\)[0-9]', phone):
        return False
    
    # Remove all non-alphanumeric characters for normalization
    normalized = re.sub(r'[^0-9+]', '', phone)
    
    # Check for letters in the phone number
    if re.search(r'[a-zA-Z]', phone):
        return False
    
    # Check for international format (starting with +)
    if normalized.startswith('+'):
        # International numbers should have at least 8 digits after the country code
        return len(normalized) >= 9 and normalized[1:].isdigit()
    
    # For US/Canada numbers, expect 10 digits
    return len(normalized) == 10 and normalized.isdigit()


def is_valid_url(url: Any) -> bool:
    """
    Validate if the input is a properly formatted URL.

    Validates URLs with http or https protocols.

    Args:
        url: The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if not isinstance(url, str):
        return False

    # Check for specific invalid formats first
    if url == "":
        return False
    
    # Check for spaces
    if ' ' in url:
        return False
    
    # Check for double dots
    if '..' in url:
        return False
    
    # Check for trailing dot
    if url.endswith('.'):
        return False
    
    # URL regex pattern that validates common URL formats
    pattern = r'^(https?:\/\/)' + \
              r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
              r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
              r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
              r'(#[-a-zA-Z0-9%_]+)?$'
    
    # Basic pattern match
    if not re.match(pattern, url):
        return False
    
    # Check for domain part
    domain_part = url.split('://')[1].split('/')[0].split(':')[0]
    if domain_part.startswith('-') or domain_part.endswith('-'):
        return False
    
    return True
```
