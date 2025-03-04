This file is a merged representation of a subset of the codebase, containing files not matching ignore patterns, combined into a single document by Repomix.

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
- Files matching these patterns are excluded: tests/
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded

## Additional Info

# Directory Structure
```
.repomixignore
config/__init__.py
config/settings.py
config/settings.yml
coverage_output.txt
coverage_report.txt
main.py
repomix.config.json
utils/__init__.py
utils/date_utils.py
utils/env_utils.py
utils/error_handler.py
utils/json_utils.py
utils/logging_config.py
utils/loguru_stubs.pyi
utils/number_utils.py
utils/path_util.py
utils/string_utils.py
utils/validation_utils.py
utils/validation_utils.py,cover
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
# Environment settings
env:
  ENVIRONMENT: "development"  # Default environment (can be overridden by OS environment variables)
  LOG_LEVEL: "INFO"
  DEBUG: true

# Database settings
databases:
  sqlite:
    url: "sqlite:///fca_dashboard.db"
  postgresql:
    url: "${POSTGRES_URL}"
    
# Pipeline settings
pipeline_settings:
  batch_size: 5000
  log_level: "${LOG_LEVEL}"  # Uses the environment variable from env section
  
# Table mappings
tables:
  equipment:
    mapping_type: "direct"
    column_mappings:
      tag: "Tag"
      name: "Name"
      description: "Description"
```

## File: coverage_output.txt
```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.4, pluggy-1.5.0 -- C:\Repos\fca-dashboard4\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Repos\fca-dashboard4
configfile: pytest.ini
plugins: cov-4.1.0
collecting ... collected 18 items

tests\unit\test_validation_utils.py::TestEmailValidation::test_valid_emails PASSED [  5%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_invalid_emails PASSED [ 11%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_consecutive_dots PASSED [ 16%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_trailing_dot PASSED [ 22%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_with_spaces PASSED [ 27%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_email_domain_with_hyphens PASSED [ 33%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_domain_parts_with_hyphens PASSED [ 38%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_none_input PASSED [ 44%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_non_string_input PASSED [ 50%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_valid_phone_numbers PASSED [ 55%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_invalid_phone_numbers PASSED [ 61%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_international_phone_formats PASSED [ 66%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_none_input PASSED [ 72%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_non_string_input PASSED [ 77%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_valid_urls PASSED [ 83%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_invalid_urls PASSED [ 88%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_none_input PASSED [ 94%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_non_string_input PASSED [100%]

---------- coverage: platform win32, python 3.12.6-final-0 -----------
Name                        Stmts   Miss  Cover
-----------------------------------------------
utils\validation_utils.py      50      3    94%
-----------------------------------------------
TOTAL                          50      3    94%


============================= 18 passed in 0.48s ==============================
```

## File: coverage_report.txt
```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.4, pluggy-1.5.0 -- C:\Repos\fca-dashboard4\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Repos\fca-dashboard4
configfile: pytest.ini
plugins: cov-4.1.0
collecting ... collected 14 items

tests\unit\test_validation_utils.py::TestEmailValidation::test_valid_emails PASSED [  7%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_invalid_emails PASSED [ 14%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_domain_parts_with_hyphens PASSED [ 21%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_none_input PASSED [ 28%]
tests\unit\test_validation_utils.py::TestEmailValidation::test_non_string_input PASSED [ 35%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_valid_phone_numbers PASSED [ 42%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_invalid_phone_numbers PASSED [ 50%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_international_phone_formats PASSED [ 57%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_none_input PASSED [ 64%]
tests\unit\test_validation_utils.py::TestPhoneValidation::test_non_string_input PASSED [ 71%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_valid_urls PASSED [ 78%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_invalid_urls PASSED [ 85%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_none_input PASSED [ 92%]
tests\unit\test_validation_utils.py::TestUrlValidation::test_non_string_input PASSED [100%]

---------- coverage: platform win32, python 3.12.6-final-0 -----------
Name                        Stmts   Miss  Cover   Missing
---------------------------------------------------------
utils\validation_utils.py      50      3    94%   35, 37, 42
---------------------------------------------------------
TOTAL                          50      3    94%


============================= 14 passed in 0.48s ==============================
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

## File: utils/env_utils.py
```python
"""
Environment and configuration utilities.

This module provides functions to safely access environment variables
and check the current running environment. It integrates with the application's
settings module for consistent configuration access.
"""

import os
from typing import Any

from fca_dashboard.config.settings import settings

# The environment variable name used to determine the current environment
ENV_VAR_NAME = "ENVIRONMENT"


def get_env_var(key: str, fallback: Any = None) -> Any:
    """
    Safely access environment variables with an optional fallback value.
    
    This function first checks if the environment variable is set directly in
    the OS environment. If not found, it attempts to retrieve it from the
    application settings. If still not found, it returns the fallback value.
    
    Args:
        key: The name of the environment variable to retrieve
        fallback: The value to return if the environment variable is not set
        
    Returns:
        The value of the environment variable if it exists, otherwise the fallback value
    """
    # First check OS environment variables
    value = os.environ.get(key)
    
    # If not found in OS environment, check application settings
    if value is None:
        # Look for the key in the env section of settings
        value = settings.get(f"env.{key}")
        
        # If still not found, look for it at the top level
        if value is None:
            value = settings.get(key)
    
    # If still not found, return the fallback
    if value is None:
        return fallback
        
    return value


def is_dev() -> bool:
    """
    Check if the current environment is development.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is development.
    
    Returns:
        True if the current environment is development, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["development", "dev"]


def is_prod() -> bool:
    """
    Check if the current environment is production.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is production.
    
    Returns:
        True if the current environment is production, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["production", "prod"]


def get_environment() -> str:
    """
    Get the current environment name.
    
    Returns:
        The current environment name (e.g., 'development', 'production', 'staging')
        or 'unknown' if not set
    """
    return str(get_env_var(ENV_VAR_NAME, "unknown")).lower()
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


def round_to(value: NumericType, places: int = 0) -> NumericType:
    """
    Round a number to a specified number of decimal places with ROUND_HALF_UP rounding.

    This function handles both positive and negative decimal places:
    - Positive places round to that many decimal places
    - Zero places round to the nearest integer
    - Negative places round to tens, hundreds, etc.

    Args:
        value: The numeric value to round.
        places: Number of decimal places to round to (default: 0).

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
        >>> round_to(125, -1)
        130
    """
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Preserve the original type
    original_type = type(value)
    
    # Convert to Decimal for consistent rounding behavior
    if not isinstance(value, Decimal):
        decimal_value = Decimal(str(value))
    else:
        decimal_value = value
    
    # Calculate the factor based on places
    factor = Decimal("10") ** places
    
    if places >= 0:
        # For positive places (decimal places)
        result = decimal_value.quantize(Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP)
    else:
        # For negative places (tens, hundreds, etc.)
        # First divide by factor, round to integer, then multiply back
        factor = Decimal("10") ** abs(places)
        result = (decimal_value / factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * factor
    
    # Return the result in the original type
    if original_type == int or (places == 0 and original_type == float):
        # Convert to int if original was int or if rounding to integer (places=0)
        return int(result)
    elif original_type == float:
        return float(result)
    else:
        return result  # Already a Decimal


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
    return all(not part.endswith('-') for part in domain_parts)


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
              r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|' + \
              r'(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
              r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
              r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
              r'(#[-a-zA-Z0-9%_]+)?$'
    
    # Basic pattern match
    if not re.match(pattern, url):
        return False
    
    # Check for domain part
    domain_part = url.split('://')[1].split('/')[0].split(':')[0]
    return not (domain_part.startswith('-') or domain_part.endswith('-'))
```

## File: utils/validation_utils.py,cover
```
> """
> Validation utilities for common data formats.
  
> This module provides functions to validate common data formats such as
> email addresses, phone numbers, and URLs.
> """
> import re
> from typing import Any
  
  
> def is_valid_email(email: Any) -> bool:
>     """
>     Validate if the input is a properly formatted email address.
  
>     Args:
>         email: The email address to validate.
  
>     Returns:
>         bool: True if the email is valid, False otherwise.
>     """
>     if not isinstance(email, str):
>         return False
  
      # RFC 5322 compliant email regex pattern with additional validations
>     pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
      
      # Basic pattern match
>     if not re.match(pattern, email):
>         return False
      
      # Additional validations
>     if '..' in email:  # No consecutive dots
>         return False
>     if email.endswith('.'):  # No trailing dot
!         return False
>     if ' ' in email:  # No spaces
!         return False
      
      # Check domain part
>     domain = email.split('@')[1]
>     if domain.startswith('-') or domain.endswith('-'):  # No leading/trailing hyphens in domain
!         return False
      
      # Check for hyphens at the end of domain parts
>     domain_parts = domain.split('.')
>     return all(not part.endswith('-') for part in domain_parts)
  
  
> def is_valid_phone(phone: Any) -> bool:
>     """
>     Validate if the input is a properly formatted phone number.
  
>     Accepts various formats including:
>     - 10 digits: 1234567890
>     - Hyphenated: 123-456-7890
>     - Parentheses: (123) 456-7890
>     - International: +1 123-456-7890
>     - Dots: 123.456.7890
>     - Spaces: 123 456 7890
  
>     Args:
>         phone: The phone number to validate.
  
>     Returns:
>         bool: True if the phone number is valid, False otherwise.
>     """
>     if not isinstance(phone, str):
>         return False
  
      # Check for specific invalid formats first
>     if phone == "":
>         return False
      
      # Check for spaces around hyphens
>     if " - " in phone:
>         return False
      
      # Check for missing space after parentheses in format like (123)456-7890
>     if re.search(r'\)[0-9]', phone):
>         return False
      
      # Remove all non-alphanumeric characters for normalization
>     normalized = re.sub(r'[^0-9+]', '', phone)
      
      # Check for letters in the phone number
>     if re.search(r'[a-zA-Z]', phone):
>         return False
      
      # Check for international format (starting with +)
>     if normalized.startswith('+'):
          # International numbers should have at least 8 digits after the country code
>         return len(normalized) >= 9 and normalized[1:].isdigit()
      
      # For US/Canada numbers, expect 10 digits
>     return len(normalized) == 10 and normalized.isdigit()
  
  
> def is_valid_url(url: Any) -> bool:
>     """
>     Validate if the input is a properly formatted URL.
  
>     Validates URLs with http or https protocols.
  
>     Args:
>         url: The URL to validate.
  
>     Returns:
>         bool: True if the URL is valid, False otherwise.
>     """
>     if not isinstance(url, str):
>         return False
  
      # Check for specific invalid formats first
>     if url == "":
>         return False
      
      # Check for spaces
>     if ' ' in url:
>         return False
      
      # Check for double dots
>     if '..' in url:
>         return False
      
      # Check for trailing dot
>     if url.endswith('.'):
>         return False
      
      # URL regex pattern that validates common URL formats
>     pattern = r'^(https?:\/\/)' + \
>               r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
>               r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
>               r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
>               r'(#[-a-zA-Z0-9%_]+)?$'
      
      # Basic pattern match
>     if not re.match(pattern, url):
>         return False
      
      # Check for domain part
>     domain_part = url.split('://')[1].split('/')[0].split(':')[0]
>     return not (domain_part.startswith('-') or domain_part.endswith('-'))
```
