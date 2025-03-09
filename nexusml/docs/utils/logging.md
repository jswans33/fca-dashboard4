# Utility Module: logging

## Overview

The `logging` module provides a unified logging interface for the NexusML system that works both standalone and when integrated with the fca_dashboard application. It offers a consistent way to configure logging and obtain logger instances throughout the application.

Key features include:

1. **Unified Interface**: Works seamlessly whether used standalone or integrated with fca_dashboard
2. **Flexible Configuration**: Supports different logging levels, file output, and format options
3. **Integration Support**: Automatically detects and uses fca_dashboard logging if available
4. **Fallback Mechanism**: Falls back to standard Python logging when fca_dashboard is not available
5. **Simplified Access**: Provides a simple function to get logger instances

## Functions

### `configure_logging(level: Union[str, int] = "INFO", log_file: Optional[str] = None, simple_format: bool = False) -> logging.Logger`

Configure application logging.

**Parameters:**

- `level` (Union[str, int], optional): Logging level (e.g., "INFO", "DEBUG", etc.). Default is "INFO".
- `log_file` (Optional[str], optional): Path to log file (if None, logs to console only). Default is None.
- `simple_format` (bool, optional): Whether to use a simplified log format. Default is False.

**Returns:**

- logging.Logger: Configured root logger

**Example:**
```python
from nexusml.utils.logging import configure_logging

# Configure logging with default settings (INFO level, console only)
logger = configure_logging()
logger.info("Application started")

# Configure logging with DEBUG level and file output
logger = configure_logging(
    level="DEBUG",
    log_file="logs/application.log",
    simple_format=False
)
logger.debug("Detailed debugging information")

# Configure logging with simplified format
logger = configure_logging(
    level="INFO",
    simple_format=True
)
logger.info("This message will be displayed without timestamp or level")
```

**Notes:**

- If fca_dashboard is available, it uses fca_dashboard's configure_logging function
- Otherwise, it falls back to standard Python logging
- When using the fallback:
  - It creates the log directory if it doesn't exist
  - It configures the root logger with the specified level
  - It removes existing handlers to avoid duplicates
  - It adds a console handler that outputs to stdout
  - It adds a file handler if log_file is specified
- The simple_format option determines the log message format:
  - If True: "%(message)s"
  - If False: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

### `get_logger(name: str = "nexusml") -> logging.Logger`

Get a logger instance.

**Parameters:**

- `name` (str, optional): Logger name. Default is "nexusml".

**Returns:**

- logging.Logger: Logger instance

**Example:**
```python
from nexusml.utils.logging import get_logger

# Get the default nexusml logger
logger = get_logger()
logger.info("Using default nexusml logger")

# Get a module-specific logger
logger = get_logger(__name__)
logger.info("Using module-specific logger")

# Get a custom-named logger
logger = get_logger("data_processing")
logger.info("Using custom-named logger")
```

**Notes:**

- This function is a simple wrapper around logging.getLogger()
- It provides a consistent way to get logger instances throughout the application
- The default name "nexusml" creates a logger in the nexusml namespace
- Using **name** as the name creates a logger in the module's namespace
- Logger instances are hierarchical, so a logger named "nexusml.utils" is a child of "nexusml"

## Integration with fca_dashboard

The module attempts to integrate with fca_dashboard's logging system if available:

```python
# Try to use fca_dashboard logging if available
try:
    from fca_dashboard.utils.logging_config import (
        configure_logging as fca_configure_logging,
    )

    FCA_LOGGING_AVAILABLE = True
    FCA_CONFIGURE_LOGGING = fca_configure_logging
except ImportError:
    FCA_LOGGING_AVAILABLE = False
    FCA_CONFIGURE_LOGGING = None
```

When fca_dashboard is available:

- The configure_logging function delegates to fca_dashboard's configure_logging
- This ensures consistent logging behavior across both applications
- It uses type casting to maintain proper type hints

When fca_dashboard is not available:

- The module falls back to standard Python logging
- It provides similar functionality to fca_dashboard's logging system
- This allows code to work the same way regardless of the environment

## Logging Levels

The module supports all standard Python logging levels:

| Level    | Numeric Value | Description                                           |
|----------|---------------|-------------------------------------------------------|
| CRITICAL | 50            | Critical errors that may prevent the program from continuing |
| ERROR    | 40            | Errors that don't prevent the program from continuing |
| WARNING  | 30            | Warnings about potential issues                       |
| INFO     | 20            | General information about program execution           |
| DEBUG    | 10            | Detailed information for debugging                    |
| NOTSET   | 0             | No level set (inherits from parent logger)            |

## Log Format Options

The module supports two log format options:

1. **Standard Format** (simple_format=False):
   ```
   2023-03-09 15:30:45,123 - nexusml.utils - INFO - This is an info message
   ```
   This format includes:
   - Timestamp (asctime)
   - Logger name (name)
   - Log level (levelname)
   - Message (message)

2. **Simple Format** (simple_format=True):
   ```
   This is an info message
   ```
   This format includes only the message, which is useful for cleaner output in certain contexts.

## Usage Patterns

### Basic Application Setup

```python
from nexusml.utils.logging import configure_logging, get_logger

# Configure application-wide logging
configure_logging(
    level="INFO",
    log_file="logs/application.log"
)

# Get a module-specific logger
logger = get_logger(__name__)

# Use the logger
logger.info("Application started")
logger.debug("This won't be displayed with INFO level")
logger.warning("This is a warning message")
```

### Module-Level Usage

```python
from nexusml.utils.logging import get_logger

# Get a logger for the current module
logger = get_logger(__name__)

def process_data(data):
    """Process some data."""
    logger.info(f"Processing {len(data)} items")
    try:
        # Process data
        result = [item * 2 for item in data]
        logger.debug(f"Processed data: {result}")
        return result
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise
```

### Temporary Debug Logging

```python
from nexusml.utils.logging import configure_logging, get_logger
import logging

# Store the original level
original_level = logging.getLogger().level

# Temporarily increase logging detail
configure_logging(level="DEBUG")
logger = get_logger("debug_session")

try:
    # Perform operations with detailed logging
    logger.debug("Starting detailed logging")
    # ... operations ...
    logger.debug("Detailed logging complete")
finally:
    # Restore original logging level
    configure_logging(level=original_level)
```

## Dependencies

- **logging**: Standard library module for logging
- **os**: Standard library module for file operations
- **sys**: Standard library module for system-specific parameters and functions
- **typing**: Standard library module for type hints
- **fca_dashboard.utils.logging_config**: Optional dependency for integration with fca_dashboard

## Notes and Warnings

- The module attempts to use fca_dashboard's logging system if available, but falls back to standard Python logging if not
- When using the fallback, it configures the root logger, which affects all logging in the application
- The fallback removes existing handlers from the root logger to avoid duplicate log messages
- If log_file is specified, the module creates the log directory if it doesn't exist
- The module uses type casting to maintain proper type hints when delegating to fca_dashboard's logging system
- The get_logger function returns a logger in the specified namespace, which inherits settings from the root logger
- Logger instances are hierarchical, so a logger named "nexusml.utils" is a child of "nexusml"
- The default logger name "nexusml" creates a logger in the nexusml namespace
