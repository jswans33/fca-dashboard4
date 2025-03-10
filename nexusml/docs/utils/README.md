# NexusML Utility Modules Documentation

This directory contains documentation for the utility modules of the NexusML package. These modules provide helper functions and classes that support the core functionality.

## Overview

The utility modules in NexusML provide various helper functions and classes for common tasks such as file handling, data manipulation, and logging. These utilities are designed to be reusable across different parts of the package.

## Utility Modules

- [csv_utils.py](csv_utils.md): Utilities for working with CSV files
- [data_selection.py](data_selection.md): Utilities for selecting and filtering data
- [excel_utils.py](excel_utils.md): Utilities for working with Excel files
- [logging.py](logging.md): Utilities for logging and monitoring
- [notebook_utils.py](notebook_utils.md): Utilities for Jupyter notebooks
- [path_utils.py](path_utils.md): Utilities for file path handling
- [verification.py](verification.md): Utilities for verification and validation

## Key Utilities

### CSV Utilities

The CSV utilities provide functions for:

- Reading CSV files with proper encoding and error handling
- Writing CSV files with consistent formatting
- Converting between CSV and other formats
- Validating CSV file structure

### Data Selection Utilities

The data selection utilities provide functions for:

- Selecting subsets of data based on criteria
- Filtering data by column values
- Sampling data for training and testing
- Balancing datasets for machine learning

### Excel Utilities

The Excel utilities provide functions for:

- Reading Excel files with proper sheet selection
- Writing Excel files with formatting
- Converting between Excel and other formats
- Handling Excel-specific features like formulas and formatting

### Logging Utilities

The logging utilities provide functions for:

- Setting up logging with consistent formatting
- Logging to files and console
- Configuring log levels
- Tracking progress of long-running operations

### Notebook Utilities

The notebook utilities provide functions for:

- Displaying data in Jupyter notebooks
- Creating interactive visualizations
- Monitoring model training in notebooks
- Integrating with other notebook tools

### Path Utilities

The path utilities provide functions for:

- Finding files and directories
- Creating directory structures
- Resolving relative paths
- Handling platform-specific path issues

### Verification Utilities

The verification utilities provide functions for:

- Verifying data integrity
- Validating configuration
- Checking system requirements
- Testing component functionality

## Usage Examples

```python
# Example of using CSV utilities
from nexusml.utils.csv_utils import read_csv_safe

# Read a CSV file safely with proper error handling
df = read_csv_safe("path/to/file.csv", encoding="utf-8")

# Example of using path utilities
from nexusml.utils.path_utils import ensure_directory_exists

# Ensure a directory exists, creating it if necessary
ensure_directory_exists("path/to/directory")
```

## Integration with Other Components

The utility modules are used throughout NexusML:

- Core modules use utilities for common tasks
- CLI tools use utilities for file handling and logging
- Examples use utilities for demonstration purposes
- Tests use utilities for verification and validation

## Next Steps

After reviewing the utility modules documentation, you might want to:

1. Explore the [Examples](../examples/README.md) for practical usage examples
2. Check the [API Reference](../api_reference.md) for detailed information on classes and methods
3. Read the [Core Modules Documentation](../modules/README.md) to see how utilities are used in core functionality