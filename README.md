# FCA Dashboard

A data extraction, transformation, and loading (ETL) pipeline for the FCA Dashboard application.

## Overview

The FCA Dashboard is a Python-based ETL application designed to process data from various sources, transform it according to business rules, and load it into a database for reporting and analysis.

## Features

- Configurable ETL pipeline
- Support for multiple data sources (Excel, databases)
- Extensible architecture with strategy patterns
- Comprehensive logging
- 100% test coverage

## Project Structure

```text
fca_dashboard/
├── config/           # Configuration files
├── core/             # Core functionality and interfaces
├── extractors/       # Data extraction modules
├── loaders/          # Data loading modules
├── mappers/          # Data mapping and transformation
├── pipelines/        # ETL pipeline orchestration
├── tests/            # Test suite
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
└── utils/            # Utility functions
```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd fca-dashboard4
   ```

2. Install dependencies:

   ```bash
   make install
   ```

## Installing as a Python Module

You can install this package as a Python module, allowing you to import and use it in other projects.

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a faster alternative to pip for installing Python packages.

1. Install uv if you don't have it:

   ```bash
   pip install uv
   ```

2. Install the package in development mode (editable):

   ```bash
   uv pip install -e .
   ```

   Or for a regular installation:

   ```bash
   uv pip install .
   ```

   To include development dependencies:

   ```bash
   uv pip install -e ".[dev]"
   ```

### Using pip

1. Install the package in development mode (editable):

   ```bash
   pip install -e .
   ```

   Or for a regular installation:

   ```bash
   pip install .
   ```

   To include development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

### Using the Package

Once installed, you can import and use the package in your Python code:

```python
import fca_dashboard

# Check the version
print(fca_dashboard.__version__)

# Use modules from the package
from fca_dashboard.classifier import model
from fca_dashboard.config import settings
# etc.
```

## Usage

Run the ETL pipeline with default settings:

```bash
make run
```

Or specify custom configuration:

```bash
python fca_dashboard/main.py --config path/to/config.yml
```

Additional command-line options:

- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--excel-file`: Specify an Excel file to process
- `--table-name`: Specify a table name to process

## Development

### Code Quality

Ensure code quality by running linting checks:

```bash
make lint
```

Format code automatically:

```bash
make format
```

### Testing

Run all tests:

```bash
make test
```

Run unit tests only:

```bash
make test-unit
```

Run integration tests only:

```bash
make test-integration
```

### Test Coverage

Generate and view test coverage report:

```bash
make coverage
```

This will:

1. Run all tests with coverage analysis
2. Generate an HTML coverage report
3. Open the report in your default web browser

The project maintains 100% test coverage across all modules.

### Cleaning Up

Remove generated files and caches:

```bash
make clean
```

## Database

Initialize the database schema:

```bash
make init-db
```

## Configuration

The application uses YAML configuration files located in the `fca_dashboard/config/` directory. The default configuration file is `settings.yml`.

Example configuration:

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

## License

[License information]
