# Utility Module: verification

## Overview

The `verification` module provides a script to verify that all necessary components are in place to run the NexusML system. It performs comprehensive checks for required packages, data files, and module imports, helping users identify and resolve potential issues before running the NexusML examples.

Key features include:

1. **Package Verification**: Check if all required packages are installed and their versions
2. **Data File Verification**: Verify that training data files exist and can be read
3. **Module Import Verification**: Ensure that all required NexusML modules can be imported
4. **Comprehensive Reporting**: Provide detailed reports of verification results
5. **Cross-Platform Compatibility**: Work across different execution environments

## Functions

### `get_package_version(package_name: str) -> str`

Get the version of a package in a type-safe way.

**Parameters:**

- `package_name` (str): Name of the package

**Returns:**

- str: Version string or "unknown" if version cannot be determined

**Example:**
```python
from nexusml.utils.verification import get_package_version

# Get the version of numpy
numpy_version = get_package_version("numpy")
print(f"NumPy version: {numpy_version}")

# Get the version of pandas
pandas_version = get_package_version("pandas")
print(f"Pandas version: {pandas_version}")

# Get the version of a package that might not be installed
try:
    version = get_package_version("non_existent_package")
    print(f"Version: {version}")
except Exception as e:
    print(f"Error: {e}")
```

**Notes:**

- This function tries multiple methods to get the package version:
  1. First, it tries to get the version directly from the module's `__version__` attribute
  2. If that fails, it tries to use `importlib.metadata.version` (Python 3.8+)
  3. If that fails, it falls back to `pkg_resources.get_distribution().version`
  4. If all methods fail, it returns "unknown"
- This function is designed to be robust and not raise exceptions, making it safe to use in verification scripts

### `read_csv_safe(filepath: Union[str, Path]) -> DataFrame`

Type-safe wrapper for pd.read_csv.

**Parameters:**

- `filepath` (Union[str, Path]): Path to the CSV file

**Returns:**

- DataFrame: DataFrame containing the CSV data

**Example:**
```python
from nexusml.utils.verification import read_csv_safe

# Read a CSV file safely
try:
    df = read_csv_safe("data/sample.csv")
    print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"Error reading CSV: {e}")
```

**Notes:**

- This function is a simple wrapper around pandas.read_csv
- It includes a type ignore comment to suppress Pylance warnings about complex types
- It's used in the verification script to ensure type safety when reading CSV files

### `check_package_versions()`

Check if all required packages are installed and compatible.

**Returns:**

- bool: True if all packages are installed and compatible, False otherwise

**Example:**
```python
from nexusml.utils.verification import check_package_versions

# Check if all required packages are installed
if check_package_versions():
    print("All required packages are installed")
else:
    print("Some required packages are missing or incompatible")
```

**Notes:**

- This function checks for the following packages:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - imbalanced-learn
- For each package, it prints:
  - A checkmark (✓) and the version if the package is installed
  - An X (✗) if the package is not installed
- It returns True only if all packages are installed
- This function is useful for verifying that all dependencies are met before running NexusML

### `check_data_file()`

Check if the training data file exists.

**Returns:**

- bool: True if the data file exists and can be read, False otherwise

**Example:**
```python
from nexusml.utils.verification import check_data_file

# Check if the training data file exists
if check_data_file():
    print("Training data file exists and can be read")
else:
    print("Training data file is missing or cannot be read")
```

**Notes:**

- This function tries to locate the training data file using several methods:
  1. First, it tries to load the path from settings.yml in the fca_dashboard context
  2. If that fails, it tries to load from settings.yml in the nexusml context
  3. If that fails, it falls back to a default path in the nexusml package
- It checks if the file exists and can be read
- It prints detailed information about the file location and status
- This function is useful for verifying that the required data files are available before running NexusML

### `check_module_imports()`

Check if all required module imports work correctly.

**Returns:**

- bool: True if all module imports work correctly, False otherwise

**Example:**
```python
from nexusml.utils.verification import check_module_imports

# Check if all required module imports work correctly
if check_module_imports():
    print("All required module imports work correctly")
else:
    print("Some required module imports failed")
```

**Notes:**

- This function checks if the following modules and attributes can be imported:
  - nexusml.core.model.train_enhanced_model
  - nexusml.core.model.predict_with_enhanced_model
  - nexusml.core.data_preprocessing.load_and_preprocess_data
  - nexusml.core.feature_engineering.enhance_features
- For each module and attribute, it prints:
  - A checkmark (✓) if the import is successful
  - An X (✗) if the import fails, along with the error message
- It returns True only if all imports are successful
- This function is useful for verifying that the NexusML package is correctly installed and accessible

### `main()`

Run all verification checks.

**Returns:**

- int: 0 if all checks pass, 1 if any check fails

**Example:**
```python
from nexusml.utils.verification import main

# Run all verification checks
exit_code = main()
print(f"Verification completed with exit code: {exit_code}")
```

**Notes:**

- This function runs all verification checks:
  1. check_package_versions()
  2. check_data_file()
  3. check_module_imports()
- It prints a summary of the results
- If all checks pass, it prints instructions for running the NexusML example
- If any check fails, it prints a message to fix the issues
- It returns 0 if all checks pass, 1 if any check fails
- This function is the entry point for the verification script

## Command-Line Usage

The module can be run as a script to verify the NexusML installation:

```bash
python -m nexusml.utils.verification
```

This will run all verification checks and print a summary of the results:

```
============================================================
NEXUSML VERIFICATION
============================================================
Checking package versions...
✓ numpy: 1.24.3
✓ pandas: 2.0.1
✓ scikit-learn: 1.2.2
✓ matplotlib: 3.7.1
✓ seaborn: 0.12.2
✓ imbalanced-learn: 0.10.1

Checking data file: /path/to/nexusml/ingest/data/eq_ids.csv
✓ Data file exists: /path/to/nexusml/ingest/data/eq_ids.csv
✓ Data file can be read: 1000 rows, 10 columns

Checking module imports...
✓ Successfully imported nexusml.core.model.train_enhanced_model
✓ Successfully imported nexusml.core.model.predict_with_enhanced_model
✓ Successfully imported nexusml.core.data_preprocessing.load_and_preprocess_data
✓ Successfully imported nexusml.core.feature_engineering.enhance_features

============================================================
VERIFICATION SUMMARY
============================================================
Packages: ✓ OK
Data file: ✓ OK
Module imports: ✓ OK

All checks passed! You can run the NexusML example with:

    python -m nexusml.examples.simple_example
```

If any check fails, the script will print details about the issues and exit with a non-zero status code.

## Required Packages

The verification script checks for the following packages:

1. **numpy**: Numerical computing library
2. **pandas**: Data manipulation and analysis library
3. **scikit-learn**: Machine learning library
4. **matplotlib**: Plotting library
5. **seaborn**: Statistical data visualization library
6. **imbalanced-learn**: Library for handling imbalanced datasets

## Required Modules

The verification script checks for the following NexusML modules and attributes:

1. **nexusml.core.model.train_enhanced_model**: Function for training the enhanced model
2. **nexusml.core.model.predict_with_enhanced_model**: Function for making predictions with the enhanced model
3. **nexusml.core.data_preprocessing.load_and_preprocess_data**: Function for loading and preprocessing data
4. **nexusml.core.feature_engineering.enhance_features**: Function for enhancing features

## Data File Locations

The verification script tries to locate the training data file in the following locations:

1. Path specified in fca_dashboard settings.yml under classifier.data_paths.training_data
2. Path specified in nexusml settings.yml under data_paths.training_data
3. Default path: nexusml/ingest/data/eq_ids.csv

## Dependencies

- **importlib**: Standard library module for importing modules
- **os**: Standard library module for file operations
- **sys**: Standard library module for system-specific parameters and functions
- **pathlib**: Standard library module for path manipulation
- **typing**: Standard library module for type hints
- **pandas**: Used for DataFrame operations
- **yaml**: Optional dependency for loading settings

## Notes and Warnings

- The verification script is designed to be robust and not raise exceptions, making it safe to run in any environment
- The script tries multiple methods to locate the training data file, which may result in different paths depending on the environment
- The script checks for specific modules and attributes that are required for the NexusML examples to run
- If any check fails, the script will print details about the issues and exit with a non-zero status code
- The script is particularly useful for verifying that all dependencies are met before running NexusML examples
- The script can be run as a standalone script or imported and used programmatically
- The script is cross-platform compatible and should work on Windows, macOS, and Linux
