# Utility Module: csv_utils

## Overview

The `csv_utils` module provides utilities for working with CSV files in the NexusML system. It includes functions for cleaning, verification, and safe reading of potentially malformed CSV files. This module is particularly useful when dealing with CSV files from external sources that may have inconsistencies or formatting issues.

Key features include:

1. **CSV Verification**: Verify CSV files for common issues such as incorrect field counts or missing columns
2. **Issue Fixing**: Automatically fix common CSV issues like inconsistent field counts
3. **Safe Reading**: Safely read CSV files with built-in verification and error handling
4. **OmniClass Cleaning**: Special handling for OmniClass CSV files, which have specific formatting requirements

## Functions

### `verify_csv_file(filepath: Union[str, Path], expected_columns: Optional[List[str]] = None, expected_field_count: Optional[int] = None, fix_issues: bool = False, output_filepath: Optional[Union[str, Path]] = None) -> Tuple[bool, Optional[str]]`

Verify a CSV file for common issues and optionally fix them.

**Parameters:**

- `filepath` (Union[str, Path]): Path to the CSV file
- `expected_columns` (Optional[List[str]], optional): List of column names that should be present. Default is None.
- `expected_field_count` (Optional[int], optional): Expected number of fields per row. Default is None.
- `fix_issues` (bool, optional): Whether to attempt to fix issues. Default is False.
- `output_filepath` (Optional[Union[str, Path]], optional): Path to save the fixed file (if fix_issues is True). If None, will use the original filename with "_fixed" appended.

**Returns:**

- Tuple[bool, Optional[str]]: Tuple of (is_valid, error_message)
  - is_valid: True if the file is valid or was fixed successfully
  - error_message: Description of the issue if not valid, or None if valid

**Example:**
```python
from nexusml.utils.csv_utils import verify_csv_file
from pathlib import Path

# Verify a CSV file without fixing issues
is_valid, error_message = verify_csv_file(
    "data/equipment_data.csv",
    expected_columns=["Asset Tag", "Description", "Service Life"],
    expected_field_count=10
)

if is_valid:
    print("CSV file is valid")
else:
    print(f"CSV file has issues: {error_message}")

# Verify and fix issues
is_valid, error_message = verify_csv_file(
    "data/equipment_data.csv",
    expected_columns=["Asset Tag", "Description", "Service Life"],
    expected_field_count=10,
    fix_issues=True,
    output_filepath="data/equipment_data_fixed.csv"
)

if is_valid:
    print("CSV file was fixed successfully")
else:
    print(f"Could not fix CSV file: {error_message}")
```

**Notes:**

- This function first tries to read the file with pandas to check for basic validity
- If pandas can't read it, it uses a more manual approach with the csv module
- It checks for the correct number of fields in each row
- If fix_issues is True, it will:
  - Add empty columns to rows with too few fields
  - Combine extra columns for rows with too many fields
- The function logs warnings for issues found and info messages for fixes applied

### `read_csv_safe(filepath: Union[str, Path], expected_columns: Optional[List[str]] = None, expected_field_count: Optional[int] = None, fix_issues: bool = True, **kwargs: Any) -> DataFrame`

Safely read a CSV file, handling common issues.

**Parameters:**

- `filepath` (Union[str, Path]): Path to the CSV file
- `expected_columns` (Optional[List[str]], optional): List of column names that should be present. Default is None.
- `expected_field_count` (Optional[int], optional): Expected number of fields per row. Default is None.
- `fix_issues` (bool, optional): Whether to attempt to fix issues. Default is True.
- `**kwargs`: Additional arguments to pass to pd.read_csv

**Returns:**

- DataFrame: DataFrame containing the CSV data

**Raises:**

- ValueError: If the file is invalid and couldn't be fixed

**Example:**
```python
from nexusml.utils.csv_utils import read_csv_safe
import pandas as pd

try:
    # Read a CSV file safely, fixing issues if needed
    df = read_csv_safe(
        "data/equipment_data.csv",
        expected_columns=["Asset Tag", "Description", "Service Life"],
        expected_field_count=10,
        fix_issues=True,
        # Additional pandas read_csv arguments
        dtype={"Asset Tag": str, "Service Life": float},
        na_values=["N/A", "Unknown"]
    )
    
    # Process the DataFrame
    print(f"Loaded {len(df)} rows from CSV file")
    print(f"Columns: {df.columns.tolist()}")
    
except ValueError as e:
    print(f"Error reading CSV file: {e}")
```

**Notes:**

- This function first verifies the CSV file using verify_csv_file
- If the file is valid, it reads it directly with pandas
- If the file is invalid but fix_issues is True, it tries to read the fixed file
- If the file is invalid and couldn't be fixed, it raises a ValueError
- You can pass additional arguments to pandas.read_csv through **kwargs

### `clean_omniclass_csv(input_filepath: Union[str, Path], output_filepath: Optional[Union[str, Path]] = None, expected_columns: Optional[List[str]] = None) -> str`

Clean the OmniClass CSV file, handling specific issues with this format.

**Parameters:**

- `input_filepath` (Union[str, Path]): Path to the input OmniClass CSV file
- `output_filepath` (Optional[Union[str, Path]], optional): Path to save the cleaned file. If None, will use input_filepath with "_cleaned" appended.
- `expected_columns` (Optional[List[str]], optional): List of expected column names. Default is None.

**Returns:**

- str: Path to the cleaned CSV file

**Raises:**

- ValueError: If the file couldn't be cleaned

**Example:**
```python
from nexusml.utils.csv_utils import clean_omniclass_csv

try:
    # Clean an OmniClass CSV file
    cleaned_file = clean_omniclass_csv(
        "data/omniclass_table23.csv",
        output_filepath="data/omniclass_table23_cleaned.csv",
        expected_columns=["code", "title", "description"]
    )
    
    print(f"OmniClass CSV file cleaned and saved to: {cleaned_file}")
    
except ValueError as e:
    print(f"Error cleaning OmniClass CSV file: {e}")
```

**Notes:**

- This function is specifically designed for OmniClass CSV files, which have specific formatting requirements
- It determines the expected field count from the expected_columns or from the header row
- It uses verify_csv_file with fix_issues=True to clean the file
- If the file couldn't be cleaned, it raises a ValueError
- The function logs info messages for the cleaning process

## Command-Line Usage

The module can also be used as a command-line tool to clean and verify CSV files:

```bash
python -m nexusml.utils.csv_utils input_file.csv --output output_file.csv --fields 10 --columns "Asset Tag" "Description" "Service Life"
```

### Arguments

- `input_file`: Path to the input CSV file (required)
- `--output`, `-o`: Path to the output CSV file (optional)
- `--fields`, `-f`: Expected number of fields per row (optional)
- `--columns`, `-c`: Expected column names (optional, multiple values)

## Common CSV Issues Handled

The module handles the following common CSV issues:

1. **Inconsistent Field Counts**: Rows with different numbers of fields
2. **Missing Columns**: Required columns that are missing from the file
3. **Malformed CSV**: CSV files that pandas can't read directly
4. **Encoding Issues**: Attempts to read with UTF-8 encoding

## Fixing Strategies

When fix_issues is True, the module applies the following fixing strategies:

1. **Too Few Fields**: Add empty columns to rows with too few fields
2. **Too Many Fields**: Combine extra columns for rows with too many fields
3. **Missing Columns**: Log a warning (columns can't be added if they don't exist in the header)

## Dependencies

- **csv**: Standard library module for CSV file operations
- **logging**: Standard library module for logging
- **os**: Standard library module for file operations
- **pathlib**: Standard library module for path manipulation
- **typing**: Standard library module for type hints
- **pandas**: Used for DataFrame operations
- **nexusml.utils.logging**: Used for getting a logger

## Notes and Warnings

- The module assumes UTF-8 encoding for CSV files
- When fixing issues, the original file is not modified; a new file is created instead
- The module logs warnings for issues found and info messages for fixes applied
- The OmniClass cleaning function is specifically designed for OmniClass CSV files, which have specific formatting requirements
- When used as a command-line tool, the module configures basic logging to the console
- The module doesn't handle all possible CSV issues, just the most common ones
- If a CSV file has serious structural issues, the module may not be able to fix it
