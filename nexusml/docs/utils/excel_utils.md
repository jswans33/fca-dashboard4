# Utility Module: excel_utils

## Overview

The `excel_utils` module provides utilities for working with Excel files in the NexusML system, particularly for data extraction and cleaning. It offers functions to handle common Excel-related tasks such as extracting data with specific configurations, normalizing sheet names, and cleaning DataFrames.

Key features include:

1. **Excel Data Extraction**: Extract data from Excel files with configurable options
2. **Sheet Name Management**: Get, normalize, and find specific sheet names
3. **DataFrame Cleaning**: Clean and standardize DataFrames extracted from Excel files
4. **Path Resolution**: Resolve file paths to absolute paths
5. **OmniClass Data Handling**: Special handling for OmniClass data

## Classes

### `DataCleaningError`

Exception raised for errors in the data cleaning process.

**Example:**
```python
from nexusml.utils.excel_utils import DataCleaningError

try:
    # Some data cleaning operation
    if problem_detected:
        raise DataCleaningError("Failed to clean data: invalid format")
except DataCleaningError as e:
    print(f"Data cleaning error: {e}")
```

**Notes:**

- This is a custom exception class that extends the base Exception class
- It doesn't add any additional functionality but provides a specific exception type for data cleaning errors

## Functions

### `get_logger(name: str)`

Simple logger function.

**Parameters:**

- `name` (str): Name for the logger

**Returns:**

- Logger: Configured logger instance

**Example:**
```python
from nexusml.utils.excel_utils import get_logger

# Get a logger for the current module
logger = get_logger(__name__)

# Use the logger
logger.info("Processing Excel file...")
logger.warning("Missing expected column")
```

**Notes:**

- This function configures basic logging with INFO level
- It returns a logger with the specified name
- This is a utility function used internally by the module

### `resolve_path(path: Union[str, Path, None]) -> Path`

Resolve a path to an absolute path.

**Parameters:**

- `path` (Union[str, Path, None]): The path to resolve. If None, returns the current working directory.

**Returns:**

- Path: The resolved path as a Path object

**Example:**
```python
from nexusml.utils.excel_utils import resolve_path
from pathlib import Path

# Resolve a string path
abs_path = resolve_path("data/excel_file.xlsx")
print(f"Absolute path: {abs_path}")

# Resolve a Path object
path_obj = Path("data/excel_file.xlsx")
abs_path = resolve_path(path_obj)
print(f"Absolute path: {abs_path}")

# Get current working directory
cwd = resolve_path(None)
print(f"Current working directory: {cwd}")
```

**Notes:**

- If path is None, returns the current working directory
- If path is a string, converts it to a Path object
- Always returns an absolute path using Path.resolve()

### `get_sheet_names(file_path: Union[str, Path]) -> List[str]`

Get sheet names from an Excel file.

**Parameters:**

- `file_path` (Union[str, Path]): Path to the Excel file

**Returns:**

- List[str]: List of sheet names as strings

**Example:**
```python
from nexusml.utils.excel_utils import get_sheet_names

# Get all sheet names from an Excel file
sheet_names = get_sheet_names("data/equipment_data.xlsx")
print(f"Excel file contains {len(sheet_names)} sheets:")
for name in sheet_names:
    print(f"  - {name}")
```

**Notes:**

- Uses pandas.ExcelFile to read the sheet names
- Converts all sheet names to strings to ensure type safety
- This function is useful for exploring Excel files before extraction

### `extract_excel_with_config(file_path: Union[str, Path], config: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]`

Extract data from Excel file using a configuration.

**Parameters:**

- `file_path` (Union[str, Path]): Path to the Excel file
- `config` (Dict[str, Dict[str, Any]]): Configuration dictionary with sheet names as keys and sheet configs as values. Each sheet config can have the following keys:
  - header_row: Row index to use as header (default: 0)
  - drop_empty_rows: Whether to drop empty rows (default: False)
  - strip_whitespace: Whether to strip whitespace from string columns (default: False)

**Returns:**

- Dict[str, pd.DataFrame]: Dictionary with sheet names as keys and DataFrames as values

**Example:**
```python
from nexusml.utils.excel_utils import extract_excel_with_config

# Define extraction configuration
config = {
    "Equipment": {
        "header_row": 1,  # Header is in the second row (0-indexed)
        "drop_empty_rows": True,
        "strip_whitespace": True
    },
    "Systems": {
        "header_row": 0,  # Header is in the first row
        "drop_empty_rows": True,
        "strip_whitespace": False
    }
}

# Extract data from Excel file using the configuration
data = extract_excel_with_config("data/equipment_data.xlsx", config)

# Access extracted DataFrames
equipment_df = data["Equipment"]
systems_df = data["Systems"]

print(f"Equipment data shape: {equipment_df.shape}")
print(f"Systems data shape: {systems_df.shape}")
```

**Notes:**

- This function allows for flexible extraction of data from multiple sheets
- Each sheet can have its own configuration for header row, empty row handling, and whitespace stripping
- The function returns a dictionary with sheet names as keys and the extracted DataFrames as values
- If a sheet specified in the config doesn't exist in the Excel file, pandas will raise a ValueError

### `normalize_sheet_names(file_path: Union[str, Path]) -> Dict[str, str]`

Normalize sheet names in an Excel file.

**Parameters:**

- `file_path` (Union[str, Path]): Path to the Excel file

**Returns:**

- Dict[str, str]: Dictionary mapping original sheet names to normalized names

**Example:**
```python
from nexusml.utils.excel_utils import normalize_sheet_names

# Get normalized sheet names
name_mapping = normalize_sheet_names("data/equipment_data.xlsx")
print("Sheet name mapping:")
for original, normalized in name_mapping.items():
    print(f"  {original} -> {normalized}")
```

**Notes:**

- Normalization converts sheet names to lowercase and replaces spaces with underscores
- This is useful for creating consistent keys when working with sheet names
- The function returns a dictionary mapping original sheet names to their normalized versions

### `find_flat_sheet(sheet_names: List[str]) -> Optional[str]`

Find the sheet name that contains 'FLAT' in it.

**Parameters:**

- `sheet_names` (List[str]): List of sheet names to search through

**Returns:**

- Optional[str]: The name of the sheet containing 'FLAT', or None if not found

**Example:**
```python
from nexusml.utils.excel_utils import get_sheet_names, find_flat_sheet

# Get all sheet names
sheet_names = get_sheet_names("data/equipment_data.xlsx")

# Find the flat sheet
flat_sheet = find_flat_sheet(sheet_names)
if flat_sheet:
    print(f"Found flat sheet: {flat_sheet}")
else:
    print("No flat sheet found")
```

**Notes:**

- This function is useful for finding "flat" data sheets in Excel files
- It searches for the string "FLAT" in uppercase within each sheet name
- Returns the first matching sheet name, or None if no match is found
- This is particularly useful for files that contain both hierarchical and flat data representations

### `clean_dataframe(df: pd.DataFrame, header_patterns: Optional[List[str]] = None, copyright_patterns: Optional[List[str]] = None, column_mapping: Optional[Dict[str, str]] = None, is_omniclass: bool = False) -> pd.DataFrame`

Clean a DataFrame.

**Parameters:**

- `df` (pd.DataFrame): The DataFrame to clean
- `header_patterns` (Optional[List[str]], optional): List of patterns to identify the header row. Default is None.
- `copyright_patterns` (Optional[List[str]], optional): List of patterns to identify copyright rows. Default is None.
- `column_mapping` (Optional[Dict[str, str]], optional): Dictionary mapping original column names to standardized names. Default is None.
- `is_omniclass` (bool, optional): Whether the DataFrame contains OmniClass data, which requires special handling. Default is False.

**Returns:**

- pd.DataFrame: A cleaned DataFrame

**Example:**
```python
from nexusml.utils.excel_utils import clean_dataframe
import pandas as pd

# Load a DataFrame
df = pd.read_excel("data/equipment_data.xlsx")

# Clean the DataFrame
cleaned_df = clean_dataframe(
    df,
    header_patterns=["ID", "Name", "Description"],
    copyright_patterns=["©", "Copyright"],
    column_mapping={"ID": "asset_id", "Name": "asset_name"},
    is_omniclass=False
)

print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {cleaned_df.shape}")
```

**Notes:**

- This function performs basic cleaning operations on a DataFrame
- It drops completely empty rows
- If is_omniclass is True, it applies special handling for OmniClass data:
  - Looks for common OmniClass column names and renames them to standardized names
  - Columns containing "number" are renamed to "OmniClass_Code"
  - Columns containing "title" are renamed to "OmniClass_Title"
  - Columns containing "definition" are renamed to "Description"
- The function returns a new DataFrame, leaving the original unchanged

### `standardize_column_names(df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame`

Standardize column names in a DataFrame.

**Parameters:**

- `df` (pd.DataFrame): The DataFrame to standardize
- `column_mapping` (Optional[Dict[str, str]], optional): Dictionary mapping original column names to standardized names. If None, uses default mapping. Default is None.

**Returns:**

- pd.DataFrame: A new DataFrame with standardized column names

**Example:**
```python
from nexusml.utils.excel_utils import standardize_column_names
import pandas as pd

# Load a DataFrame
df = pd.read_excel("data/equipment_data.xlsx")

# Define column mapping
column_mapping = {
    "asset_id": "ID",
    "asset_name": "Name",
    "asset_description": "Description"
}

# Standardize column names
standardized_df = standardize_column_names(df, column_mapping)

print("Original columns:", df.columns.tolist())
print("Standardized columns:", standardized_df.columns.tolist())
```

**Notes:**

- This function renames columns in a DataFrame based on a mapping
- The column_mapping dictionary should map standardized names to original names
- The function inverts the mapping to rename columns from original to standardized
- If column_mapping is None, no changes are made
- The function returns a new DataFrame with renamed columns

## Common Use Cases

### 1. Extracting Data from Multiple Sheets

```python
from nexusml.utils.excel_utils import extract_excel_with_config

# Define extraction configuration for multiple sheets
config = {
    "Equipment": {
        "header_row": 1,
        "drop_empty_rows": True,
        "strip_whitespace": True
    },
    "Systems": {
        "header_row": 0,
        "drop_empty_rows": True
    },
    "Locations": {
        "header_row": 0,
        "strip_whitespace": True
    }
}

# Extract data from all sheets
data = extract_excel_with_config("data/facility_data.xlsx", config)

# Process each sheet
for sheet_name, df in data.items():
    print(f"Processing sheet: {sheet_name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
```

### 2. Finding and Processing a Specific Sheet

```python
from nexusml.utils.excel_utils import get_sheet_names, find_flat_sheet, extract_excel_with_config

# Get all sheet names
sheet_names = get_sheet_names("data/complex_data.xlsx")

# Find the flat sheet
flat_sheet = find_flat_sheet(sheet_names)
if flat_sheet:
    print(f"Found flat sheet: {flat_sheet}")
    
    # Extract only the flat sheet
    config = {
        flat_sheet: {
            "header_row": 0,
            "drop_empty_rows": True,
            "strip_whitespace": True
        }
    }
    
    data = extract_excel_with_config("data/complex_data.xlsx", config)
    flat_df = data[flat_sheet]
    
    # Process the flat data
    print(f"Flat data shape: {flat_df.shape}")
else:
    print("No flat sheet found")
```

### 3. Cleaning and Standardizing OmniClass Data

```python
from nexusml.utils.excel_utils import extract_excel_with_config, clean_dataframe

# Extract OmniClass data
config = {
    "OmniClass": {
        "header_row": 0,
        "drop_empty_rows": True
    }
}

data = extract_excel_with_config("data/omniclass_table23.xlsx", config)
omniclass_df = data["OmniClass"]

# Clean and standardize OmniClass data
cleaned_df = clean_dataframe(
    omniclass_df,
    is_omniclass=True
)

print("Original columns:", omniclass_df.columns.tolist())
print("Cleaned columns:", cleaned_df.columns.tolist())
```

## Dependencies

- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **pandas**: Used for DataFrame operations and Excel file handling
- **logging**: Used for logging (imported in get_logger function)

## Notes and Warnings

- The module assumes that Excel files are well-formed and can be read by pandas
- Some functions, like clean_dataframe, have parameters that are not fully utilized in the current implementation
- The OmniClass-specific handling in clean_dataframe is based on common column naming patterns and may not work for all OmniClass files
- The standardize_column_names function inverts the column_mapping dictionary, which may cause issues if the mapping is not one-to-one
- The module doesn't handle Excel files with password protection or other security features
- For large Excel files, memory usage may be a concern, especially when extracting data from multiple sheets
