# Utility Module: data_selection

## Overview

The `data_selection` module provides utilities for finding and loading data files from different locations in the NexusML system. It simplifies the process of locating and loading data files, particularly useful in notebooks and interactive scripts where you need to quickly access available data.

Key features include:

1. **Data File Discovery**: Automatically find data files in common project locations
2. **File Format Handling**: Load data from different file formats (CSV, Excel)
3. **Interactive Selection**: List available data files and select one to load
4. **Flexible Path Resolution**: Find files relative to the project root

## Functions

### `get_project_root() -> str`

Get the absolute path to the project root directory.

**Returns:**

- str: Absolute path to the project root directory

**Example:**
```python
from nexusml.utils.data_selection import get_project_root

# Get the project root directory
project_root = get_project_root()
print(f"Project root: {project_root}")

# Use it to construct paths to other directories
import os
data_dir = os.path.join(project_root, "data")
print(f"Data directory: {data_dir}")
```

**Notes:**

- This function assumes that the module is located in nexusml/utils
- It navigates up two levels from the module directory to find the project root

### `find_data_files(locations: Optional[List[str]] = None, extensions: List[str] = [".xlsx", ".csv"]) -> Dict[str, str]`

Find all data files with specified extensions in the given locations.

**Parameters:**

- `locations` (Optional[List[str]], optional): List of directory paths to search. If None, uses default locations.
- `extensions` (List[str], optional): List of file extensions to include. Default is [".xlsx", ".csv"].

**Returns:**

- Dict[str, str]: Dictionary mapping file names to their full paths

**Example:**
```python
from nexusml.utils.data_selection import find_data_files

# Find all Excel and CSV files in default locations
data_files = find_data_files()
print(f"Found {len(data_files)} data files")

# Find only CSV files in specific locations
csv_files = find_data_files(
    locations=["/path/to/data", "/path/to/uploads"],
    extensions=[".csv"]
)
print(f"Found {len(csv_files)} CSV files")

# Print all found files
for file_name, file_path in data_files.items():
    print(f"{file_name}: {file_path}")
```

**Notes:**

- If locations is None, it searches in the following default locations:
  - {project_root_parent}/examples
  - {project_root_parent}/uploads
  - {project_root}/data
- The function only includes files with the specified extensions
- The returned dictionary uses file names as keys and full file paths as values

### `load_data(file_path: str) -> pd.DataFrame`

Load data from a file based on its extension.

**Parameters:**

- `file_path` (str): Path to the data file

**Returns:**

- pd.DataFrame: Pandas DataFrame containing the loaded data

**Raises:**

- ValueError: If the file format is not supported

**Example:**
```python
from nexusml.utils.data_selection import load_data

# Load an Excel file
excel_data = load_data("path/to/data.xlsx")
print(f"Loaded Excel data with shape: {excel_data.shape}")

# Load a CSV file
csv_data = load_data("path/to/data.csv")
print(f"Loaded CSV data with shape: {csv_data.shape}")

# Try to load an unsupported file format
try:
    unsupported_data = load_data("path/to/data.json")
except ValueError as e:
    print(f"Error: {e}")
```

**Notes:**

- The function determines the file format based on the file extension
- Currently supports .xlsx and .csv file formats
- For Excel files, it uses pandas.read_excel with default parameters
- For CSV files, it uses pandas.read_csv with default parameters
- Raises ValueError for unsupported file formats

### `list_available_data() -> Dict[str, str]`

List all available data files in the default locations.

**Returns:**

- Dict[str, str]: Dictionary mapping file names to their full paths

**Example:**
```python
from nexusml.utils.data_selection import list_available_data

# List all available data files
data_files = list_available_data()

# Use the returned dictionary to access specific files
if "sample_data.xlsx" in data_files:
    print(f"Sample data file path: {data_files['sample_data.xlsx']}")
```

**Notes:**

- This function calls find_data_files() with default parameters
- It prints a numbered list of available data files to the console
- The function returns the same dictionary as find_data_files()
- This is particularly useful in interactive environments like Jupyter notebooks

### `select_and_load_data(file_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]`

Select and load a data file.

**Parameters:**

- `file_name` (Optional[str], optional): Name of the file to load. If None, uses the first available file.

**Returns:**

- Tuple[pd.DataFrame, str]: Tuple of (loaded DataFrame, file path)

**Raises:**

- FileNotFoundError: If no data files are found or the specified file is not found

**Example:**
```python
from nexusml.utils.data_selection import select_and_load_data

# Load a specific file
try:
    data, data_path = select_and_load_data("sample_data.xlsx")
    print(f"Loaded data from {data_path} with shape {data.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}")

# Let it choose the first available file
try:
    data, data_path = select_and_load_data()
    print(f"Loaded data from {data_path} with shape {data.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}")
```

**Notes:**

- If file_name is None, the function selects the first available file
- If file_name is provided but not found, it raises FileNotFoundError
- The function prints information about the selected file and loaded data
- It returns both the loaded DataFrame and the file path for reference

## Usage in Notebooks

The module is particularly useful in Jupyter notebooks for quickly accessing and loading data files. Here's a typical usage pattern:

```python
from nexusml.utils.data_selection import list_available_data, select_and_load_data

# First, list all available data files to see what's available
list_available_data()

# Then, load a specific file based on the list
data, data_path = select_and_load_data("sample_data.xlsx")

# Or let it choose the first available file
data, data_path = select_and_load_data()

# Now you can work with the loaded data
print(f"Data columns: {data.columns.tolist()}")
print(f"Data preview:\n{data.head()}")
```

## Default Search Locations

When no locations are specified, the module searches for data files in the following default locations:

1. `{project_root_parent}/examples`: Example data files
2. `{project_root_parent}/uploads`: Uploaded data files
3. `{project_root}/data`: Project data files

Where:

- `project_root` is the root directory of the NexusML project
- `project_root_parent` is the parent directory of the project root

## Supported File Formats

The module currently supports the following file formats:

1. **Excel (.xlsx)**: Loaded using pandas.read_excel
2. **CSV (.csv)**: Loaded using pandas.read_csv

## Dependencies

- **os**: Standard library module for file operations
- **typing**: Standard library module for type hints
- **pandas**: Used for DataFrame operations and file loading

## Notes and Warnings

- The module assumes a specific project structure with the module located in nexusml/utils
- The default search locations may not exist in all installations
- The module only supports Excel and CSV file formats by default
- When listing available data files, the function prints to the console, which may not be ideal in all contexts
- The module does not handle file encoding issues or other advanced file loading options
- For more control over file loading, you may need to use pandas.read_csv or pandas.read_excel directly with additional parameters
