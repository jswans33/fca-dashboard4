# Utility Module: path_utils

## Overview

The `path_utils` module provides robust path handling utilities for the NexusML system, ensuring consistent path resolution across different execution contexts such as scripts, notebooks, and applications. It offers functions to locate important directories, resolve paths, find data files, and set up the Python path for proper imports.

Key features include:

1. **Project Root Detection**: Reliably find the project root directory
2. **Path Resolution**: Convert relative paths to absolute paths
3. **Python Path Management**: Ensure the NexusML package is importable
4. **Data File Discovery**: Find data files in common locations
5. **Notebook Environment Setup**: Configure paths for Jupyter notebooks

## Functions

### `get_project_root() -> Path`

Get the absolute path to the project root directory.

**Returns:**

- Path: Path object pointing to the project root directory

**Example:**
```python
from nexusml.utils.path_utils import get_project_root

# Get the project root directory
project_root = get_project_root()
print(f"Project root: {project_root}")

# Use it to construct paths to other directories
data_dir = project_root / "data"
print(f"Data directory: {data_dir}")
```

**Notes:**

- This function assumes that the module is located in nexusml/utils/
- It navigates up two levels from the module directory to find the project root
- It returns a Path object, not a string

### `get_nexusml_root() -> Path`

Get the absolute path to the nexusml package root directory.

**Returns:**

- Path: Path object pointing to the nexusml package root

**Example:**
```python
from nexusml.utils.path_utils import get_nexusml_root

# Get the nexusml package root directory
nexusml_root = get_nexusml_root()
print(f"NexusML root: {nexusml_root}")

# Use it to construct paths to package components
utils_dir = nexusml_root / "utils"
print(f"Utils directory: {utils_dir}")
```

**Notes:**

- This function assumes that the module is located in nexusml/utils/
- It navigates up one level from the module directory to find the nexusml package root
- It returns a Path object, not a string
- This is useful when you need to access files within the nexusml package

### `ensure_nexusml_in_path() -> None`

Ensure that the nexusml package is in the Python path.

**Example:**
```python
from nexusml.utils.path_utils import ensure_nexusml_in_path

# Ensure nexusml is in the Python path
ensure_nexusml_in_path()

# Now you can import nexusml modules
import nexusml.core.model
```

**Notes:**

- This function adds the project root directory to the Python path if it's not already there
- It also adds the parent directory of the project root to support direct imports
- It prints a message when it adds a directory to the path
- This is particularly useful in Jupyter notebooks and scripts that need to import nexusml

### `resolve_path(path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path`

Resolve a path to an absolute path.

**Parameters:**

- `path` (Union[str, Path]): The path to resolve
- `relative_to` (Optional[Union[str, Path]], optional): The directory to resolve relative paths against. If None, uses the current working directory. Default is None.

**Returns:**

- Path: Resolved absolute Path object

**Example:**
```python
from nexusml.utils.path_utils import resolve_path

# Resolve a relative path against the current working directory
abs_path = resolve_path("data/sample.csv")
print(f"Absolute path: {abs_path}")

# Resolve a relative path against a specific directory
abs_path = resolve_path("sample.csv", relative_to="/path/to/data")
print(f"Absolute path: {abs_path}")

# Resolve an already absolute path
abs_path = resolve_path("/absolute/path/to/file.txt")
print(f"Absolute path: {abs_path}")
```

**Notes:**

- If relative_to is None, the path is resolved against the current working directory
- If relative_to is provided, the path is resolved against that directory
- The function returns a Path object, not a string
- This function is useful for ensuring consistent path handling regardless of the execution context

### `find_data_files(search_paths: Optional[List[Union[str, Path]]] = None, file_extensions: List[str] = [".xlsx", ".csv"], recursive: bool = False) -> Dict[str, str]`

Find data files in the specified search paths.

**Parameters:**

- `search_paths` (Optional[List[Union[str, Path]]], optional): List of paths to search. If None, uses default locations. Default is None.
- `file_extensions` (List[str], optional): List of file extensions to include. Default is [".xlsx", ".csv"].
- `recursive` (bool, optional): Whether to search recursively in subdirectories. Default is False.

**Returns:**

- Dict[str, str]: Dictionary mapping file names to their full paths

**Example:**
```python
from nexusml.utils.path_utils import find_data_files

# Find all Excel and CSV files in default locations
data_files = find_data_files()
print(f"Found {len(data_files)} data files")

# Find only CSV files in specific locations
csv_files = find_data_files(
    search_paths=["/path/to/data", "/path/to/uploads"],
    file_extensions=[".csv"]
)
print(f"Found {len(csv_files)} CSV files")

# Find all Excel files recursively
excel_files = find_data_files(
    file_extensions=[".xlsx"],
    recursive=True
)
print(f"Found {len(excel_files)} Excel files")

# Print all found files
for file_name, file_path in data_files.items():
    print(f"{file_name}: {file_path}")
```

**Notes:**

- If search_paths is None, it searches in the following default locations:
  - {project_root}/data
  - {project_root}/examples
  - {project_root_parent}/examples
  - {project_root_parent}/uploads
- The function only includes files with the specified extensions
- If recursive is True, it searches in subdirectories as well
- The returned dictionary uses file names as keys and full file paths as values
- This function is useful for discovering data files without knowing their exact locations

### `setup_notebook_environment() -> Dict[str, str]`

Set up the environment for Jupyter notebooks.

**Returns:**

- Dict[str, str]: Dictionary of useful paths for notebooks

**Example:**
```python
from nexusml.utils.path_utils import setup_notebook_environment

# Set up the notebook environment
paths = setup_notebook_environment()

# Access the paths
print(f"Project root: {paths['project_root']}")
print(f"NexusML root: {paths['nexusml_root']}")
print(f"Data directory: {paths['data_dir']}")
print(f"Examples directory: {paths['examples_dir']}")
print(f"Outputs directory: {paths['outputs_dir']}")
```

**Notes:**

- This function calls ensure_nexusml_in_path() to ensure that the nexusml package can be imported
- It returns a dictionary of useful paths for notebooks:
  - project_root: Absolute path to the project root directory
  - nexusml_root: Absolute path to the nexusml package root directory
  - data_dir: Path to the data directory
  - examples_dir: Path to the examples directory
  - outputs_dir: Path to the outputs directory
- All paths in the dictionary are strings, not Path objects
- This function is particularly useful at the beginning of Jupyter notebooks

## Usage Patterns

### Setting Up a Jupyter Notebook

```python
# Import path utilities
from nexusml.utils.path_utils import setup_notebook_environment

# Set up the notebook environment
paths = setup_notebook_environment()

# Now you can import nexusml modules
from nexusml.core.model import EquipmentClassifier
from nexusml.utils.data_selection import discover_and_load_data

# And use the paths to access files
import os
data_file = os.path.join(paths["data_dir"], "sample_data.csv")
```

### Finding and Loading Data Files

```python
from nexusml.utils.path_utils import find_data_files
import pandas as pd

# Find all Excel and CSV files
data_files = find_data_files(recursive=True)

# Filter for specific files
equipment_files = {name: path for name, path in data_files.items() if "equipment" in name.lower()}

# Load the first equipment file
if equipment_files:
    file_name = next(iter(equipment_files))
    file_path = equipment_files[file_name]
    print(f"Loading {file_name} from {file_path}")
    
    # Determine file type and load
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    
    print(f"Loaded data with shape: {df.shape}")
```

### Resolving Paths in Different Contexts

```python
from nexusml.utils.path_utils import resolve_path, get_project_root

# Get the project root
project_root = get_project_root()

# Resolve paths relative to the project root
data_path = resolve_path("data/sample.csv", relative_to=project_root)
output_path = resolve_path("outputs/results.json", relative_to=project_root)

# Ensure output directory exists
output_dir = output_path.parent
output_dir.mkdir(parents=True, exist_ok=True)

# Use the resolved paths
print(f"Data path: {data_path}")
print(f"Output path: {output_path}")
```

## Default Search Paths

When no search paths are specified for find_data_files(), the module searches for data files in the following default locations:

1. `{project_root}/data`: Project data directory
2. `{project_root}/examples`: Project examples directory
3. `{project_root_parent}/examples`: Parent examples directory
4. `{project_root_parent}/uploads`: Parent uploads directory

Where:

- `project_root` is the root directory of the NexusML project
- `project_root_parent` is the parent directory of the project root

## Command-Line Usage

The module can also be run as a script to print the project root and ensure nexusml is in the Python path:

```bash
python -m nexusml.utils.path_utils
```

This will output:
```
Project root: /path/to/project
NexusML root: /path/to/project/nexusml
Added /path/to/project to Python path
Added /path/to to Python path
Python path: ['/path/to/project', '/path/to', ...]
```

## Dependencies

- **os**: Standard library module for file operations
- **sys**: Standard library module for system-specific parameters and functions
- **pathlib**: Standard library module for path manipulation
- **typing**: Standard library module for type hints

## Notes and Warnings

- The module assumes a specific project structure with the module located in nexusml/utils/
- The get_project_root() and get_nexusml_root() functions rely on the module's location in the file system
- The ensure_nexusml_in_path() function modifies the Python path, which may affect other imports
- The find_data_files() function only includes files with the specified extensions
- The setup_notebook_environment() function is designed for use in Jupyter notebooks and may not be suitable for other contexts
- All paths returned by the functions are absolute paths to ensure consistency across different execution contexts
- The module uses pathlib.Path for path manipulation, which provides a more object-oriented approach than os.path
