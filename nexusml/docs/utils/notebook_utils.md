# Utility Module: notebook_utils

## Overview

The `notebook_utils` module provides utility functions for use in Jupyter notebooks with the NexusML system. It offers a collection of helper functions that make notebooks more modular, maintainable, and consistent. These utilities handle common tasks such as environment setup, data loading, exploratory data analysis, pipeline component setup, and visualization.

Key features include:

1. **Notebook Environment Setup**: Configure matplotlib and seaborn for consistent visualization styling
2. **Data Discovery and Loading**: Find and load data files from various locations
3. **Exploratory Data Analysis**: Quickly analyze and summarize DataFrame contents
4. **Pipeline Component Setup**: Set up standard NexusML pipeline components for experiments
5. **Visualization Helpers**: Create common visualizations for model metrics and confusion matrices

## Functions

### `setup_notebook_environment()`

Set up the notebook environment with common configurations.

**Returns:**

- Dict: Dictionary of useful paths in the project

**Example:**
```python
from nexusml.utils.notebook_utils import setup_notebook_environment

# Set up the notebook environment
paths = setup_notebook_environment()

# Access the paths
print(f"Project root: {paths['project_root']}")
print(f"Data directory: {paths['data_dir']}")
print(f"Examples directory: {paths['examples_dir']}")
print(f"Outputs directory: {paths['outputs_dir']}")
```

**Notes:**

- This function sets up matplotlib with the "seaborn-v0_8-whitegrid" style
- It configures seaborn with the "notebook" context
- It returns a dictionary of useful paths in the project:
  - project_root: Absolute path to the project root directory
  - data_dir: Path to the data directory
  - examples_dir: Path to the examples directory
  - outputs_dir: Path to the outputs directory

### `get_project_root() -> str`

Get the absolute path to the project root directory.

**Returns:**

- str: Absolute path to the project root directory

**Example:**
```python
from nexusml.utils.notebook_utils import get_project_root

# Get the project root directory
project_root = get_project_root()
print(f"Project root: {project_root}")

# Use it to construct paths to other directories
import os
data_dir = os.path.join(project_root, "data")
print(f"Data directory: {data_dir}")
```

**Notes:**

- This function assumes that the module is located in nexusml/utils/
- It navigates up two levels from the module directory to find the project root

### `discover_and_load_data(file_name: Optional[str] = None, search_paths: Optional[List[str]] = None, file_extensions: Optional[List[str]] = None, show_available: bool = True) -> Tuple[pd.DataFrame, str]`

Discover available data files and load the specified one.

**Parameters:**

- `file_name` (Optional[str], optional): Name of the file to load. If None, uses the first available file. Default is None.
- `search_paths` (Optional[List[str]], optional): List of paths to search for data files. If None, uses default paths. Default is None.
- `file_extensions` (Optional[List[str]], optional): List of file extensions to include. If None, uses defaults. Default is None.
- `show_available` (bool, optional): Whether to print the list of available files. Default is True.

**Returns:**

- Tuple[pd.DataFrame, str]: Tuple of (loaded DataFrame, file path)

**Raises:**

- FileNotFoundError: If no data files are found or the specified file is not found

**Example:**
```python
from nexusml.utils.notebook_utils import discover_and_load_data

# Load the first available data file
data, file_path = discover_and_load_data()
print(f"Loaded data from {file_path} with shape {data.shape}")

# Load a specific file
data, file_path = discover_and_load_data(file_name="sample_data.xlsx")
print(f"Loaded data from {file_path} with shape {data.shape}")

# Search in specific paths and for specific file extensions
data, file_path = discover_and_load_data(
    search_paths=["data", "examples/data"],
    file_extensions=[".csv", ".xlsx"],
    show_available=True
)
print(f"Loaded data from {file_path} with shape {data.shape}")
```

**Notes:**

- This function uses the StandardDataLoader from nexusml.core.pipeline.components.data_loader
- It discovers available data files in the specified search paths
- If no search paths are specified, it uses the default paths from the data loader
- If no file extensions are specified, it uses the default extensions from the data loader
- If show_available is True, it prints a list of available data files
- If file_name is None, it uses the first available file
- If file_name is specified but not found, it raises a FileNotFoundError
- It returns both the loaded DataFrame and the file path

### `explore_data(data: pd.DataFrame, show_summary: bool = True, show_missing: bool = True) -> Dict`

Explore a DataFrame and return useful statistics.

**Parameters:**

- `data` (pd.DataFrame): DataFrame to explore
- `show_summary` (bool, optional): Whether to print summary statistics. Default is True.
- `show_missing` (bool, optional): Whether to print missing value information. Default is True.

**Returns:**

- Dict: Dictionary of exploration results

**Example:**
```python
from nexusml.utils.notebook_utils import discover_and_load_data, explore_data

# Load data
data, _ = discover_and_load_data()

# Explore the data
results = explore_data(data)

# Access specific exploration results
dtypes = results["dtypes"]
missing = results["missing"]
summary = results["summary"]

# Explore with custom options
results = explore_data(
    data,
    show_summary=True,
    show_missing=True
)
```

**Notes:**

- This function prints and returns various statistics about the DataFrame:
  - Data types for each column
  - Missing value counts and percentages (if show_missing is True)
  - Summary statistics (if show_summary is True)
- The returned dictionary contains:
  - "dtypes": Data types for each column
  - "missing": DataFrame with missing value counts and percentages (if show_missing is True)
  - "summary": Summary statistics from df.describe() (if show_summary is True)
- This function is useful for quickly understanding the structure and quality of a dataset

### `setup_pipeline_components()`

Set up the standard pipeline components for a NexusML experiment.

**Returns:**

- Dict: Dictionary containing the pipeline components

**Example:**
```python
from nexusml.utils.notebook_utils import setup_pipeline_components

# Set up pipeline components
components = setup_pipeline_components()

# Access specific components
registry = components["registry"]
container = components["container"]
factory = components["factory"]
context = components["context"]
orchestrator = components["orchestrator"]

# Use the orchestrator to run a pipeline
data, _ = discover_and_load_data()
results = orchestrator.run_pipeline(data)
```

**Notes:**

- This function sets up the standard pipeline components for a NexusML experiment:
  - ComponentRegistry: Registry for pipeline components
  - DIContainer: Dependency injection container
  - PipelineFactory: Factory for creating pipeline instances
  - PipelineContext: Context for sharing data between pipeline components
  - PipelineOrchestrator: Orchestrator for running pipelines
- It registers the StandardDataLoader for CSV and Excel files
- It attempts to import and register other components:
  - StandardDataPreprocessor
  - StandardFeatureEngineer
  - RandomForestModelBuilder
  - StandardModelTrainer
  - EnhancedModelEvaluator
  - PickleModelSerializer
- If a component cannot be imported, it logs a warning and continues
- The returned dictionary contains all the created components

### `visualize_metrics(metrics: Dict, figsize: Tuple[int, int] = (10, 6))`

Visualize model metrics.

**Parameters:**

- `metrics` (Dict): Dictionary of metrics
- `figsize` (Tuple[int, int], optional): Figure size as (width, height). Default is (10, 6).

**Example:**
```python
from nexusml.utils.notebook_utils import visualize_metrics

# Define metrics
metrics = {
    "Accuracy": 0.85,
    "Precision": 0.82,
    "Recall": 0.79,
    "F1 Score": 0.80,
    "AUC": 0.88
}

# Visualize metrics
visualize_metrics(metrics)

# Visualize with custom figure size
visualize_metrics(metrics, figsize=(12, 8))
```

**Notes:**

- This function creates a bar chart of the metrics
- It uses seaborn's barplot function
- The metrics dictionary should have metric names as keys and metric values as values
- The function displays the plot using plt.show()
- This is useful for quickly visualizing model performance metrics

### `visualize_confusion_matrix(cm, figsize: Tuple[int, int] = (10, 8))`

Visualize a confusion matrix.

**Parameters:**

- `cm`: Confusion matrix
- `figsize` (Tuple[int, int], optional): Figure size as (width, height). Default is (10, 8).

**Example:**
```python
from nexusml.utils.notebook_utils import visualize_confusion_matrix
import numpy as np

# Create a confusion matrix
cm = np.array([
    [50, 10, 5],
    [8, 45, 7],
    [4, 6, 40]
])

# Visualize the confusion matrix
visualize_confusion_matrix(cm)

# Visualize with custom figure size
visualize_confusion_matrix(cm, figsize=(12, 10))
```

**Notes:**

- This function creates a heatmap of the confusion matrix
- It uses seaborn's heatmap function with annotations
- The confusion matrix should be a numpy array or similar
- The function displays the plot using plt.show()
- This is useful for visualizing model classification performance

## Usage in Notebooks

The module is designed to be used in Jupyter notebooks to streamline common tasks. Here's a typical usage pattern:

```python
# Import notebook utilities
from nexusml.utils.notebook_utils import (
    setup_notebook_environment,
    discover_and_load_data,
    explore_data,
    setup_pipeline_components,
    visualize_metrics,
    visualize_confusion_matrix
)

# Set up the notebook environment
paths = setup_notebook_environment()

# Discover and load data
data, file_path = discover_and_load_data()

# Explore the data
exploration_results = explore_data(data)

# Set up pipeline components
components = setup_pipeline_components()
orchestrator = components["orchestrator"]

# Run a pipeline
results = orchestrator.run_pipeline(data)

# Visualize metrics
visualize_metrics(results["metrics"])

# Visualize confusion matrix
visualize_confusion_matrix(results["confusion_matrix"])
```

## Dependencies

- **logging**: Standard library module for logging
- **os**: Standard library module for file operations
- **pathlib**: Standard library module for path manipulation
- **typing**: Standard library module for type hints
- **matplotlib.pyplot**: Used for creating visualizations
- **pandas**: Used for DataFrame operations
- **seaborn**: Used for enhanced visualizations
- **nexusml.core.pipeline.components.data_loader**: Used for loading data
- **nexusml.core.di.container**: Used for dependency injection
- **nexusml.core.pipeline.context**: Used for pipeline context
- **nexusml.core.pipeline.factory**: Used for pipeline factory
- **nexusml.core.pipeline.interfaces**: Used for pipeline interfaces
- **nexusml.core.pipeline.orchestrator**: Used for pipeline orchestration
- **nexusml.core.pipeline.registry**: Used for component registry

## Notes and Warnings

- The module assumes a specific project structure with the module located in nexusml/utils/
- The setup_notebook_environment function sets matplotlib and seaborn styles, which may override existing settings
- The discover_and_load_data function prints information about available files and the loading process, which may produce verbose output
- The explore_data function prints various statistics, which may produce verbose output
- The setup_pipeline_components function attempts to import various components, which may produce warnings if components are not available
- The visualize_metrics and visualize_confusion_matrix functions display plots using plt.show(), which may not work in all notebook environments
- The module is designed for use in Jupyter notebooks and may not be suitable for other contexts
- Some functions may have dependencies on other parts of the NexusML system, which may cause errors if those parts are not available
