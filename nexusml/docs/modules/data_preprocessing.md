# Module: data_preprocessing

## Overview

The `data_preprocessing` module handles loading and preprocessing data for the equipment classification model. It follows the Single Responsibility Principle by focusing solely on data loading and cleaning operations. This module provides functions to:

1. Load data configuration from YAML files
2. Verify and create required columns in datasets
3. Load and preprocess data from CSV files with proper encoding handling

## Functions

### `load_data_config() -> Dict`

Load the data preprocessing configuration from a YAML file.

**Returns:**
- Dict: Configuration dictionary containing settings for data preprocessing

**Example:**
```python
from nexusml.core.data_preprocessing import load_data_config

# Load the configuration
config = load_data_config()

# Access configuration settings
required_columns = config.get("required_columns", [])
training_data_config = config.get("training_data", {})
```

**Notes:**
- The function looks for a configuration file at `nexusml/config/data_config.yml`
- If the configuration file is not found or cannot be loaded, a minimal default configuration is returned
- The default configuration includes empty required columns and default paths for training data

### `verify_required_columns(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame`

Verify that all required columns exist in the DataFrame and create them if they don't.

**Parameters:**
- `df` (pd.DataFrame): Input DataFrame to verify
- `config` (Dict, optional): Configuration dictionary. If None, loads from file.

**Returns:**
- pd.DataFrame: DataFrame with all required columns

**Example:**
```python
import pandas as pd
from nexusml.core.data_preprocessing import verify_required_columns, load_data_config

# Load sample data
data = pd.DataFrame({
    "equipment_tag": ["AHU-01", "CHW-02"],
    "category_name": ["Air Handler", "Chiller"]
})

# Load configuration
config = load_data_config()

# Verify and create required columns
processed_data = verify_required_columns(data, config)
```

**Notes:**
- The function creates a copy of the input DataFrame to avoid modifying the original
- For each required column defined in the configuration, it checks if the column exists
- If a required column is missing, it creates the column with the specified default value and data type
- Supported data types are "str", "float", and "int"
- If an unknown data type is specified, it defaults to string

### `load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame`

Load and preprocess data from a CSV file.

**Parameters:**
- `data_path` (str, optional): Path to the CSV file. If None, uses the default path from configuration.

**Returns:**
- pd.DataFrame: Preprocessed DataFrame ready for model training or prediction

**Raises:**
- FileNotFoundError: If the data file cannot be found at the specified path

**Example:**
```python
from nexusml.core.data_preprocessing import load_and_preprocess_data

# Load data from default path
data = load_and_preprocess_data()

# Load data from custom path
custom_data = load_and_preprocess_data("path/to/custom_data.csv")
```

**Notes:**
- If no data path is provided, the function tries to determine the path in the following order:
  1. From the fca_dashboard settings.yml file (if running in that context)
  2. From the default path specified in the data configuration
  3. Fallback to a hardcoded default path
- The function handles encoding issues by first trying with the primary encoding (default: utf-8)
- If the primary encoding fails, it falls back to an alternative encoding (default: latin1)
- Column names are cleaned by removing leading and trailing whitespace
- NaN values in text columns are replaced with empty strings
- Required columns are verified and created if missing

## Usage Examples

### Basic Data Loading

```python
from nexusml.core.data_preprocessing import load_and_preprocess_data

# Load data from default path
data = load_and_preprocess_data()

# Print the first few rows
print(data.head())

# Check the columns
print(data.columns.tolist())
```

### Custom Configuration and Data Path

```python
import pandas as pd
from nexusml.core.data_preprocessing import verify_required_columns, load_and_preprocess_data

# Define custom configuration
custom_config = {
    "required_columns": [
        {"name": "equipment_tag", "default_value": "UNKNOWN", "data_type": "str"},
        {"name": "category_name", "default_value": "Unknown Equipment", "data_type": "str"},
        {"name": "service_life", "default_value": "20.0", "data_type": "float"},
        {"name": "condition_score", "default_value": "3", "data_type": "int"}
    ],
    "training_data": {
        "encoding": "utf-8",
        "fallback_encoding": "latin1"
    }
}

# Load data from custom path
data = load_and_preprocess_data("path/to/custom_data.csv")

# Verify and create required columns with custom configuration
processed_data = verify_required_columns(data, custom_config)

# Check that all required columns exist
for column_info in custom_config["required_columns"]:
    column_name = column_info["name"]
    assert column_name in processed_data.columns, f"Column {column_name} is missing"
```

### Complete Data Preprocessing Pipeline

```python
import pandas as pd
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.data_mapper import map_staging_to_model_input
from nexusml.core.feature_engineering import GenericFeatureEngineer

# Step 1: Load and preprocess the data
raw_data = load_and_preprocess_data("path/to/equipment_data.csv")

# Step 2: Map staging data to model input format
mapped_data = map_staging_to_model_input(raw_data)

# Step 3: Apply feature engineering
feature_engineer = GenericFeatureEngineer()
engineered_data = feature_engineer.transform(mapped_data)

# Step 4: Check the final processed data
print(f"Processed {len(engineered_data)} equipment records")
print(f"Final columns: {engineered_data.columns.tolist()}")
```

## Dependencies

- **pandas**: Used for DataFrame operations and data manipulation
- **yaml**: Used for loading configuration from YAML files
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **os**: Used for file operations
- **fca_dashboard.utils.path_util** (optional): Used for path resolution when running in the fca_dashboard context

## Notes and Warnings

- The module attempts to handle encoding issues automatically by trying multiple encodings, but in some cases, you may need to specify the correct encoding manually.
- If running outside the fca_dashboard context, the module will use its own configuration and default paths.
- The module creates a copy of the input DataFrame to avoid modifying the original data.
- Required columns are defined in the configuration file and are automatically created with default values if missing.
- The module assumes CSV files as the data source. For other file formats, you would need to modify the code or create a new module.