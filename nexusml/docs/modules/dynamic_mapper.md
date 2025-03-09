# Module: dynamic_mapper

## Overview

The `dynamic_mapper` module provides a flexible way to map input fields to the expected format for the ML model, regardless of the exact column names in the input data. This is particularly useful when dealing with data from different sources that may use different naming conventions for the same concepts.

The module uses pattern matching and configuration-driven mapping to identify the appropriate columns in the input data and transform them to the standardized format expected by the model.

## Classes

### Class: DynamicFieldMapper

Maps input data fields to model fields using flexible pattern matching based on configuration.

#### Attributes

- `config_path` (str): Path to the configuration YAML file
- `config` (Dict): Loaded configuration dictionary
- `field_mappings` (List[Dict]): List of field mapping configurations
- `classification_targets` (List[Dict]): List of classification target configurations

#### Methods

##### `__init__(config_path: Optional[str] = None)`

Initialize the mapper with a configuration file.

**Parameters:**

- `config_path` (Optional[str]): Path to the configuration YAML file. If None, uses the default path.

**Example:**
```python
from nexusml.core.dynamic_mapper import DynamicFieldMapper

# Use default configuration
mapper = DynamicFieldMapper()

# Use custom configuration
custom_mapper = DynamicFieldMapper("path/to/custom_config.yml")
```

##### `load_config() -> None`

Load the field mapping configuration from the specified or default path.

**Example:**
```python
from nexusml.core.dynamic_mapper import DynamicFieldMapper

mapper = DynamicFieldMapper()
# Reload configuration if needed
mapper.load_config()
```

**Notes:**

- This method is called automatically during initialization
- The default configuration path is `nexusml/config/classification_config.yml`
- The configuration should contain `input_field_mappings` and `classification_targets` sections

##### `get_best_match(available_columns: List[str], target_field: str) -> Optional[str]`

Find the best matching column for a target field.

**Parameters:**

- `available_columns` (List[str]): List of available column names
- `target_field` (str): Target field name to match

**Returns:**

- Optional[str]: Best matching column name or None if no match found

**Example:**
```python
from nexusml.core.dynamic_mapper import DynamicFieldMapper

mapper = DynamicFieldMapper()
available_columns = ["Asset Tag", "Equipment Type", "System"]
target_field = "Asset Category"

# Find the best match for "Asset Category"
best_match = mapper.get_best_match(available_columns, target_field)
print(f"Best match for '{target_field}': {best_match}")
```

**Notes:**

- The method first tries an exact match
- If no exact match is found, it tries pattern matching based on the configuration
- Pattern matching is case-insensitive

##### `map_dataframe(df: pd.DataFrame) -> pd.DataFrame`

Map input dataframe columns to the format expected by the ML model.

**Parameters:**

- `df` (pd.DataFrame): Input DataFrame with arbitrary column names

**Returns:**

- pd.DataFrame: DataFrame with columns mapped to what the model expects

**Example:**
```python
import pandas as pd
from nexusml.core.dynamic_mapper import DynamicFieldMapper

# Create sample data with arbitrary column names
data = pd.DataFrame({
    "Asset Tag": ["AHU-01", "CHW-02"],
    "Equipment Type": ["Air Handler", "Chiller"],
    "System": ["HVAC", "Cooling"],
    "Age": [10, 5]
})

# Map to standardized format
mapper = DynamicFieldMapper()
mapped_data = mapper.map_dataframe(data)
print(mapped_data.head())
```

**Notes:**

- The method creates a new DataFrame with the required fields
- Required fields are determined from the feature configuration file
- If a required field has no match in the input data, an empty column is created

##### `get_classification_targets() -> List[str]`

Get the list of classification targets.

**Returns:**

- List[str]: List of classification target names

**Example:**
```python
from nexusml.core.dynamic_mapper import DynamicFieldMapper

mapper = DynamicFieldMapper()
targets = mapper.get_classification_targets()
print(f"Classification targets: {targets}")
```

**Notes:**

- Classification targets are defined in the configuration file
- These are the fields that the model will predict

##### `get_required_db_fields() -> Dict[str, Dict]`

Get the mapping of classification targets to database fields.

**Returns:**

- Dict[str, Dict]: Dictionary mapping classification names to DB field info

**Example:**
```python
from nexusml.core.dynamic_mapper import DynamicFieldMapper

mapper = DynamicFieldMapper()
db_fields = mapper.get_required_db_fields()
print("Required database fields:")
for target, field_info in db_fields.items():
    print(f"  {target}: {field_info}")
```

**Notes:**

- This method returns only the targets marked as required in the configuration
- Each target includes information about the corresponding database fields

## Configuration Format

The DynamicFieldMapper relies on two configuration files:

### 1. Classification Configuration (classification_config.yml)

```yaml
# Example classification_config.yml
input_field_mappings:
  - target: "Asset Category"
    patterns:
      - "Equipment Type"
      - "Category"
      - "Asset Type"
      - "Equipment Category"
  
  - target: "Service Life"
    patterns:
      - "Expected Life"
      - "Life Expectancy"
      - "Useful Life"
      - "Age"

classification_targets:
  - name: "Equipment_Category"
    required: true
    master_db:
      table: "Equipment_Categories"
      field: "CategoryName"
      id_field: "CategoryID"
  
  - name: "System_Type"
    required: true
    master_db:
      table: "System_Types"
      field: "SystemName"
      id_field: "SystemID"
```

### 2. Feature Configuration (feature_config.yml)

```yaml
# Example feature_config.yml
text_combinations:
  - name: "equipment_description"
    columns:
      - "Asset Category"
      - "Equip Name ID"
      - "System Type ID"
      - "Precon System"
      - "Sub System Type"

numeric_columns:
  - name: "Service Life"
    default: 20.0

hierarchies:
  - name: "system_hierarchy"
    parents:
      - "System Type ID"
      - "Sub System Type"

column_mappings:
  - source: "Asset Category"
    target: "category_name"
  - source: "Service Life"
    target: "service_life"
```

## Usage Examples

### Basic Usage

```python
import pandas as pd
from nexusml.core.dynamic_mapper import DynamicFieldMapper

# Create sample data with arbitrary column names
data = pd.DataFrame({
    "Asset Tag": ["AHU-01", "CHW-02", "VAV-03"],
    "Equipment Type": ["Air Handler", "Chiller", "VAV Box"],
    "System": ["HVAC", "Cooling", "HVAC"],
    "Age": [10, 5, 3],
    "Location": ["Building A", "Building B", "Building A"]
})

# Create mapper with default configuration
mapper = DynamicFieldMapper()

# Map data to standardized format
mapped_data = mapper.map_dataframe(data)

# Get classification targets
targets = mapper.get_classification_targets()
print(f"Classification targets: {targets}")

# Get required database fields
db_fields = mapper.get_required_db_fields()
print("Required database fields:")
for target, field_info in db_fields.items():
    print(f"  {target}: {field_info}")
```

### Custom Configuration

```python
import pandas as pd
import yaml
from pathlib import Path
from nexusml.core.dynamic_mapper import DynamicFieldMapper

# Create a custom configuration file
custom_config = {
    "input_field_mappings": [
        {
            "target": "Asset Category",
            "patterns": ["Type", "Category", "Equipment"]
        },
        {
            "target": "Service Life",
            "patterns": ["Life", "Age", "Years"]
        },
        {
            "target": "Manufacturer",
            "patterns": ["Vendor", "Supplier", "Mfr"]
        }
    ],
    "classification_targets": [
        {
            "name": "Equipment_Type",
            "required": True,
            "master_db": {
                "table": "Equipment_Types",
                "field": "TypeName",
                "id_field": "TypeID"
            }
        }
    ]
}

# Save custom configuration
config_path = Path("custom_classification_config.yml")
with open(config_path, "w") as f:
    yaml.dump(custom_config, f)

# Create mapper with custom configuration
mapper = DynamicFieldMapper(str(config_path))

# Create sample data
data = pd.DataFrame({
    "ID": ["001", "002", "003"],
    "Type": ["Pump", "Fan", "Valve"],
    "Age": [15, 8, 12],
    "Vendor": ["Grundfos", "Trane", "Johnson Controls"]
})

# Map data
mapped_data = mapper.map_dataframe(data)
print(mapped_data.head())
```

### Integration with Data Preprocessing Pipeline

```python
import pandas as pd
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.dynamic_mapper import DynamicFieldMapper
from nexusml.core.feature_engineering import GenericFeatureEngineer

# Step 1: Load and preprocess the data
raw_data = load_and_preprocess_data("path/to/equipment_data.csv")

# Step 2: Map columns dynamically based on configuration
mapper = DynamicFieldMapper()
mapped_data = mapper.map_dataframe(raw_data)

# Step 3: Apply feature engineering
feature_engineer = GenericFeatureEngineer()
engineered_data = feature_engineer.transform(mapped_data)

# Step 4: Check the final processed data
print(f"Processed {len(engineered_data)} equipment records")
print(f"Final columns: {engineered_data.columns.tolist()}")

# Step 5: Get classification targets for model training
targets = mapper.get_classification_targets()
print(f"Model will predict: {targets}")
```

## Dependencies

- **pandas**: Used for DataFrame operations and data manipulation
- **yaml**: Used for loading configuration from YAML files
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **re**: Used for regular expression pattern matching (though not directly used in the current implementation)

## Notes and Warnings

- The module relies heavily on configuration files. Make sure these files are properly set up before using the mapper.
- The default configuration path is relative to the module location. If you're using the module from a different location, you may need to provide an absolute path.
- If a required field has no match in the input data, an empty column is created. This may lead to poor model performance if important fields are missing.
- The pattern matching is currently based on exact matches (case-insensitive). Future versions could implement more sophisticated matching using regular expressions or fuzzy matching.
- Changes to the configuration files require reloading the configuration (either by creating a new mapper instance or calling `load_config()`).
