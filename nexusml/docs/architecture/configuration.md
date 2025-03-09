# Configuration System

## Overview

NexusML uses a centralized configuration system to manage settings across the application. The system provides:

- YAML-based configuration files
- Environment variable overrides
- Schema validation
- Singleton access pattern
- Type-safe configuration objects

## Diagram

The following diagram illustrates the configuration system architecture:

- [Configuration System](../../diagrams/nexusml/configuration_system.puml) - Components and relationships of the configuration system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### ConfigProvider

Singleton provider that ensures only one configuration instance exists throughout the application.

```python
from nexusml.core.config.provider import ConfigProvider

# Get configuration
config = ConfigProvider.get_config()

# Initialize with specific config file (optional)
ConfigProvider.initialize("path/to/custom_config.yml")

# Reset provider (mainly for testing)
ConfigProvider.reset()
```

### Configuration

Root configuration object containing all configuration sections.

```python
# Access configuration sections
feature_config = config.feature_engineering
model_config = config.model_building
data_config = config.data_loading
reference_config = config.reference_data
paths_config = config.paths
```

### YAMLConfigLoader

Loads configuration from YAML files and applies environment variable overrides.

```python
from nexusml.core.config.loader import YAMLConfigLoader

# Create loader
loader = YAMLConfigLoader()

# Load configuration
config_dict = loader.load("path/to/config.yml")
```

## Configuration Files

### Default Configuration Files

NexusML looks for configuration files in the following locations (in order of precedence):

1. Custom path specified in `ConfigProvider.initialize()`
2. Environment variable `NEXUSML_CONFIG_PATH`
3. `nexusml_config.yml` in the current working directory
4. `config/nexusml_config.yml` in the package directory

### Configuration File Format

Configuration files use YAML format:

```yaml
# nexusml_config.yml
feature_engineering:
  text_columns:
    - description
    - name
  numerical_columns:
    - service_life
    - cost
  categorical_columns:
    - category
    - type
  transformers:
    text:
      type: tfidf
      max_features: 1000
    numerical:
      type: standard_scaler
    categorical:
      type: one_hot_encoder

model_building:
  model_type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
  optimization:
    method: grid_search
    cv: 5
    scoring: f1_macro

data_loading:
  encoding: utf-8
  delimiter: ","
  required_columns:
    - description
    - service_life
    - category

paths:
  output_dir: outputs
  models_dir: outputs/models
  data_dir: data
```

## Environment Variable Overrides

Configuration values can be overridden using environment variables:

```bash
# Override feature engineering settings
export NEXUSML_FEATURE_ENGINEERING_TEXT_COLUMNS=description,name,manufacturer
export NEXUSML_MODEL_BUILDING_HYPERPARAMETERS_N_ESTIMATORS=200
export NEXUSML_PATHS_OUTPUT_DIR=/custom/output/path
```

Environment variables use the format:
- Prefix: `NEXUSML_`
- Section and keys: Uppercase with underscores
- Nested keys: Separated by underscores
- Lists: Comma-separated values

## Schema Validation

Configuration files are validated against a JSON Schema to ensure correctness.

```python
from nexusml.core.config.schema import ConfigSchema

# Get schema
schema = ConfigSchema.get_schema()
```

Validation errors will be raised during configuration loading if the file doesn't match the schema.

## Configuration Sections

### FeatureEngineeringConfig

Controls feature engineering process.

```python
feature_config = config.feature_engineering

# Access settings
text_columns = feature_config.text_columns
numerical_columns = feature_config.numerical_columns
categorical_columns = feature_config.categorical_columns
transformers = feature_config.transformers
```

### ModelBuildingConfig

Controls model building and hyperparameters.

```python
model_config = config.model_building

# Access settings
model_type = model_config.model_type
hyperparameters = model_config.hyperparameters
optimization = model_config.optimization
```

### DataLoadingConfig

Controls data loading process.

```python
data_config = config.data_loading

# Access settings
encoding = data_config.encoding
delimiter = data_config.delimiter
required_columns = data_config.required_columns
```

### ReferenceDataConfig

Controls reference data sources.

```python
reference_config = config.reference_data

# Access settings
paths = reference_config.paths
mappings = reference_config.mappings
```

### PathsConfig

Controls file paths.

```python
paths_config = config.paths

# Access settings
output_dir = paths_config.output_dir
models_dir = paths_config.models_dir
data_dir = paths_config.data_dir
```

## Creating Custom Configuration

To add custom configuration sections:

1. Define a Pydantic model for the section:

```python
from pydantic import BaseModel
from typing import List, Dict, Any

class CustomConfig(BaseModel):
    setting1: str
    setting2: List[int]
    setting3: Dict[str, Any]
```

2. Add the section to the Configuration class:

```python
class Configuration(BaseModel):
    # Existing sections...
    custom: CustomConfig
```

3. Update the schema to include the new section:

```python
SCHEMA = {
    # Existing schema...
    "properties": {
        # Existing properties...
        "custom": {
            "type": "object",
            "properties": {
                "setting1": {"type": "string"},
                "setting2": {"type": "array", "items": {"type": "integer"}},
                "setting3": {"type": "object"}
            },
            "required": ["setting1", "setting2"]
        }
    }
}
```

4. Add the section to your configuration file:

```yaml
# Existing sections...
custom:
  setting1: value1
  setting2:
    - 1
    - 2
    - 3
  setting3:
    key1: value1
    key2: value2