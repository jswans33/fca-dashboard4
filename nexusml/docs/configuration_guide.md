# NexusML Configuration Guide

This guide provides detailed information about the NexusML configuration system,
including available configuration options, environment variable overrides, and
validation.

## Configuration Files

NexusML uses YAML files for configuration. The main configuration files are:

- `nexusml_config.yml` - Main configuration file for the pipeline
- `data_config.yml` - Configuration for data handling
- `feature_config.yml` - Configuration for feature engineering
- `classification_config.yml` - Configuration for model building and training
- `model_card_config.yml` - Configuration for model cards

## Configuration Sections

### Main Configuration (`nexusml_config.yml`)

The main configuration file contains the following sections:

#### Feature Engineering

```yaml
feature_engineering:
  text_combinations: [] # Text combination configurations
  numeric_columns: [] # Numeric column configurations
  hierarchies: [] # Hierarchy configurations
  column_mappings: [] # Column mapping configurations
  classification_systems: [] # Classification system configurations
  direct_mappings: [] # Direct mapping configurations
  eav_integration:
    enabled: false # Whether EAV integration is enabled
```

#### Classification

```yaml
classification:
  classification_targets: [] # Classification target configurations
  input_field_mappings: [] # Input field mapping configurations
```

#### Data

```yaml
data:
  required_columns:
    - name: id # Column name
      default_value: 0 # Default value for missing data
      data_type: int # Data type (str, int, float, bool, date, datetime)
    - name: name
      default_value: ''
      data_type: str
    - name: description
      default_value: ''
      data_type: str
    - name: category
      default_value: 'Unknown'
      data_type: str
    - name: value
      default_value: 0.0
      data_type: float
  training_data:
    default_path: 'nexusml/data/training_data/fake_training_data.csv' # Path to training data
    encoding: 'utf-8' # Encoding for training data
    fallback_encoding: 'latin1' # Fallback encoding if primary fails
```

#### Output

```yaml
output:
  output_dir: 'nexusml/output' # Default output directory for all outputs

  model:
    save_model: true # Whether to save the model
    model_dir: 'nexusml/output/models' # Directory to save models
    model_format: 'pickle' # Format to save models (pickle, joblib, onnx)

  results:
    save_results: true # Whether to save results
    results_dir: 'nexusml/output/results' # Directory to save results
    format: 'csv' # Format to save results (csv, json, excel)

  evaluation:
    save_evaluation: true # Whether to save evaluation
    evaluation_dir: 'nexusml/output/evaluation' # Directory to save evaluation
    format: 'json' # Format to save evaluation (json, csv, text)

  model_card:
    save_model_card: true # Whether to save model card
    model_card_dir: 'nexusml/output/model_cards' # Directory to save model cards
    format: 'json' # Format to save model cards (json, markdown, html)
```

### Data Configuration (`data_config.yml`)

The data configuration file contains settings for data handling, including
required columns, data types, and default values.

### Feature Configuration (`feature_config.yml`)

The feature configuration file contains settings for feature engineering,
including text combinations, numeric columns, hierarchies, and column mappings.

### Classification Configuration (`classification_config.yml`)

The classification configuration file contains settings for model building and
training, including model type, hyperparameters, and evaluation metrics.

### Model Card Configuration (`model_card_config.yml`)

The model card configuration file contains settings for model cards, including
model details, inputs, outputs, and performance metrics.

## Environment Variable Overrides

All configuration options can be overridden using environment variables. The
environment variables follow the pattern:

```
NEXUSML_CONFIG_[CONFIG_NAME]_[SECTION]_[KEY]
```

For example:

- `NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR=/custom/output/path` - Override the
  output directory
- `NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL=true` - Override whether to
  save the model
- `NEXUSML_CONFIG_DATA_TRAINING_DATA_DEFAULT_PATH=/path/to/data.csv` - Override
  the default training data path

The system will automatically convert the environment variable value to the
appropriate type based on the existing configuration value.

## Configuration Validation

NexusML validates configurations against JSON schemas to ensure they meet the
requirements of the pipeline. The validation checks:

- Required fields are present
- Field types are correct
- Field values are within allowed ranges
- Field values are in the correct format

If validation fails, an error message will be displayed indicating the issue.

## Environment-Specific Configuration

NexusML supports environment-specific configuration files. The environment is
determined by the `NEXUSML_ENV` environment variable, which defaults to
`production`.

For example, if `NEXUSML_ENV` is set to `development`, NexusML will look for
`nexusml_config.development.yml` before falling back to `nexusml_config.yml`.

## Configuration Manager

The `ConfigurationManager` class provides a centralized approach to
configuration management, with methods for loading, validating, and accessing
configurations.

Example usage:

```python
from nexusml.config.manager import ConfigurationManager

# Create a configuration manager
config_manager = ConfigurationManager()

# Get the pipeline configuration
pipeline_config = config_manager.get_pipeline_config()

# Get the output directory
output_dir = pipeline_config.output_dir

# Get the data configuration
data_config = config_manager.get_data_config()

# Get the required columns
required_columns = data_config.required_columns
```

## Best Practices

1. **Use environment variables for deployment-specific settings**: Use
   environment variables to override settings that vary between environments,
   such as file paths and database connections.

2. **Validate configurations**: Always validate configurations before using them
   to ensure they meet the requirements of the pipeline.

3. **Use environment-specific configuration files**: Use environment-specific
   configuration files for settings that vary between environments but don't
   change frequently.

4. **Document configuration options**: Document all configuration options,
   including their purpose, allowed values, and default values.

5. **Use sensible defaults**: Provide sensible defaults for all configuration
   options to make it easier to get started.
