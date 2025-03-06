# Feature Engineering in NexusML

This document explains the new config-driven feature engineering approach in
NexusML, which makes feature transformations more generic and reusable.

## Overview

The feature engineering system has been refactored to use a config-driven
approach with custom scikit-learn transformers. This makes it easier to:

1. Add new features without changing code
2. Reuse transformations across different projects
3. Maintain a clear separation between configuration and implementation
4. Create a more maintainable and extensible codebase

## Configuration File

Feature engineering is now controlled by a YAML configuration file located at
`nexusml/config/feature_config.yml`. This file defines:

- Text column combinations
- Numeric column transformations
- Hierarchical category creation
- Column mappings

Example configuration:

```yaml
text_combinations:
  - name: 'combined_text'
    columns:
      [
        'Asset Category',
        'Equip Name ID',
        'Sub System Type',
        'Sub System ID',
        'Title',
      ]
    separator: ' '

numeric_columns:
  - name: 'Service Life'
    new_name: 'service_life'
    fill_value: 0
    dtype: 'float'

hierarchies:
  - new_col: 'Equipment_Type'
    parents: ['Asset Category', 'Equip Name ID']
    separator: '-'

column_mappings:
  - source: 'Asset Category'
    target: 'Equipment_Category'
```

## Custom Transformers

The implementation uses scikit-learn custom transformers that extend
`BaseEstimator` and `TransformerMixin`. These include:

1. **TextCombiner**: Combines multiple text columns into one
2. **NumericCleaner**: Cleans and transforms numeric columns
3. **HierarchyBuilder**: Creates hierarchical category columns
4. **ColumnMapper**: Maps source columns to target columns
5. **GenericFeatureEngineer**: Orchestrates all transformations based on the
   config

## Using the Feature Engineering System

### Direct Usage

You can use the `GenericFeatureEngineer` class directly:

```python
from nexusml.core.feature_engineering import GenericFeatureEngineer

# Create a feature engineer with default config
engineer = GenericFeatureEngineer()

# Or specify a custom config path
# engineer = GenericFeatureEngineer(config_path="path/to/custom_config.yml")

# Transform a DataFrame
df_transformed = engineer.transform(df)
```

### In the Model Pipeline

The feature engineering is integrated into the model training pipeline:

```python
from nexusml.core.model import train_enhanced_model

# Train a model with default feature engineering
model, df = train_enhanced_model()

# Or specify a custom feature config
# model, df = train_enhanced_model(feature_config_path="path/to/custom_config.yml")
```

## Adding New Features

To add new features, simply update the configuration file. For example, to add a
new text combination:

```yaml
text_combinations:
  - name: 'combined_text'
    columns: ['Asset Category', 'Equip Name ID', 'Sub System Type']
    separator: ' '

  # Add a new text combination
  - name: 'technical_specs'
    columns: ['Technical Specifications', 'Manufacturer Notes']
    separator: ' | '
```

No code changes are required to add these new features!

## Backward Compatibility

For backward compatibility, the old functions `enhance_features()` and
`create_hierarchical_categories()` are still available, but they now use the new
config-driven approach internally.

## Example

See `nexusml/examples/feature_engineering_example.py` for a complete example of
how to use the new feature engineering system.

## Benefits

This refactoring provides several benefits:

1. **Flexibility**: Easily adapt to different data structures without code
   changes
2. **Maintainability**: Clear separation of configuration and implementation
3. **Reusability**: Transformers can be reused across different projects
4. **Extensibility**: New transformers can be added without changing existing
   code
5. **Testability**: Each transformer can be tested independently
