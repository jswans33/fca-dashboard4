# NexusML Configuration Migration Guide

## Introduction

This document provides guidance on migrating from the old configuration system
to the new unified configuration system in NexusML. It explains how to convert
old configuration files to the new format, how to use the migration utilities,
and provides examples of common migration scenarios.

The new configuration system centralizes all settings in a single file, provides
validation through Pydantic models, supports loading from YAML files or
environment variables, and ensures consistent access through a singleton
provider.

## Old vs. New Configuration System

### Old Configuration System

The old configuration system had several limitations:

1. **Scattered Configuration**: Configuration was scattered across multiple
   files:

   - `classification_config.yml`
   - `data_config.yml`
   - `feature_config.yml`
   - `reference_config.yml`
   - `eav/equipment_attributes.json`
   - `mappings/masterformat_primary.json`
   - `mappings/masterformat_equipment.json`

2. **Inconsistent Loading**: Each configuration file was loaded differently,
   with no consistent pattern.

3. **No Validation**: Configuration values were not validated, leading to
   runtime errors.

4. **No Default Values**: Many configuration values had no default values,
   requiring explicit configuration.

5. **No Centralized Access**: There was no centralized way to access
   configuration values.

### New Configuration System

The new configuration system addresses these limitations:

1. **Unified Configuration**: All configuration is centralized in a single file:
   `nexusml_config.yml`.

2. **Consistent Loading**: Configuration is loaded consistently through the
   `ConfigurationProvider` class.

3. **Validation**: Configuration values are validated through Pydantic models.

4. **Default Values**: All configuration values have default values, making
   explicit configuration optional.

5. **Centralized Access**: Configuration is accessed through the
   `ConfigurationProvider` singleton.

## Migration Utilities

The `nexusml.core.config.migration` module provides utilities for migrating from
the old configuration system to the new unified system.

### `migrate_from_default_paths`

The `migrate_from_default_paths` function migrates configurations from the
default paths of the old configuration files to the new unified format.

```python
from nexusml.core.config.migration import migrate_from_default_paths

# Migrate configurations and save to the default path
config = migrate_from_default_paths()
```

### `migrate_from_paths`

The `migrate_from_paths` function migrates configurations from specified paths
to the new unified format.

```python
from nexusml.core.config.migration import migrate_from_paths

# Migrate configurations and save to a specified path
config = migrate_from_paths(
    classification_config_path="path/to/classification_config.yml",
    data_config_path="path/to/data_config.yml",
    feature_config_path="path/to/feature_config.yml",
    reference_config_path="path/to/reference_config.yml",
    equipment_attributes_path="path/to/equipment_attributes.json",
    masterformat_primary_path="path/to/masterformat_primary.json",
    masterformat_equipment_path="path/to/masterformat_equipment.json",
    output_path="path/to/nexusml_config.yml"
)
```

### `migrate_from_dict`

The `migrate_from_dict` function migrates configurations from dictionaries to
the new unified format.

```python
from nexusml.core.config.migration import migrate_from_dict

# Migrate configurations from dictionaries
config = migrate_from_dict(
    classification_config=classification_config_dict,
    data_config=data_config_dict,
    feature_config=feature_config_dict,
    reference_config=reference_config_dict,
    equipment_attributes=equipment_attributes_dict,
    masterformat_primary=masterformat_primary_dict,
    masterformat_equipment=masterformat_equipment_dict
)
```

## Migration Process

The migration process consists of the following steps:

1. **Identify Configuration Files**: Identify all configuration files used in
   your code.
2. **Migrate Configuration Files**: Use the migration utilities to migrate the
   configuration files to the new unified format.
3. **Update Code**: Update your code to use the new configuration system.
4. **Test Configuration**: Test that the configuration is loaded and validated
   correctly.

### Step 1: Identify Configuration Files

First, identify all configuration files used in your code. This may include:

- `classification_config.yml`
- `data_config.yml`
- `feature_config.yml`
- `reference_config.yml`
- `eav/equipment_attributes.json`
- `mappings/masterformat_primary.json`
- `mappings/masterformat_equipment.json`

You may also have custom configuration files that are not part of the standard
set.

### Step 2: Migrate Configuration Files

Use the migration utilities to migrate the configuration files to the new
unified format.

```python
from nexusml.core.config.migration import migrate_from_paths

# Migrate configurations and save to a specified path
config = migrate_from_paths(
    classification_config_path="path/to/classification_config.yml",
    data_config_path="path/to/data_config.yml",
    feature_config_path="path/to/feature_config.yml",
    reference_config_path="path/to/reference_config.yml",
    equipment_attributes_path="path/to/equipment_attributes.json",
    masterformat_primary_path="path/to/masterformat_primary.json",
    masterformat_equipment_path="path/to/masterformat_equipment.json",
    output_path="path/to/nexusml_config.yml"
)
```

If you're using the default paths, you can use the `migrate_from_default_paths`
function:

```python
from nexusml.core.config.migration import migrate_from_default_paths

# Migrate configurations and save to the default path
config = migrate_from_default_paths()
```

### Step 3: Update Code

Update your code to use the new configuration system. This involves replacing
direct loading of configuration files with the `ConfigurationProvider`
singleton.

**Old Code**:

```python
import yaml

# Load classification configuration
with open("classification_config.yml", "r") as f:
    classification_config = yaml.safe_load(f)

# Load data configuration
with open("data_config.yml", "r") as f:
    data_config = yaml.safe_load(f)

# Load feature configuration
with open("feature_config.yml", "r") as f:
    feature_config = yaml.safe_load(f)

# Use configuration values
classification_targets = classification_config.get("classification_targets", [])
required_columns = data_config.get("required_columns", [])
text_combinations = feature_config.get("text_combinations", [])
```

**New Code**:

```python
from nexusml.core.config.provider import ConfigurationProvider

# Get the configuration provider
config_provider = ConfigurationProvider()

# Get the configuration
config = config_provider.config

# Use configuration values
classification_targets = config.classification.classification_targets
required_columns = config.data.required_columns
text_combinations = config.feature_engineering.text_combinations
```

### Step 4: Test Configuration

Test that the configuration is loaded and validated correctly. This involves
verifying that the configuration values are accessible and that validation
errors are raised for invalid values.

```python
from nexusml.core.config.provider import ConfigurationProvider

# Get the configuration provider
config_provider = ConfigurationProvider()

# Get the configuration
config = config_provider.config

# Verify that configuration values are accessible
assert config.classification.classification_targets is not None
assert config.data.required_columns is not None
assert config.feature_engineering.text_combinations is not None

# Verify that validation errors are raised for invalid values
try:
    config.data.required_columns = "invalid"  # Should raise a validation error
    assert False, "Validation error not raised"
except Exception:
    pass
```

## Common Migration Scenarios

### Scenario 1: Default Configuration

If you're using the default configuration files, you can use the
`migrate_from_default_paths` function to migrate to the new unified format.

```python
from nexusml.core.config.migration import migrate_from_default_paths

# Migrate configurations and save to the default path
config = migrate_from_default_paths()
```

### Scenario 2: Custom Configuration Paths

If you're using custom configuration paths, you can use the `migrate_from_paths`
function to migrate to the new unified format.

```python
from nexusml.core.config.migration import migrate_from_paths

# Migrate configurations and save to a specified path
config = migrate_from_paths(
    classification_config_path="custom/path/to/classification_config.yml",
    data_config_path="custom/path/to/data_config.yml",
    feature_config_path="custom/path/to/feature_config.yml",
    reference_config_path="custom/path/to/reference_config.yml",
    equipment_attributes_path="custom/path/to/equipment_attributes.json",
    masterformat_primary_path="custom/path/to/masterformat_primary.json",
    masterformat_equipment_path="custom/path/to/masterformat_equipment.json",
    output_path="custom/path/to/nexusml_config.yml"
)
```

### Scenario 3: Programmatic Configuration

If you're creating configuration programmatically, you can use the
`migrate_from_dict` function to migrate to the new unified format.

```python
from nexusml.core.config.migration import migrate_from_dict

# Create configuration dictionaries
classification_config = {
    "classification_targets": [
        {"name": "category_name", "description": "Equipment category", "required": True},
        {"name": "uniformat_code", "description": "UniFormat code", "required": False}
    ]
}

data_config = {
    "required_columns": [
        {"name": "description", "default_value": "Unknown", "data_type": "str"},
        {"name": "service_life", "default_value": 15.0, "data_type": "float"}
    ]
}

feature_config = {
    "text_combinations": [
        {"name": "combined_text", "columns": ["description", "manufacturer", "model"], "separator": " "}
    ]
}

# Migrate configurations
config = migrate_from_dict(
    classification_config=classification_config,
    data_config=data_config,
    feature_config=feature_config
)
```

### Scenario 4: Environment Variable Configuration

If you're using environment variables for configuration, you can set the
`NEXUSML_CONFIG` environment variable to the path of your configuration file.

```python
import os
from nexusml.core.config.provider import ConfigurationProvider

# Set the environment variable
os.environ["NEXUSML_CONFIG"] = "path/to/nexusml_config.yml"

# Get the configuration provider
config_provider = ConfigurationProvider()

# Get the configuration
config = config_provider.config
```

### Scenario 5: Custom Configuration Provider

If you need to customize the configuration provider, you can create a subclass
of `ConfigurationProvider` and override the `_load_config` method.

```python
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.config.configuration import NexusMLConfig

class CustomConfigurationProvider(ConfigurationProvider):
    """Custom configuration provider."""

    def _load_config(self) -> NexusMLConfig:
        """
        Load the configuration from a custom source.

        Returns:
            NexusMLConfig: The loaded configuration
        """
        # Custom loading logic
        return NexusMLConfig(
            feature_engineering=FeatureEngineeringConfig(
                text_combinations=[
                    TextCombination(
                        name="combined_text",
                        columns=["description", "manufacturer", "model"],
                        separator=" "
                    )
                ]
            ),
            classification=ClassificationConfig(
                classification_targets=[
                    ClassificationTarget(
                        name="category_name",
                        description="Equipment category",
                        required=True
                    ),
                    ClassificationTarget(
                        name="uniformat_code",
                        description="UniFormat code",
                        required=False
                    )
                ]
            ),
            data=DataConfig(
                required_columns=[
                    RequiredColumn(
                        name="description",
                        default_value="Unknown",
                        data_type="str"
                    ),
                    RequiredColumn(
                        name="service_life",
                        default_value=15.0,
                        data_type="float"
                    )
                ]
            )
        )
```

## Troubleshooting

### Missing Configuration Files

If you're missing some of the old configuration files, you can still migrate the
ones you have. The migration utilities will use default values for missing
files.

```python
from nexusml.core.config.migration import migrate_from_paths

# Migrate configurations with missing files
config = migrate_from_paths(
    classification_config_path="path/to/classification_config.yml",
    # Missing data_config_path
    feature_config_path="path/to/feature_config.yml",
    # Missing reference_config_path
    # Missing equipment_attributes_path
    # Missing masterformat_primary_path
    # Missing masterformat_equipment_path
    output_path="path/to/nexusml_config.yml"
)
```

### Invalid Configuration Values

If you have invalid configuration values, the migration utilities will raise
validation errors. You can catch these errors and fix the configuration values.

```python
from nexusml.core.config.migration import migrate_from_paths
from pydantic import ValidationError

try:
    # Migrate configurations
    config = migrate_from_paths(
        classification_config_path="path/to/classification_config.yml",
        data_config_path="path/to/data_config.yml",
        feature_config_path="path/to/feature_config.yml",
        reference_config_path="path/to/reference_config.yml",
        equipment_attributes_path="path/to/equipment_attributes.json",
        masterformat_primary_path="path/to/masterformat_primary.json",
        masterformat_equipment_path="path/to/masterformat_equipment.json",
        output_path="path/to/nexusml_config.yml"
    )
except ValidationError as e:
    # Handle validation errors
    print(f"Validation error: {e}")
    # Fix configuration values
    # ...
```

### Configuration Not Found

If the configuration file is not found, the `ConfigurationProvider` will raise a
`FileNotFoundError`. You can catch this error and create a default
configuration.

```python
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.config.configuration import NexusMLConfig

try:
    # Get the configuration provider
    config_provider = ConfigurationProvider()

    # Get the configuration
    config = config_provider.config
except FileNotFoundError:
    # Create a default configuration
    config = NexusMLConfig()

    # Set the configuration
    config_provider = ConfigurationProvider()
    config_provider.set_config(config)
```

## Conclusion

Migrating from the old configuration system to the new unified configuration
system is a straightforward process that can be done incrementally. By using the
migration utilities and updating your code to use the `ConfigurationProvider`
singleton, you can take advantage of the new configuration system's features
while maintaining backward compatibility.

For more information about the new configuration system, see the
[Configuration System](../architecture/configuration.md) documentation.
