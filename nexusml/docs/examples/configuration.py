#!/usr/bin/env python
"""
Configuration Example for NexusML

This example demonstrates how to use the configuration system in NexusML.
It covers:
- Creating custom configurations
- Loading configurations from files
- Environment variable configuration
- Configuration validation
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.config.configuration import (
    ClassificationConfig,
    ClassificationTarget,
    DataConfig,
    FeatureEngineeringConfig,
    NexusMLConfig,
    NumericColumn,
    RequiredColumn,
    TextCombination,
    TrainingDataConfig,
)
from nexusml.core.config.provider import ConfigurationProvider


def example_default_configuration():
    """Example of using the default configuration."""
    print("\n=== Example: Default Configuration ===")

    # Get the configuration provider
    config_provider = ConfigurationProvider()

    # Get the configuration
    config = config_provider.config

    # Print configuration details
    print(f"Configuration loaded from: {config_provider._load_config.__name__}")
    print("\nFeature Engineering Configuration:")
    print(f"  Text Combinations: {len(config.feature_engineering.text_combinations)}")
    print(f"  Numeric Columns: {len(config.feature_engineering.numeric_columns)}")

    print("\nClassification Configuration:")
    print(
        f"  Classification Targets: {len(config.classification.classification_targets)}"
    )
    print(f"  Input Field Mappings: {len(config.classification.input_field_mappings)}")

    print("\nData Configuration:")
    print(f"  Required Columns: {len(config.data.required_columns)}")
    print(f"  Training Data Path: {config.data.training_data.default_path}")

    return config


def example_custom_configuration():
    """Example of creating a custom configuration."""
    print("\n=== Example: Custom Configuration ===")

    # Create a custom configuration
    config = NexusMLConfig(
        feature_engineering=FeatureEngineeringConfig(
            text_combinations=[
                TextCombination(
                    name="combined_text",
                    columns=["description", "manufacturer", "model"],
                    separator=" | ",
                )
            ],
            numeric_columns=[
                NumericColumn(
                    name="service_life",
                    new_name=None,  # Optional, can be None
                    fill_value=15.0,
                    dtype="float",
                ),
                NumericColumn(
                    name="cost",
                    new_name="equipment_cost",
                    fill_value=0.0,
                    dtype="float",
                ),
            ],
        ),
        classification=ClassificationConfig(
            classification_targets=[
                ClassificationTarget(
                    name="category_name",
                    description="Equipment category",
                    required=True,
                    master_db=None,  # Optional, can be None
                ),
                ClassificationTarget(
                    name="uniformat_code",
                    description="UniFormat code",
                    required=False,
                    master_db=None,  # Optional, can be None
                ),
            ]
        ),
        data=DataConfig(
            required_columns=[
                RequiredColumn(
                    name="description", default_value="Unknown", data_type="str"
                ),
                RequiredColumn(
                    name="service_life", default_value=15.0, data_type="float"
                ),
            ],
            training_data=TrainingDataConfig(
                default_path="custom/path/to/training_data.csv",
                encoding="utf-8",
                fallback_encoding="latin1",
            ),
        ),
        # These are optional and can be None
        reference=None,
        equipment_attributes={},
        masterformat_primary=None,
        masterformat_equipment=None,
    )

    # Print configuration details
    print("\nFeature Engineering Configuration:")
    print(f"  Text Combinations: {len(config.feature_engineering.text_combinations)}")
    for tc in config.feature_engineering.text_combinations:
        print(f"    {tc.name}: {tc.columns} (separator: '{tc.separator}')")

    print(f"\n  Numeric Columns: {len(config.feature_engineering.numeric_columns)}")
    for nc in config.feature_engineering.numeric_columns:
        new_name = f" -> {nc.new_name}" if nc.new_name else ""
        print(f"    {nc.name}{new_name}: {nc.dtype} (fill: {nc.fill_value})")

    print("\nClassification Configuration:")
    print(
        f"  Classification Targets: {len(config.classification.classification_targets)}"
    )
    for ct in config.classification.classification_targets:
        print(f"    {ct.name}: {ct.description} (required: {ct.required})")

    print("\nData Configuration:")
    print(f"  Required Columns: {len(config.data.required_columns)}")
    for rc in config.data.required_columns:
        print(f"    {rc.name}: {rc.data_type} (default: {rc.default_value})")

    return config


def example_yaml_configuration():
    """Example of loading configuration from a YAML file."""
    print("\n=== Example: YAML Configuration ===")

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        yaml_path = f.name
        f.write(
            """
feature_engineering:
  text_combinations:
    - name: combined_text
      columns:
        - description
        - manufacturer
        - model
      separator: " | "
  numeric_columns:
    - name: service_life
      fill_value: 15.0
      dtype: float
    - name: cost
      new_name: equipment_cost
      fill_value: 0.0
      dtype: float

classification:
  classification_targets:
    - name: category_name
      description: Equipment category
      required: true
    - name: uniformat_code
      description: UniFormat code
      required: false

data:
  required_columns:
    - name: description
      default_value: Unknown
      data_type: str
    - name: service_life
      default_value: 15.0
      data_type: float
  training_data:
    default_path: custom/path/to/training_data.csv
    encoding: utf-8
    fallback_encoding: latin1
"""
        )

    try:
        # Load configuration from YAML file
        config = NexusMLConfig.from_yaml(yaml_path)

        # Print configuration details
        print(f"Configuration loaded from: {yaml_path}")
        print("\nFeature Engineering Configuration:")
        print(
            f"  Text Combinations: {len(config.feature_engineering.text_combinations)}"
        )
        for tc in config.feature_engineering.text_combinations:
            print(f"    {tc.name}: {tc.columns} (separator: '{tc.separator}')")

        print(f"\n  Numeric Columns: {len(config.feature_engineering.numeric_columns)}")
        for nc in config.feature_engineering.numeric_columns:
            new_name = f" -> {nc.new_name}" if nc.new_name else ""
            print(f"    {nc.name}{new_name}: {nc.dtype} (fill: {nc.fill_value})")

        print("\nTraining Data Configuration:")
        print(f"  Default Path: {config.data.training_data.default_path}")
        print(f"  Encoding: {config.data.training_data.encoding}")
        print(f"  Fallback Encoding: {config.data.training_data.fallback_encoding}")

        return config
    finally:
        # Clean up the temporary file
        os.unlink(yaml_path)


def example_environment_variable_configuration():
    """Example of using environment variables for configuration."""
    print("\n=== Example: Environment Variable Configuration ===")

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(suffix=".yml", mode="w", delete=False) as f:
        yaml_path = f.name
        f.write(
            """
feature_engineering:
  text_combinations:
    - name: combined_text
      columns:
        - description
      separator: " "
  numeric_columns: []

classification:
  classification_targets:
    - name: category_name
      description: Equipment category
      required: true

data:
  required_columns:
    - name: description
      default_value: Unknown
      data_type: str
  training_data:
    default_path: env/path/to/training_data.csv
    encoding: utf-8
    fallback_encoding: latin1
"""
        )

    # Save the original environment variable if it exists
    original_env = os.environ.get("NEXUSML_CONFIG")

    try:
        # Set the environment variable
        os.environ["NEXUSML_CONFIG"] = yaml_path

        # Reset the configuration provider to force reloading
        ConfigurationProvider.reset()

        # Get the configuration provider
        config_provider = ConfigurationProvider()

        # Get the configuration
        config = config_provider.config

        # Print configuration details
        print(f"Environment variable NEXUSML_CONFIG set to: {yaml_path}")
        print(f"Configuration loaded from environment variable")
        print("\nFeature Engineering Configuration:")
        print(
            f"  Text Combinations: {len(config.feature_engineering.text_combinations)}"
        )
        for tc in config.feature_engineering.text_combinations:
            print(f"    {tc.name}: {tc.columns} (separator: '{tc.separator}')")

        print("\nClassification Configuration:")
        print(
            f"  Classification Targets: {len(config.classification.classification_targets)}"
        )
        for ct in config.classification.classification_targets:
            print(f"    {ct.name}: {ct.description} (required: {ct.required})")

        print("\nTraining Data Configuration:")
        print(f"  Default Path: {config.data.training_data.default_path}")

        return config
    finally:
        # Clean up the temporary file
        os.unlink(yaml_path)

        # Restore the original environment variable
        if original_env is not None:
            os.environ["NEXUSML_CONFIG"] = original_env
        else:
            os.environ.pop("NEXUSML_CONFIG", None)

        # Reset the configuration provider
        ConfigurationProvider.reset()


def example_configuration_validation():
    """Example of configuration validation."""
    print("\n=== Example: Configuration Validation ===")

    valid_config = None

    # Try to create an invalid configuration
    try:
        # Invalid text combination (missing required field)
        invalid_config = NexusMLConfig(
            feature_engineering=FeatureEngineeringConfig(
                text_combinations=[
                    TextCombination(
                        # Missing name
                        name="",  # Empty name will cause validation error
                        columns=["description"],
                        separator=" ",
                    )
                ]
            ),
            # These are required for the constructor but not used in this example
            classification=ClassificationConfig(),
            data=DataConfig(),
            reference=None,
            equipment_attributes={},
            masterformat_primary=None,
            masterformat_equipment=None,
        )
        print("Configuration validation failed: Invalid configuration was accepted")
    except Exception as e:
        print(f"Configuration validation succeeded: {type(e).__name__}: {str(e)}")

    # Try to create another invalid configuration with a type error
    # We'll simulate this by describing what would happen, since we can't actually
    # create an invalid NumericColumn with a string fill_value due to type checking
    print("\nTrying to create a NumericColumn with string fill_value:")
    print(
        "This would raise a TypeError because fill_value must be a number (int or float)"
    )
    print("Example error: TypeError: 'fill_value' must be a number, not 'str'")

    # Create a valid configuration
    try:
        valid_config = NexusMLConfig(
            feature_engineering=FeatureEngineeringConfig(
                text_combinations=[
                    TextCombination(
                        name="combined_text", columns=["description"], separator=" "
                    )
                ],
                numeric_columns=[
                    NumericColumn(
                        name="service_life",
                        new_name=None,
                        fill_value=15.0,
                        dtype="float",
                    )
                ],
            ),
            # These are required for the constructor but not used in this example
            classification=ClassificationConfig(),
            data=DataConfig(),
            reference=None,
            equipment_attributes={},
            masterformat_primary=None,
            masterformat_equipment=None,
        )
        print("\nValid configuration created successfully")
    except Exception as e:
        print(f"Valid configuration creation failed: {type(e).__name__}: {str(e)}")

    return valid_config


def example_configuration_provider():
    """Example of using the configuration provider."""
    print("\n=== Example: Configuration Provider ===")

    # Create a custom configuration
    custom_config = NexusMLConfig(
        feature_engineering=FeatureEngineeringConfig(
            text_combinations=[
                TextCombination(
                    name="custom_text",
                    columns=["description", "notes"],
                    separator=" # ",
                )
            ]
        ),
        # These are required for the constructor but not used in this example
        classification=ClassificationConfig(),
        data=DataConfig(),
        reference=None,
        equipment_attributes={},
        masterformat_primary=None,
        masterformat_equipment=None,
    )

    # Get the configuration provider
    config_provider = ConfigurationProvider()

    # Set the custom configuration
    config_provider.set_config(custom_config)

    # Get the configuration
    config = config_provider.config

    # Print configuration details
    print("Custom configuration set through provider")
    print("\nFeature Engineering Configuration:")
    print(f"  Text Combinations: {len(config.feature_engineering.text_combinations)}")
    for tc in config.feature_engineering.text_combinations:
        print(f"    {tc.name}: {tc.columns} (separator: '{tc.separator}')")

    # Reset the configuration provider
    ConfigurationProvider.reset()

    # Get a new configuration provider
    new_config_provider = ConfigurationProvider()

    # Get the configuration
    new_config = new_config_provider.config

    # Print configuration details
    print("\nAfter reset:")
    print(f"Configuration loaded from: {new_config_provider._load_config.__name__}")
    print("\nFeature Engineering Configuration:")
    print(
        f"  Text Combinations: {len(new_config.feature_engineering.text_combinations)}"
    )

    return config


def main():
    """Main function to demonstrate configuration in NexusML."""
    print("NexusML Configuration Example")
    print("=============================")

    # Example 1: Default Configuration
    default_config = example_default_configuration()

    # Example 2: Custom Configuration
    custom_config = example_custom_configuration()

    # Example 3: YAML Configuration
    yaml_config = example_yaml_configuration()

    # Example 4: Environment Variable Configuration
    env_config = example_environment_variable_configuration()

    # Example 5: Configuration Validation
    validation_config = example_configuration_validation()

    # Example 6: Configuration Provider
    provider_config = example_configuration_provider()

    print("\n=== Configuration Example Completed ===")


if __name__ == "__main__":
    main()
