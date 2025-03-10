"""
Configuration Manager Module for NexusML

This module provides a centralized approach to configuration management,
implementing the ConfigurationManager class that loads and manages all configuration
files with type-safe access and validation.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

import yaml

from nexusml.config import get_project_root
from nexusml.config.sections import ConfigSection

T = TypeVar("T")


class DataConfig(ConfigSection):
    """Configuration for data handling."""

    @property
    def required_columns(self) -> list:
        """Get the required columns."""
        return [col["name"] for col in self.data.get("required_columns", [])]

    @property
    def source_columns(self) -> list:
        """Get only the source columns (not derived during feature engineering)."""
        return [
            col["name"]
            for col in self.data.get("required_columns", [])
            if not col["name"].startswith(
                ("Equipment_", "Uniformat_", "System_", "combined_", "service_life")
            )
        ]

    @property
    def target_columns(self) -> list:
        """Get only the target columns (derived during feature engineering)."""
        return [
            col["name"]
            for col in self.data.get("required_columns", [])
            if col["name"].startswith(
                ("Equipment_", "Uniformat_", "System_", "combined_", "service_life")
            )
        ]

    @property
    def critical_columns(self) -> list:
        """Get the critical columns that must not have missing values."""
        return ["equipment_tag", "category_name", "mcaa_system_category"]

    def get_column_default(self, column_name: str) -> Any:
        """Get the default value for a column."""
        for col in self.data.get("required_columns", []):
            if col["name"] == column_name:
                return col.get("default_value")
        return None

    def get_column_data_type(self, column_name: str) -> str:
        """Get the data type for a column."""
        for col in self.data.get("required_columns", []):
            if col["name"] == column_name:
                return col.get("data_type", "str")
        return "str"


class FeatureConfig(ConfigSection):
    """Configuration for feature engineering."""

    @property
    def text_combinations(self) -> list:
        """Get text combination configurations."""
        return self.data.get("text_combinations", [])

    @property
    def numeric_columns(self) -> list:
        """Get numeric column configurations."""
        return self.data.get("numeric_columns", [])

    @property
    def hierarchies(self) -> list:
        """Get hierarchy configurations."""
        return self.data.get("hierarchies", [])

    @property
    def column_mappings(self) -> list:
        """Get column mapping configurations."""
        return self.data.get("column_mappings", [])

    @property
    def classification_systems(self) -> list:
        """Get classification system configurations."""
        return self.data.get("classification_systems", [])

    @property
    def eav_integration_enabled(self) -> bool:
        """Check if EAV integration is enabled."""
        eav_config = self.data.get("eav_integration", {})
        return eav_config.get("enabled", False)


class ModelConfig(ConfigSection):
    """Configuration for model building and training."""

    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self.data.get("model_type", "random_forest")

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get the model hyperparameters."""
        return self.data.get("hyperparameters", {})

    @property
    def evaluation_metrics(self) -> list:
        """Get the evaluation metrics."""
        return self.data.get("evaluation_metrics", ["accuracy", "f1"])

    @property
    def cross_validation(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return self.data.get("cross_validation", {"enabled": False, "folds": 5})


class PipelineConfig(ConfigSection):
    """Configuration for pipeline orchestration."""

    @property
    def stages(self) -> list:
        """Get the pipeline stages."""
        return self.data.get("stages", [])

    @property
    def components(self) -> Dict[str, str]:
        """Get the component implementations to use."""
        return self.data.get("components", {})

    @property
    def output_dir(self) -> str:
        """Get the output directory."""
        # Check if output.output_dir exists
        if "output" in self.data and "output_dir" in self.data["output"]:
            return self.data["output"]["output_dir"]
        # Fall back to the old location
        return self.data.get("output_dir", "outputs")

    @property
    def visualizations_enabled(self) -> bool:
        """Check if visualizations are enabled."""
        return self.data.get("visualizations", {}).get("enabled", False)


class ConfigurationManager:
    """Manager for all configuration files."""

    def __init__(self):
        self.root = get_project_root()
        # Try to find the config directory
        nexusml_config_dir = self.root / "nexusml" / "config"
        if nexusml_config_dir.exists():
            self.config_dir = nexusml_config_dir
        else:
            self.config_dir = self.root / "config"

        self.configs = {}

        # Environment-specific configuration
        self.environment = os.environ.get("NEXUSML_ENV", "production")

        # Environment variable prefix for configuration overrides
        self.env_prefix = "NEXUSML_CONFIG_"

        # Print the config directory for debugging
        print(f"Using configuration directory: {self.config_dir}")

    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            Configuration as a dictionary

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        # Check if config is already loaded
        if name in self.configs:
            return self.configs[name]

        # Try environment-specific config first
        if self.environment != "production":
            env_path = self.config_dir / f"{name}.{self.environment}.yml"
            if env_path.exists():
                with open(env_path, "r") as f:
                    config = yaml.safe_load(f) or {}
                self.configs[name] = config
                return config

        # Try standard config
        path = self.config_dir / f"{name}.yml"
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        config = self._apply_env_overrides(config, name)

        self.configs[name] = config
        return config

    def _apply_env_overrides(
        self, config: Dict[str, Any], config_name: str
    ) -> Dict[str, Any]:
        """
        Apply environment variable overrides to the configuration.

        Environment variables should follow the pattern:
        NEXUSML_CONFIG_[CONFIG_NAME]_[SECTION]_[KEY]

        For example:
        NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR=/custom/output/path

        Args:
            config: The configuration dictionary to override
            config_name: The name of the configuration file

        Returns:
            The configuration with environment variable overrides applied
        """
        # Create a copy of the config to avoid modifying the original
        result = config.copy()

        # Convert config name to uppercase for environment variable matching
        # The config_name is already "nexusml_config", so we need to use just "NEXUSML" for the prefix
        if config_name.upper().startswith("NEXUSML"):
            # For nexusml_config, use just NEXUSML as the prefix
            config_prefix = f"{self.env_prefix}NEXUSML_"
        else:
            config_prefix = f"{self.env_prefix}{config_name.upper()}_"
        print(f"\nLooking for environment variables with prefix: {config_prefix}")

        # Print all environment variables for debugging
        print("\nAll environment variables:")
        for env_var in os.environ:
            if env_var.startswith(self.env_prefix):
                print(f"  {env_var} = {os.environ[env_var]}")

        # Get all environment variables that start with the prefix
        for env_var, value in os.environ.items():
            if env_var.startswith(config_prefix):
                print(f"\nProcessing environment variable: {env_var} = {value}")

                # Remove the prefix to get the key path
                key_path = env_var[len(config_prefix) :].lower()
                print(f"  Key path: {key_path}")

                # Handle special cases for key paths
                if key_path == "output_output_dir":
                    key_parts = ["output", "output_dir"]
                elif key_path == "output_model_save_model":
                    key_parts = ["output", "model", "save_model"]
                elif key_path == "data_required_columns_0_default_value":
                    # This is a special case for array access
                    key_parts = ["data", "required_columns", 0, "default_value"]
                else:
                    # Default case: split by underscores
                    key_parts = key_path.split("_")
                print(f"  Key parts: {key_parts}")

                # Navigate to the correct location in the config
                current = result
                for i, part in enumerate(key_parts[:-1]):
                    print(f"  Navigating to part: {part}")

                    # Handle array indices (numeric parts)
                    if isinstance(part, int) or (
                        isinstance(part, str) and part.isdigit()
                    ):
                        # Convert string to int if it's a digit
                        idx = int(part) if isinstance(part, str) else part

                        # If the current key doesn't exist or isn't a list, create it
                        if key_parts[i - 1] not in current or not isinstance(
                            current[key_parts[i - 1]], list
                        ):
                            print(f"    Creating new list for {key_parts[i-1]}")
                            current[key_parts[i - 1]] = []

                        # Ensure the list has enough elements
                        while len(current[key_parts[i - 1]]) <= idx:
                            current[key_parts[i - 1]].append({})

                        # Move to the list element
                        current = current[key_parts[i - 1]][idx]
                    else:
                        # Create nested dictionaries if they don't exist
                        if part not in current or not isinstance(current[part], dict):
                            print(f"    Creating new dictionary for {part}")
                            current[part] = {}
                        current = current[part]

                # Set the value, converting to the appropriate type
                last_key = key_parts[-1]
                print(f"  Last key: {last_key}")

                # Try to infer the type from the existing value if it exists
                if last_key in current and current[last_key] is not None:
                    existing_type = type(current[last_key])
                    print(
                        f"  Existing value: {current[last_key]} (type: {existing_type.__name__})"
                    )
                    try:
                        # Convert the value to the same type as the existing value
                        if existing_type == bool:
                            # Special handling for boolean values
                            value = value.lower() in ("true", "yes", "1", "y")
                            print(f"  Converted to boolean: {value}")
                        elif existing_type == int:
                            value = int(value)
                            print(f"  Converted to int: {value}")
                        elif existing_type == float:
                            value = float(value)
                            print(f"  Converted to float: {value}")
                        elif existing_type == list:
                            # Split by commas for list values
                            value = [item.strip() for item in value.split(",")]
                            print(f"  Converted to list: {value}")
                        # Otherwise, keep as string
                    except (ValueError, TypeError):
                        # If conversion fails, use the string value
                        print(
                            f"  Warning: Could not convert environment variable {env_var} to type {existing_type.__name__}"
                        )
                else:
                    print(f"  No existing value for {last_key}, using as is: {value}")

                # Set the value in the config
                current[last_key] = value
                print(f"  Applied environment override: {env_var} = {value}")

        print("\nFinal configuration:")
        print(f"  output_dir = {result.get('output', {}).get('output_dir')}")
        if "output" in result and "model" in result["output"]:
            print(f"  save_model = {result['output']['model'].get('save_model')}")

        # Safely check for required_columns
        try:
            if (
                "data" in result
                and "required_columns" in result["data"]
                and isinstance(result["data"]["required_columns"], list)
                and len(result["data"]["required_columns"]) > 0
                and isinstance(result["data"]["required_columns"][0], dict)
                and "default_value" in result["data"]["required_columns"][0]
            ):
                print(
                    f"  required_columns[0]['default_value'] = {result['data']['required_columns'][0]['default_value']}"
                )
        except (KeyError, IndexError, TypeError) as e:
            print(f"  Error accessing required_columns: {e}")

        return result

    def get_config_section(self, name: str, section_class: Type[T]) -> T:
        """
        Get a typed configuration section.

        Args:
            name: Name of the configuration file (without extension)
            section_class: Class to instantiate with the configuration

        Returns:
            Instance of section_class initialized with the configuration
        """
        config = self.load_config(name)
        # Cast to the correct type to satisfy the type checker
        return cast(T, section_class(config))

    def get_data_config(self, name: str = "production_data_config") -> DataConfig:
        """
        Get the data configuration.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            DataConfig instance
        """
        return self.get_config_section(name, DataConfig)

    def get_feature_config(self, name: str = "feature_config") -> FeatureConfig:
        """
        Get the feature configuration.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            FeatureConfig instance
        """
        return self.get_config_section(name, FeatureConfig)

    def get_model_config(self, name: str = "classification_config") -> ModelConfig:
        """
        Get the model configuration.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            ModelConfig instance
        """
        return self.get_config_section(name, ModelConfig)

    def get_pipeline_config(self, name: str = "nexusml_config") -> PipelineConfig:
        """
        Get the pipeline configuration.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            PipelineConfig instance
        """
        # Check if the configuration is already loaded
        if name in self.configs:
            # Use the in-memory configuration
            return PipelineConfig(self.configs[name])
        else:
            # Load the configuration from file
            return self.get_config_section(name, PipelineConfig)

    def get_model_card_config(self, name: str = "model_card_config") -> Any:
        """
        Get the model card configuration.

        Args:
            name: Name of the configuration file (without extension)

        Returns:
            ModelCardConfig instance
        """
        # Import here to avoid circular import
        from nexusml.config.model_card import ModelCardConfig

        return self.get_config_section(name, ModelCardConfig)

    def merge_configs(self, base_name: str, override_name: str) -> Dict[str, Any]:
        """
        Merge two configurations, with the override taking precedence.

        Args:
            base_name: Name of the base configuration file
            override_name: Name of the configuration file with overrides

        Returns:
            Merged configuration dictionary
        """
        base_config = self.load_config(base_name)
        try:
            override_config = self.load_config(override_name)
        except FileNotFoundError:
            # If override doesn't exist, just return the base config
            return base_config

        # Deep merge the dictionaries
        return self._deep_merge(base_config, override_config)

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            base: Base dictionary
            override: Dictionary with overrides

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override or add the value
                result[key] = value

        return result

    def validate_config(self, name: str, schema_name: Optional[str] = None) -> bool:
        """
        Validate a configuration against a JSON Schema.

        Args:
            name: Name of the configuration file
            schema_name: Name of the schema file (defaults to {name}_schema)

        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the configuration
            config = self.load_config(name)

            # Determine schema name if not provided
            if schema_name is None:
                # Map configuration names to schema names
                schema_mapping = {
                    "production_data_config": "data_config_schema",
                    "data_config": "data_config_schema",
                    "feature_config": "feature_config_schema",
                    "classification_config": "model_config_schema",
                    "nexusml_config": "pipeline_config_schema",
                    "model_card_config": "model_card_schema",
                }
                schema_name = schema_mapping.get(name, f"{name}_schema")

            # Use the schema validation functionality from the schemas package
            from nexusml.config.schemas import validate_config as validate_with_schema

            return validate_with_schema(config, schema_name)

        except Exception as e:
            # Validation failed
            print(f"Configuration validation failed: {e}")
            return False
