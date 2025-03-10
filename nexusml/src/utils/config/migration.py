"""
Migration script for NexusML configuration.

This module provides functionality to migrate from the legacy configuration files
to the new unified configuration format.

Note: The legacy configuration files are maintained for backward compatibility
and are planned for removal in future work chunks. Once all code is updated to
use the new unified configuration system, these files will be removed.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from nexusml.core.config.configuration import NexusMLConfig


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        file_path: Path to the YAML configuration file

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the file is not valid YAML
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        file_path: Path to the JSON configuration file

    Returns:
        Dict[str, Any]: The loaded configuration as a dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def migrate_configs(
    output_path: Union[str, Path],
    feature_config_path: Optional[Union[str, Path]] = None,
    classification_config_path: Optional[Union[str, Path]] = None,
    data_config_path: Optional[Union[str, Path]] = None,
    reference_config_path: Optional[Union[str, Path]] = None,
    equipment_attributes_path: Optional[Union[str, Path]] = None,
    masterformat_primary_path: Optional[Union[str, Path]] = None,
    masterformat_equipment_path: Optional[Union[str, Path]] = None,
) -> NexusMLConfig:
    """
    Migrate from legacy configuration files to the new unified format.

    Args:
        output_path: Path to save the unified configuration file
        feature_config_path: Path to the feature engineering configuration file
        classification_config_path: Path to the classification configuration file
        data_config_path: Path to the data preprocessing configuration file
        reference_config_path: Path to the reference data configuration file
        equipment_attributes_path: Path to the equipment attributes configuration file
        masterformat_primary_path: Path to the primary MasterFormat mappings file
        masterformat_equipment_path: Path to the equipment-specific MasterFormat mappings file

    Returns:
        NexusMLConfig: The migrated configuration

    Raises:
        FileNotFoundError: If any of the specified files don't exist
        ValueError: If the configuration is invalid
    """
    # Initialize with default values
    config_dict: Dict[str, Any] = {}

    # Load feature engineering configuration
    if feature_config_path:
        feature_config = load_yaml_config(feature_config_path)
        config_dict["feature_engineering"] = feature_config

    # Load classification configuration
    if classification_config_path:
        classification_config = load_yaml_config(classification_config_path)
        config_dict["classification"] = classification_config

    # Load data preprocessing configuration
    if data_config_path:
        data_config = load_yaml_config(data_config_path)
        config_dict["data"] = data_config

    # Load reference data configuration
    if reference_config_path:
        reference_config = load_yaml_config(reference_config_path)
        config_dict["reference"] = reference_config

    # Load equipment attributes configuration
    if equipment_attributes_path:
        equipment_attributes = load_json_config(equipment_attributes_path)
        config_dict["equipment_attributes"] = equipment_attributes

    # Load MasterFormat primary mappings
    if masterformat_primary_path:
        masterformat_primary = load_json_config(masterformat_primary_path)
        config_dict["masterformat_primary"] = masterformat_primary

    # Load MasterFormat equipment mappings
    if masterformat_equipment_path:
        masterformat_equipment = load_json_config(masterformat_equipment_path)
        config_dict["masterformat_equipment"] = masterformat_equipment

    # Create and validate the configuration
    config = NexusMLConfig.model_validate(config_dict)

    # Save the configuration
    output_path = Path(output_path)
    config.to_yaml(output_path)

    return config


def migrate_from_default_paths(
    output_path: Optional[Union[str, Path]] = None,
) -> NexusMLConfig:
    """
    Migrate from default configuration file paths to the new unified format.

    Args:
        output_path: Path to save the unified configuration file.
                    If None, uses the default path.

    Returns:
        NexusMLConfig: The migrated configuration

    Raises:
        FileNotFoundError: If any of the required files don't exist
        ValueError: If the configuration is invalid
    """
    base_path = Path("nexusml/config")

    if output_path is None:
        output_path = base_path / "nexusml_config.yml"

    return migrate_configs(
        output_path=output_path,
        feature_config_path=base_path / "feature_config.yml",
        classification_config_path=base_path / "classification_config.yml",
        data_config_path=base_path / "data_config.yml",
        reference_config_path=base_path / "reference_config.yml",
        equipment_attributes_path=base_path / "eav/equipment_attributes.json",
        masterformat_primary_path=base_path / "mappings/masterformat_primary.json",
        masterformat_equipment_path=base_path / "mappings/masterformat_equipment.json",
    )


if __name__ == "__main__":
    """
    Command-line entry point for migration script.

    Usage:
        python -m nexusml.core.config.migration [output_path]

    Args:
        output_path: Optional path to save the unified configuration file.
                    If not provided, uses the default path.
    """
    import sys

    output_file = None
    if len(sys.argv) > 1:
        output_file = sys.argv[1]

    try:
        config = migrate_from_default_paths(output_file)
        print(
            f"Successfully migrated configuration to: {output_file or NexusMLConfig.default_config_path()}"
        )
    except Exception as e:
        print(f"Error migrating configuration: {e}")
        sys.exit(1)
