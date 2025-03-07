"""
Unit tests for the configuration migration module.

This module contains tests for the migration functionality.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from nexusml.core.config.configuration import NexusMLConfig
from nexusml.core.config.migration import (
    load_json_config,
    load_yaml_config,
    migrate_configs,
)


@pytest.fixture
def sample_yaml_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Fixture providing sample YAML configurations."""
    return [
        (
            "feature_config.yml",
            {
                "text_combinations": [
                    {
                        "name": "combined_text",
                        "columns": ["equipment_tag", "manufacturer", "model"],
                        "separator": " ",
                    }
                ],
                "numeric_columns": [
                    {
                        "name": "initial_cost",
                        "new_name": "initial_cost",
                        "fill_value": 0,
                        "dtype": "float",
                    }
                ],
                "hierarchies": [],
                "column_mappings": [],
                "classification_systems": [],
                "direct_mappings": [],
                "eav_integration": {"enabled": False},
            },
        ),
        (
            "classification_config.yml",
            {
                "classification_targets": [
                    {
                        "name": "Equipment_Category",
                        "description": "Primary equipment type",
                        "required": True,
                        "master_db": {
                            "table": "Equipment_Categories",
                            "field": "CategoryName",
                            "id_field": "CategoryID",
                        },
                    }
                ],
                "input_field_mappings": [],
            },
        ),
        (
            "data_config.yml",
            {
                "required_columns": [
                    {
                        "name": "equipment_tag",
                        "default_value": "",
                        "data_type": "str",
                    }
                ],
                "training_data": {
                    "default_path": "nexusml/data/training_data/fake_training_data.csv",
                    "encoding": "utf-8",
                    "fallback_encoding": "latin1",
                },
            },
        ),
        (
            "reference_config.yml",
            {
                "paths": {
                    "omniclass": "nexusml/ingest/reference/omniclass",
                    "uniformat": "nexusml/ingest/reference/uniformat",
                    "masterformat": "nexusml/ingest/reference/masterformat",
                    "mcaa_glossary": "nexusml/ingest/reference/mcaa-glossary",
                    "mcaa_abbreviations": "nexusml/ingest/reference/mcaa-glossary",
                    "smacna": "nexusml/ingest/reference/smacna-manufacturers",
                    "ashrae": "nexusml/ingest/reference/service-life/ashrae",
                    "energize_denver": "nexusml/ingest/reference/service-life/energize-denver",
                    "equipment_taxonomy": "nexusml/ingest/reference/equipment-taxonomy",
                },
                "file_patterns": {
                    "omniclass": "*.csv",
                    "uniformat": "*.csv",
                    "masterformat": "*.csv",
                    "mcaa_glossary": "Glossary.csv",
                    "mcaa_abbreviations": "Abbreviations.csv",
                    "smacna": "*.json",
                    "ashrae": "*.csv",
                    "energize_denver": "*.csv",
                    "equipment_taxonomy": "*.csv",
                },
                "column_mappings": {
                    "omniclass": {
                        "code": "OmniClass_Code",
                        "name": "OmniClass_Title",
                        "description": "Description",
                    },
                    "uniformat": {
                        "code": "UniFormat Code",
                        "name": "UniFormat Title",
                        "description": "Description",
                    },
                    "masterformat": {
                        "code": "MasterFormat Code",
                        "name": "MasterFormat Title",
                        "description": "Description",
                    },
                    "service_life": {
                        "equipment_type": "Equipment Type",
                        "median_years": "Median Years",
                        "min_years": "Min Years",
                        "max_years": "Max Years",
                        "source": "Source",
                    },
                    "equipment_taxonomy": {
                        "asset_category": "Asset Category",
                        "equipment_id": "Equip Name ID",
                        "trade": "Trade",
                        "title": "Title",
                        "drawing_abbreviation": "Drawing Abbreviation",
                        "precon_tag": "Precon Tag",
                        "system_type_id": "System Type ID",
                        "sub_system_type": "Sub System Type",
                        "sub_system_id": "Sub System ID",
                        "sub_system_class": "Sub System Class",
                        "class_id": "Class ID",
                        "equipment_size": "Equipment Size",
                        "unit": "Unit",
                        "service_maintenance_hrs": "Service Maintenance Hrs",
                        "service_life": "Service Life",
                    },
                },
                "hierarchies": {
                    "omniclass": {"separator": "-", "levels": 3},
                    "uniformat": {"separator": "", "levels": 4},
                    "masterformat": {"separator": " ", "levels": 3},
                },
                "defaults": {"service_life": 15.0, "confidence": 0.5},
            },
        ),
    ]


@pytest.fixture
def sample_json_configs() -> List[Tuple[str, Dict[str, Any]]]:
    """Fixture providing sample JSON configurations."""
    return [
        (
            "equipment_attributes.json",
            {
                "Chiller": {
                    "omniclass_id": "23-33 11 11 11",
                    "masterformat_id": "23 64 00",
                    "uniformat_id": "D3020",
                    "required_attributes": [
                        "cooling_capacity_tons",
                        "efficiency_kw_per_ton",
                        "refrigerant_type",
                        "chiller_type",
                    ],
                    "optional_attributes": [],
                    "units": {},
                    "performance_fields": {},
                }
            },
        ),
        (
            "masterformat_primary.json",
            {
                "H": {
                    "Chiller Plant": "23 64 00",
                    "Cooling Tower Plant": "23 65 00",
                }
            },
        ),
        (
            "masterformat_equipment.json",
            {
                "Heat Exchanger": "23 57 00",
                "Water Softener": "22 31 00",
            },
        ),
    ]


@pytest.fixture
def temp_config_files(
    sample_yaml_configs, sample_json_configs, tmp_path
) -> Dict[str, Path]:
    """Fixture providing temporary configuration files."""
    config_files = {}

    # Create YAML files
    for filename, content in sample_yaml_configs:
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
        config_files[filename] = file_path

    # Create JSON files
    for filename, content in sample_json_configs:
        file_path = tmp_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f)
        config_files[filename] = file_path

    return config_files


class TestLoadConfig:
    """Tests for the configuration loading functions."""

    def test_load_yaml_config(self, temp_config_files):
        """Test loading a YAML configuration file."""
        config = load_yaml_config(temp_config_files["feature_config.yml"])
        assert isinstance(config, dict)
        assert "text_combinations" in config
        assert config["text_combinations"][0]["name"] == "combined_text"

    def test_load_json_config(self, temp_config_files):
        """Test loading a JSON configuration file."""
        config = load_json_config(temp_config_files["equipment_attributes.json"])
        assert isinstance(config, dict)
        assert "Chiller" in config
        assert config["Chiller"]["omniclass_id"] == "23-33 11 11 11"

    def test_load_yaml_config_not_found(self):
        """Test error when YAML file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("nonexistent_file.yml")

    def test_load_json_config_not_found(self):
        """Test error when JSON file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_json_config("nonexistent_file.json")


class TestMigrateConfigs:
    """Tests for the migrate_configs function."""

    def test_migrate_configs(self, temp_config_files, tmp_path):
        """Test migrating configurations to a unified format."""
        output_path = tmp_path / "nexusml_config.yml"

        config = migrate_configs(
            output_path=output_path,
            feature_config_path=temp_config_files["feature_config.yml"],
            classification_config_path=temp_config_files["classification_config.yml"],
            data_config_path=temp_config_files["data_config.yml"],
            reference_config_path=temp_config_files["reference_config.yml"],
            equipment_attributes_path=temp_config_files["equipment_attributes.json"],
            masterformat_primary_path=temp_config_files["masterformat_primary.json"],
            masterformat_equipment_path=temp_config_files[
                "masterformat_equipment.json"
            ],
        )

        # Verify the config was created
        assert isinstance(config, NexusMLConfig)

        # Verify the output file was created
        assert output_path.exists()

        # Load the file and verify contents
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_dict = yaml.safe_load(f)

        # Check that all sections were migrated
        assert "feature_engineering" in loaded_dict
        assert "classification" in loaded_dict
        assert "data" in loaded_dict
        assert "reference" in loaded_dict
        assert "equipment_attributes" in loaded_dict
        assert "masterformat_primary" in loaded_dict
        assert "masterformat_equipment" in loaded_dict

        # Check specific values
        assert (
            loaded_dict["feature_engineering"]["text_combinations"][0]["name"]
            == "combined_text"
        )
        assert (
            loaded_dict["classification"]["classification_targets"][0]["name"]
            == "Equipment_Category"
        )
        assert loaded_dict["data"]["required_columns"][0]["name"] == "equipment_tag"
        assert (
            loaded_dict["equipment_attributes"]["Chiller"]["omniclass_id"]
            == "23-33 11 11 11"
        )
        assert loaded_dict["masterformat_primary"]["H"]["Chiller Plant"] == "23 64 00"
        assert loaded_dict["masterformat_equipment"]["Heat Exchanger"] == "23 57 00"

    def test_migrate_configs_partial(self, temp_config_files, tmp_path):
        """Test migrating with only some configuration files."""
        output_path = tmp_path / "nexusml_config.yml"

        config = migrate_configs(
            output_path=output_path,
            feature_config_path=temp_config_files["feature_config.yml"],
            classification_config_path=temp_config_files["classification_config.yml"],
        )

        # Verify the config was created
        assert isinstance(config, NexusMLConfig)

        # Verify the output file was created
        assert output_path.exists()

        # Load the file and verify contents
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_dict = yaml.safe_load(f)

        # Check that specified sections were migrated
        assert "feature_engineering" in loaded_dict
        assert "classification" in loaded_dict

        # Check that other sections use defaults or are empty
        assert "data" in loaded_dict  # Data is included with default values
        assert "equipment_attributes" in loaded_dict
        assert loaded_dict["equipment_attributes"] == {}  # Empty dict

        # Check specific values
        assert (
            loaded_dict["feature_engineering"]["text_combinations"][0]["name"]
            == "combined_text"
        )
        assert (
            loaded_dict["classification"]["classification_targets"][0]["name"]
            == "Equipment_Category"
        )
