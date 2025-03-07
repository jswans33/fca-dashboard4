"""
Unit tests for the configuration module.

This module contains tests for the NexusMLConfig class and related functionality.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nexusml.core.config.configuration import (
    ClassificationConfig,
    DataConfig,
    FeatureEngineeringConfig,
    NexusMLConfig,
)


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Fixture providing a sample configuration dictionary."""
    return {
        "feature_engineering": {
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
        "classification": {
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
        "data": {
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
        "reference": {
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
        "equipment_attributes": {
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
        "masterformat_primary": {
            "H": {
                "Chiller Plant": "23 64 00",
                "Cooling Tower Plant": "23 65 00",
            }
        },
        "masterformat_equipment": {
            "Heat Exchanger": "23 57 00",
            "Water Softener": "22 31 00",
        },
    }


@pytest.fixture
def sample_config(sample_config_dict) -> NexusMLConfig:
    """Fixture providing a sample NexusMLConfig instance."""
    return NexusMLConfig.model_validate(sample_config_dict)


@pytest.fixture
def temp_config_file(sample_config_dict) -> Path:
    """Fixture providing a temporary configuration file."""
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        with open(temp_path, "w", encoding="utf-8") as f:
            yaml.dump(sample_config_dict, f)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        os.unlink(temp_path)


class TestNexusMLConfig:
    """Tests for the NexusMLConfig class."""

    def test_parse_obj(self, sample_config_dict):
        """Test parsing a configuration dictionary."""
        config = NexusMLConfig.model_validate(sample_config_dict)
        assert isinstance(config, NexusMLConfig)
        assert isinstance(config.feature_engineering, FeatureEngineeringConfig)
        assert isinstance(config.classification, ClassificationConfig)
        assert isinstance(config.data, DataConfig)

    def test_from_yaml(self, temp_config_file):
        """Test loading configuration from a YAML file."""
        config = NexusMLConfig.from_yaml(temp_config_file)
        assert isinstance(config, NexusMLConfig)
        assert config.feature_engineering.text_combinations[0].name == "combined_text"
        assert (
            config.classification.classification_targets[0].name == "Equipment_Category"
        )

    def test_to_yaml(self, sample_config, tmp_path):
        """Test saving configuration to a YAML file."""
        output_path = tmp_path / "test_config.yml"
        sample_config.to_yaml(output_path)

        # Verify the file was created
        assert output_path.exists()

        # Load the file and verify contents
        with open(output_path, "r", encoding="utf-8") as f:
            loaded_dict = yaml.safe_load(f)

        assert (
            loaded_dict["feature_engineering"]["text_combinations"][0]["name"]
            == "combined_text"
        )
        assert (
            loaded_dict["classification"]["classification_targets"][0]["name"]
            == "Equipment_Category"
        )

    def test_from_env(self, temp_config_file, monkeypatch):
        """Test loading configuration from environment variable."""
        monkeypatch.setenv("NEXUSML_CONFIG", str(temp_config_file))
        config = NexusMLConfig.from_env()
        assert isinstance(config, NexusMLConfig)
        assert config.feature_engineering.text_combinations[0].name == "combined_text"

    def test_from_env_missing(self, monkeypatch):
        """Test error when environment variable is not set."""
        monkeypatch.delenv("NEXUSML_CONFIG", raising=False)
        with pytest.raises(
            ValueError, match="NEXUSML_CONFIG environment variable not set"
        ):
            NexusMLConfig.from_env()

    def test_default_config_path(self):
        """Test getting the default configuration path."""
        path = NexusMLConfig.default_config_path()
        assert path == Path("nexusml/config/nexusml_config.yml")

    def test_file_not_found(self):
        """Test error when configuration file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            NexusMLConfig.from_yaml("nonexistent_file.yml")


class TestFeatureEngineeringConfig:
    """Tests for the FeatureEngineeringConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = FeatureEngineeringConfig()
        assert config.text_combinations == []
        assert config.numeric_columns == []
        assert config.hierarchies == []
        assert config.column_mappings == []
        assert config.classification_systems == []
        assert config.direct_mappings == []
        assert config.eav_integration.enabled is False


class TestClassificationConfig:
    """Tests for the ClassificationConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = ClassificationConfig()
        assert config.classification_targets == []
        assert config.input_field_mappings == []


class TestDataConfig:
    """Tests for the DataConfig class."""

    def test_defaults(self):
        """Test default values."""
        config = DataConfig()
        assert config.required_columns == []
        assert (
            config.training_data.default_path
            == "nexusml/data/training_data/fake_training_data.csv"
        )
        assert config.training_data.encoding == "utf-8"
        assert config.training_data.fallback_encoding == "latin1"
