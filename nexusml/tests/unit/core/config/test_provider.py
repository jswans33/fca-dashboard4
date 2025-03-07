"""
Unit tests for the configuration provider module.

This module contains tests for the ConfigurationProvider class.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from nexusml.core.config.configuration import NexusMLConfig
from nexusml.core.config.provider import ConfigurationProvider


@pytest.fixture
def sample_config_dict() -> Dict[str, Any]:
    """Fixture providing a sample configuration dictionary."""
    return {
        "feature_engineering": {
            "text_combinations": [],
            "numeric_columns": [],
            "hierarchies": [],
            "column_mappings": [],
            "classification_systems": [],
            "direct_mappings": [],
            "eav_integration": {"enabled": False},
        },
        "classification": {
            "classification_targets": [],
            "input_field_mappings": [],
        },
        "data": {
            "required_columns": [],
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
        "equipment_attributes": {},
        "masterformat_primary": {"H": {}},
        "masterformat_equipment": {},
    }


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


@pytest.fixture
def reset_provider():
    """Fixture to reset the ConfigurationProvider singleton between tests."""
    ConfigurationProvider.reset()
    yield
    ConfigurationProvider.reset()


class TestConfigurationProvider:
    """Tests for the ConfigurationProvider class."""

    def test_singleton(self, reset_provider):
        """Test that ConfigurationProvider is a singleton."""
        provider1 = ConfigurationProvider()
        provider2 = ConfigurationProvider()
        assert provider1 is provider2

    def test_set_config(self, reset_provider, sample_config_dict):
        """Test setting the configuration directly."""
        provider = ConfigurationProvider()
        config = NexusMLConfig.model_validate(sample_config_dict)
        provider.set_config(config)

        # Verify the config was set
        assert provider.config is config

    def test_set_config_from_file(self, reset_provider, temp_config_file):
        """Test setting the configuration from a file."""
        provider = ConfigurationProvider()
        provider.set_config_from_file(temp_config_file)

        # Verify the config was loaded
        assert provider.config is not None
        assert isinstance(provider.config, NexusMLConfig)

    def test_reload(self, reset_provider, temp_config_file, monkeypatch):
        """Test reloading the configuration."""
        # Set the environment variable to the temp config file
        monkeypatch.setenv("NEXUSML_CONFIG", str(temp_config_file))

        provider = ConfigurationProvider()
        provider.set_config_from_file(temp_config_file)

        # Get the initial config
        initial_config = provider.config

        # Reload and verify it's a different instance
        provider.reload()
        reloaded_config = provider.config

        assert initial_config is not reloaded_config
        assert isinstance(reloaded_config, NexusMLConfig)

    def test_load_from_env(self, reset_provider, temp_config_file, monkeypatch):
        """Test loading configuration from environment variable."""
        monkeypatch.setenv("NEXUSML_CONFIG", str(temp_config_file))

        provider = ConfigurationProvider()
        config = provider.config  # This should load from env

        assert isinstance(config, NexusMLConfig)

    def test_load_from_default_path(
        self, reset_provider, monkeypatch, tmp_path, sample_config_dict
    ):
        """Test loading from default path when env var is not set."""
        # Create a mock default config file
        default_path = tmp_path / "nexusml_config.yml"
        monkeypatch.setattr(NexusMLConfig, "default_config_path", lambda: default_path)

        # Create the file with complete content
        os.makedirs(default_path.parent, exist_ok=True)
        with open(default_path, "w", encoding="utf-8") as f:
            yaml.dump(sample_config_dict, f)  # Use the fixture value directly

        # Remove env var
        monkeypatch.delenv("NEXUSML_CONFIG", raising=False)

        # This should now work since we're providing a valid config file
        provider = ConfigurationProvider()
        config = provider.config

        # Verify the config was loaded
        assert config is not None
        assert isinstance(config, NexusMLConfig)

    def test_reset(self, reset_provider, sample_config_dict):
        """Test resetting the singleton instance."""
        provider1 = ConfigurationProvider()
        config = NexusMLConfig.model_validate(sample_config_dict)
        provider1.set_config(config)

        # Reset and create a new instance
        ConfigurationProvider.reset()
        provider2 = ConfigurationProvider()

        # Verify it's a new instance with no config
        assert provider1 is not provider2
        assert provider2._config is None
