"""
Unit tests for the Settings module.

This module contains tests for the Settings class and related functionality
in the fca_dashboard.config.settings module.
"""

import os
import tempfile
from typing import Generator

import pytest

from fca_dashboard.config.settings import Settings, get_settings


@pytest.fixture
def temp_settings_file() -> Generator[str, None, None]:
    """Create a temporary settings file for testing."""
    config_content = """
database:
  host: localhost
  port: 5432
  user: test_user
  password: secret
app:
  name: test_app
  debug: true
"""
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
        temp_file.write(config_content.encode("utf-8"))
        temp_path = temp_file.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


def test_settings_load_valid_file(temp_settings_file: str) -> None:
    """Test loading settings from a valid file."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("database.host") == "localhost"
    assert settings.get("database.port") == 5432
    assert settings.get("app.name") == "test_app"
    assert settings.get("app.debug") is True


def test_settings_load_missing_file() -> None:
    """Test that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        Settings(config_path="nonexistent_file.yml")


def test_settings_get_nonexistent_key(temp_settings_file: str) -> None:
    """Test getting a non-existent key returns the default value."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("nonexistent.key") is None
    assert settings.get("nonexistent.key", default="fallback") == "fallback"


def test_settings_get_nested_keys(temp_settings_file: str) -> None:
    """Test getting nested keys from the configuration."""
    settings = Settings(config_path=temp_settings_file)
    assert settings.get("database.user") == "test_user"
    assert settings.get("database.password") == "secret"


def test_get_settings_caching(temp_settings_file: str) -> None:
    """Test that get_settings caches instances for the same config path."""
    settings1 = get_settings(temp_settings_file)
    settings2 = get_settings(temp_settings_file)

    # Should be the same instance
    assert settings1 is settings2

    # Modify the first instance and check that the second reflects the change
    settings1.config["test_key"] = "test_value"
    assert settings2.config["test_key"] == "test_value"


def test_get_settings_default() -> None:
    """Test that get_settings returns the default instance when no path is provided."""
    settings = get_settings()
    assert isinstance(settings, Settings)

    # Should return the same default instance on subsequent calls
    settings2 = get_settings()
    assert settings is settings2
