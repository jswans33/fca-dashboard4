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


@pytest.fixture
def temp_settings_file_with_env_vars() -> Generator[str, None, None]:
    """Create a temporary settings file with environment variable placeholders."""
    config_content = """
database:
  host: localhost
  port: 5432
  user: ${TEST_DB_USER}
  password: ${TEST_DB_PASSWORD}
app:
  name: test_app
  debug: true
  environments: ["dev", "${TEST_ENV}", "prod"]
  secrets: 
    - key1: value1
    - key2: ${TEST_SECRET}
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


def test_environment_variable_substitution(temp_settings_file_with_env_vars: str) -> None:
    """Test that environment variables are substituted in the configuration."""
    # Set environment variables for testing
    os.environ["TEST_DB_USER"] = "env_user"
    os.environ["TEST_DB_PASSWORD"] = "env_password"
    
    try:
        # Load settings with environment variables
        settings = Settings(config_path=temp_settings_file_with_env_vars)
        
        # Check that environment variables were substituted
        assert settings.get("database.user") == "env_user"
        assert settings.get("database.password") == "env_password"
        
        # Check that non-environment variable settings are still loaded correctly
        assert settings.get("database.host") == "localhost"
        assert settings.get("app.name") == "test_app"
    finally:
        # Clean up environment variables
        del os.environ["TEST_DB_USER"]
        del os.environ["TEST_DB_PASSWORD"]


def test_missing_environment_variable(temp_settings_file_with_env_vars: str) -> None:
    """Test that missing environment variables keep the original placeholder."""
    # Ensure environment variables are not set
    if "TEST_DB_USER" in os.environ:
        del os.environ["TEST_DB_USER"]
    if "TEST_DB_PASSWORD" in os.environ:
        del os.environ["TEST_DB_PASSWORD"]
    
    # Load settings with missing environment variables
    settings = Settings(config_path=temp_settings_file_with_env_vars)
    
    # Check that placeholders are preserved
    assert settings.get("database.user") == "${TEST_DB_USER}"
    assert settings.get("database.password") == "${TEST_DB_PASSWORD}"


def test_environment_variable_substitution_in_lists(temp_settings_file_with_env_vars: str) -> None:
    """Test that environment variables are substituted in lists and nested structures."""
    # Set environment variables for testing
    os.environ["TEST_ENV"] = "staging"
    os.environ["TEST_SECRET"] = "secret_value"
    
    try:
        # Load settings with environment variables
        settings = Settings(config_path=temp_settings_file_with_env_vars)
        
        # Check that environment variables in lists are substituted
        environments = settings.get("app.environments")
        assert isinstance(environments, list)
        assert environments[0] == "dev"
        assert environments[1] == "staging"  # Substituted from ${TEST_ENV}
        assert environments[2] == "prod"
        
        # Check that environment variables in nested structures are substituted
        secrets = settings.get("app.secrets")
        assert isinstance(secrets, list)
        assert secrets[0]["key1"] == "value1"
        assert secrets[1]["key2"] == "secret_value"  # Substituted from ${TEST_SECRET}
    finally:
        # Clean up environment variables
        del os.environ["TEST_ENV"]
        del os.environ["TEST_SECRET"]
