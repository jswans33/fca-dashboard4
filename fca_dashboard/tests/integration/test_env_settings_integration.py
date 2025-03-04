"""
Integration tests for environment utilities and settings.

This module contains integration tests that verify the interaction between
the settings module and environment utilities, ensuring they work together
correctly for environment variable handling and environment detection.
"""
import pytest

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.env_utils import get_env_var, get_environment, is_dev, is_prod


@pytest.fixture(autouse=True)
def reset_settings_env(monkeypatch):
    """
    Ensure that for each test, the environment variable for ENVIRONMENT is unset
    and settings.config is reset to a known state.
    """
    # Remove any ENVIRONMENT from OS
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    # Reset settings.config for env section (simulate YAML settings)
    settings.config["env"] = {}
    yield
    # Cleanup after each test if needed
    monkeypatch.undo()


def test_get_env_var_from_settings(monkeypatch):
    """
    When an environment variable is not set in the OS, get_env_var should return the value
    from the settings file (under the 'env' section).
    """
    # Simulate YAML settings having a value
    settings.config["env"] = {"TEST_VAR": "from_settings"}
    # Ensure OS variable is not set
    monkeypatch.delenv("TEST_VAR", raising=False)
    assert get_env_var("TEST_VAR") == "from_settings"


def test_get_env_var_from_os(monkeypatch):
    """
    When an environment variable is set in the OS, get_env_var should return that value,
    even if a fallback exists in settings.
    """
    # Simulate YAML settings with one value
    settings.config["env"] = {"TEST_VAR": "from_settings"}
    # Set the OS environment variable
    monkeypatch.setenv("TEST_VAR", "from_env")
    # OS value takes precedence over YAML
    assert get_env_var("TEST_VAR") == "from_env"


def test_environment_detection_from_os(monkeypatch):
    """
    Test that the environment detection functions (is_dev, is_prod, get_environment)
    prioritize OS environment variables.
    """
    # Set OS variable to 'production'
    monkeypatch.setenv("ENVIRONMENT", "production")
    settings.config["env"] = {"ENVIRONMENT": "development"}  # This should be ignored
    assert is_prod() is True
    assert is_dev() is False
    # get_environment returns the OS variable in lowercase
    assert get_environment() == "production"


def test_environment_detection_from_settings(monkeypatch):
    """
    When the OS variable is missing, the environment detection functions should fall back to
    the settings file value.
    """
    # Remove OS ENVIRONMENT
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    # Set the YAML value
    settings.config["env"] = {"ENVIRONMENT": "dev"}
    assert is_dev() is True
    assert is_prod() is False
    assert get_environment() == "dev"


def test_environment_default(monkeypatch):
    """
    When neither the OS environment nor settings provides the ENVIRONMENT key,
    get_environment should return 'unknown'.
    """
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    settings.config["env"] = {}  # Clear any setting
    assert get_environment() == "unknown"
