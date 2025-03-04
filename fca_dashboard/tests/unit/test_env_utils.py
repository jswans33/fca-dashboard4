"""
Unit tests for environment utilities.

This module contains tests for the environment utility functions
in the fca_dashboard.utils.env_utils module.
"""

import os
from unittest.mock import MagicMock, patch

from fca_dashboard.utils.env_utils import get_env_var, get_environment, is_dev, is_prod


class TestGetEnvVar:
    """Test cases for the get_env_var function."""

    def test_existing_env_var(self):
        """Test retrieving an existing environment variable."""
        # Set up test environment variable
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            assert get_env_var("TEST_VAR") == "test_value"
            assert get_env_var("TEST_VAR", "fallback") == "test_value"

    def test_missing_env_var_with_fallback(self):
        """Test retrieving a missing environment variable with a fallback value."""
        # Ensure the environment variable doesn't exist
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        assert get_env_var("NONEXISTENT_VAR", "fallback") == "fallback"

    def test_missing_env_var_without_fallback(self):
        """Test retrieving a missing environment variable without a fallback value."""
        # Ensure the environment variable doesn't exist
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]

        assert get_env_var("NONEXISTENT_VAR") is None

    def test_empty_env_var(self):
        """Test retrieving an empty environment variable."""
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            assert get_env_var("EMPTY_VAR") == ""
            assert get_env_var("EMPTY_VAR", "fallback") == ""

    def test_type_conversion(self):
        """Test type conversion of environment variables."""
        with patch.dict(os.environ, {"INT_VAR": "123", "BOOL_VAR": "true"}):
            # String by default
            assert get_env_var("INT_VAR") == "123"
            assert get_env_var("BOOL_VAR") == "true"
            
    def test_settings_fallback(self):
        """Test fallback to settings when environment variable is not set."""
        # Ensure the environment variable doesn't exist
        if "SETTINGS_VAR" in os.environ:
            del os.environ["SETTINGS_VAR"]
            
        # Mock the settings object
        mock_settings = MagicMock()
        mock_settings.get.side_effect = lambda key: {
            "env.SETTINGS_VAR": "settings_value",
            "TOP_LEVEL_VAR": "top_level_value"
        }.get(key)
        
        with patch("fca_dashboard.utils.env_utils.settings", mock_settings):
            # Test fallback to env section in settings
            assert get_env_var("SETTINGS_VAR") == "settings_value"
            
            # Test fallback to top level in settings
            assert get_env_var("TOP_LEVEL_VAR") == "top_level_value"
            
            # Test fallback to provided default when not in settings
            assert get_env_var("NONEXISTENT_VAR", "default") == "default"


class TestEnvironmentChecks:
    """Test cases for environment check functions."""

    def test_is_dev_true(self):
        """Test is_dev returns True when environment is development."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_dev() is True

        with patch.dict(os.environ, {"ENVIRONMENT": "dev"}):
            assert is_dev() is True

    def test_is_dev_false(self):
        """Test is_dev returns False when environment is not development."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_dev() is False

        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            assert is_dev() is False

        # When environment variable is not set, we need to mock settings to avoid default
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
            
        # Mock the settings to return a non-development environment
        mock_settings = MagicMock()
        mock_settings.get.return_value = "production"
        
        with patch("fca_dashboard.utils.env_utils.settings", mock_settings):
            assert is_dev() is False

    def test_is_prod_true(self):
        """Test is_prod returns True when environment is production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            assert is_prod() is True

        with patch.dict(os.environ, {"ENVIRONMENT": "prod"}):
            assert is_prod() is True

    def test_is_prod_false(self):
        """Test is_prod returns False when environment is not production."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert is_prod() is False

        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            assert is_prod() is False

        # When environment variable is not set
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
        assert is_prod() is False

    def test_custom_env_var_name(self):
        """Test environment checks with custom environment variable name."""
        # Test with APP_ENV instead of ENVIRONMENT
        with patch.dict(os.environ, {"APP_ENV": "production", "ENVIRONMENT": "development"}), \
             patch("fca_dashboard.utils.env_utils.ENV_VAR_NAME", "APP_ENV"):
                assert is_prod() is True
                assert is_dev() is False


class TestGetEnvironment:
    """Test cases for the get_environment function."""
    
    def test_get_environment_known(self):
        """Test get_environment returns the correct environment name."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            assert get_environment() == "development"
            
        with patch.dict(os.environ, {"ENVIRONMENT": "PRODUCTION"}):
            assert get_environment() == "production"
            
        with patch.dict(os.environ, {"ENVIRONMENT": "Staging"}):
            assert get_environment() == "staging"
    
    def test_get_environment_unknown(self):
        """Test get_environment returns 'unknown' when environment is not set."""
        # Ensure the environment variable doesn't exist
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
            
        # Mock the settings to return None for any key
        mock_settings = MagicMock()
        mock_settings.get.return_value = None
        
        with patch("fca_dashboard.utils.env_utils.settings", mock_settings):
            assert get_environment() == "unknown"
    
    def test_get_environment_from_settings(self):
        """Test get_environment retrieves from settings when not in environment."""
        # Ensure the environment variable doesn't exist
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
            
        # Mock the settings object
        mock_settings = MagicMock()
        mock_settings.get.return_value = "test_environment"
        
        with patch("fca_dashboard.utils.env_utils.settings", mock_settings):
            assert get_environment() == "test_environment"
