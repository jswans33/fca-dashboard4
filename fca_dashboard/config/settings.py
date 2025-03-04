"""
Configuration module for loading and accessing application settings.

This module provides functionality to load settings from YAML configuration files
and access them in a structured way throughout the application.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class Settings:
    """
    Settings class for loading and accessing application configuration.

    This class provides methods to load settings from YAML files and access
    them through a simple interface.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Settings with optional config path.

        Args:
            config_path: Path to the YAML configuration file. If None, uses default.
        """
        self.config: Dict[str, Any] = {}
        self.config_path = Path(config_path) if config_path else Path(__file__).parent / "settings.yml"
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        config_path = self.config_path

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
            
        # Process environment variable substitutions
        self._process_env_vars(self.config)
        
    def _process_env_vars(self, config_section: Any) -> None:
        """
        Recursively process environment variable substitutions in the config.
        
        Args:
            config_section: A section of the configuration to process
        """
        if isinstance(config_section, dict):
            for key, value in config_section.items():
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[key] = self._substitute_env_vars(value)
        elif isinstance(config_section, list):
            for i, value in enumerate(config_section):
                if isinstance(value, (dict, list)):
                    self._process_env_vars(value)
                elif isinstance(value, str):
                    config_section[i] = self._substitute_env_vars(value)
    
    def _substitute_env_vars(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: The string value to process
            
        Returns:
            The string with environment variables substituted
        """
        # Match ${VAR_NAME} pattern
        pattern = r'\${([A-Za-z0-9_]+)}'
        
        def replace_env_var(match):
            env_var_name = match.group(1)
            env_var_value = os.environ.get(env_var_name)
            if env_var_value is None:
                # If environment variable is not set, keep the original placeholder
                return match.group(0)
            return env_var_value
            
        return re.sub(pattern, replace_env_var, value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key.

        Args:
            key: The configuration key to retrieve
            default: Default value if key is not found

        Returns:
            The configuration value or default if not found
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


# Cache for settings instances
# Consider thread-safety if accessed from multiple threads
_settings_cache: Dict[str, Settings] = {}
# Create a default settings instance
settings = Settings()


def get_settings(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """
    Get a Settings instance, with caching for repeated calls.

    Args:
        config_path: Optional path to a configuration file

    Returns:
        A Settings instance
    """
    if config_path is None:
        return settings

    # Convert to string for dictionary key
    cache_key = str(config_path)

    # Return cached instance if available
    if cache_key in _settings_cache:
        return _settings_cache[cache_key]

    # Create new instance and cache it
    new_settings = Settings(config_path)
    _settings_cache[cache_key] = new_settings
    return new_settings
