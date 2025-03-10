"""
Configuration provider for NexusML.

This module provides a singleton configuration provider for the NexusML suite,
ensuring consistent access to configuration settings throughout the application.

Note: The legacy configuration files are maintained for backward compatibility
and are planned for removal in future work chunks. Once all code is updated to
use the new unified configuration system, these files will be removed.
"""

import os
from pathlib import Path
from typing import Optional, Union

from nexusml.core.config.configuration import NexusMLConfig


class ConfigurationProvider:
    """
    Singleton provider for NexusML configuration.

    This class implements the singleton pattern to ensure that only one instance
    of the configuration is loaded and used throughout the application.
    """

    _instance: Optional["ConfigurationProvider"] = None
    _config: Optional[NexusMLConfig] = None

    def __new__(cls) -> "ConfigurationProvider":
        """
        Create a new instance of ConfigurationProvider if one doesn't exist.

        Returns:
            ConfigurationProvider: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(ConfigurationProvider, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance

    @property
    def config(self) -> NexusMLConfig:
        """
        Get the configuration instance, loading it if necessary.

        Returns:
            NexusMLConfig: The configuration instance

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> NexusMLConfig:
        """
        Load the configuration from the environment or default path.

        Returns:
            NexusMLConfig: The loaded configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        # Try to load from environment variable
        try:
            return NexusMLConfig.from_env()
        except ValueError:
            # If environment variable is not set, try default path
            default_path = NexusMLConfig.default_config_path()
            if default_path.exists():
                return NexusMLConfig.from_yaml(default_path)
            else:
                raise FileNotFoundError(
                    f"Configuration file not found at default path: {default_path}. "
                    "Please set the NEXUSML_CONFIG environment variable or "
                    "create a configuration file at the default path."
                )

    def reload(self) -> None:
        """
        Reload the configuration from the source.

        This method forces a reload of the configuration, which can be useful
        when the configuration file has been modified.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        self._config = None
        _ = self.config  # Force reload

    def set_config(self, config: NexusMLConfig) -> None:
        """
        Set the configuration instance directly.

        This method is primarily useful for testing or when the configuration
        needs to be created programmatically.

        Args:
            config: The configuration instance to use
        """
        self._config = config

    def set_config_from_file(self, file_path: Union[str, Path]) -> None:
        """
        Set the configuration from a specific file path.

        Args:
            file_path: Path to the configuration file

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        self._config = NexusMLConfig.from_yaml(file_path)

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        This method is primarily useful for testing.
        """
        cls._instance = None
        cls._config = None
