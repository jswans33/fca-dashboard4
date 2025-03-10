"""
Pipeline Plugin System Module

This module provides the PluginManager class, which is responsible for
discovering and loading plugins for the pipeline system.
"""

import importlib
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type

from nexusml.core.pipeline.registry import ComponentRegistry

# Set up logging
logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Exception raised for errors in the plugin system."""
    pass


class PluginManager:
    """
    Manager for pipeline plugins.

    The PluginManager class is responsible for discovering and loading plugins
    for the pipeline system. It scans specified directories for plugin modules
    and registers their components with the ComponentRegistry.

    Attributes:
        registry: Component registry to register plugins with.
        plugin_dirs: List of directories to scan for plugins.
        loaded_plugins: Set of loaded plugin module names.
    """

    def __init__(self, registry: ComponentRegistry):
        """
        Initialize a new PluginManager.

        Args:
            registry: Component registry to register plugins with.
        """
        self.registry = registry
        self.plugin_dirs: List[Path] = []
        self.loaded_plugins: Set[str] = set()
        logger.info("PluginManager initialized")

    def add_plugin_dir(self, plugin_dir: str) -> None:
        """
        Add a directory to scan for plugins.

        Args:
            plugin_dir: Directory to scan for plugins.

        Raises:
            PluginError: If the directory doesn't exist.
        """
        path = Path(plugin_dir).resolve()
        if not path.exists():
            raise PluginError(f"Plugin directory doesn't exist: {path}")
        if not path.is_dir():
            raise PluginError(f"Plugin path is not a directory: {path}")

        self.plugin_dirs.append(path)
        logger.info(f"Added plugin directory: {path}")

    def discover_plugins(self) -> None:
        """
        Discover and load plugins from all registered directories.

        This method scans all registered plugin directories for Python modules
        and attempts to load them as plugins.

        Raises:
            PluginError: If there's an error loading a plugin.
        """
        for plugin_dir in self.plugin_dirs:
            self._discover_plugins_in_dir(plugin_dir)

    def _discover_plugins_in_dir(self, plugin_dir: Path) -> None:
        """
        Discover and load plugins from a directory.

        Args:
            plugin_dir: Directory to scan for plugins.

        Raises:
            PluginError: If there's an error loading a plugin.
        """
        logger.info(f"Discovering plugins in {plugin_dir}")

        # Add the plugin directory to sys.path temporarily
        sys.path.insert(0, str(plugin_dir.parent))

        try:
            # Get the package name from the directory name
            package_name = plugin_dir.name

            # Import the package
            try:
                package = importlib.import_module(package_name)
            except ImportError as e:
                logger.warning(f"Failed to import package {package_name}: {str(e)}")
                return

            # Scan for modules in the package
            for _, name, is_pkg in pkgutil.iter_modules([str(plugin_dir)]):
                full_name = f"{package_name}.{name}"

                # Skip already loaded plugins
                if full_name in self.loaded_plugins:
                    logger.debug(f"Plugin {full_name} already loaded, skipping")
                    continue

                try:
                    # Import the module
                    module = importlib.import_module(full_name)

                    # Check if it's a plugin
                    if hasattr(module, "register_components"):
                        # Register the plugin's components
                        module.register_components(self.registry)
                        self.loaded_plugins.add(full_name)
                        logger.info(f"Loaded plugin: {full_name}")
                    else:
                        logger.debug(f"Module {full_name} is not a plugin (no register_components function)")

                    # If it's a package, scan it recursively
                    if is_pkg:
                        subdir = plugin_dir / name
                        self._discover_plugins_in_dir(subdir)

                except Exception as e:
                    logger.error(f"Error loading plugin {full_name}: {str(e)}")
                    raise PluginError(f"Error loading plugin {full_name}: {str(e)}") from e

        finally:
            # Remove the plugin directory from sys.path
            sys.path.pop(0)

    def load_plugin(self, plugin_name: str) -> None:
        """
        Load a specific plugin by name.

        Args:
            plugin_name: Name of the plugin to load.

        Raises:
            PluginError: If there's an error loading the plugin.
        """
        # Skip already loaded plugins
        if plugin_name in self.loaded_plugins:
            logger.debug(f"Plugin {plugin_name} already loaded, skipping")
            return

        try:
            # Import the module
            module = importlib.import_module(plugin_name)

            # Check if it's a plugin
            if hasattr(module, "register_components"):
                # Register the plugin's components
                module.register_components(self.registry)
                self.loaded_plugins.add(plugin_name)
                logger.info(f"Loaded plugin: {plugin_name}")
            else:
                raise PluginError(f"Module {plugin_name} is not a plugin (no register_components function)")

        except ImportError as e:
            raise PluginError(f"Failed to import plugin {plugin_name}: {str(e)}") from e
        except Exception as e:
            raise PluginError(f"Error loading plugin {plugin_name}: {str(e)}") from e

    def get_loaded_plugins(self) -> List[str]:
        """
        Get a list of loaded plugin names.

        Returns:
            List of loaded plugin names.
        """
        return list(self.loaded_plugins)