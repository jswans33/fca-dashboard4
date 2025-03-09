"""
Tests for Pipeline Plugin System Module

This module contains tests for the PluginManager class, ensuring that
plugins can be properly discovered and loaded.
"""

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from nexusml.core.pipeline.plugins import PluginManager, PluginError
from nexusml.core.pipeline.registry import ComponentRegistry


class TestPluginManager(unittest.TestCase):
    """
    Test case for PluginManager class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        self.registry = MagicMock(spec=ComponentRegistry)
        self.plugin_manager = PluginManager(self.registry)

    def test_add_plugin_dir(self):
        """
        Test adding a plugin directory.
        """
        # Create a temporary directory for testing
        temp_dir = Path("temp_plugin_dir")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Add the plugin directory
            self.plugin_manager.add_plugin_dir(str(temp_dir))

            # Verify that the directory was added
            self.assertEqual(len(self.plugin_manager.plugin_dirs), 1)
            self.assertEqual(self.plugin_manager.plugin_dirs[0], temp_dir.resolve())
        finally:
            # Clean up
            temp_dir.rmdir()

    def test_add_plugin_dir_nonexistent(self):
        """
        Test that adding a nonexistent plugin directory raises an error.
        """
        # Try to add a nonexistent plugin directory
        with self.assertRaises(PluginError):
            self.plugin_manager.add_plugin_dir("nonexistent_dir")

    def test_add_plugin_dir_not_dir(self):
        """
        Test that adding a file as a plugin directory raises an error.
        """
        # Create a temporary file for testing
        temp_file = Path("temp_plugin_file")
        temp_file.touch()

        try:
            # Try to add the file as a plugin directory
            with self.assertRaises(PluginError):
                self.plugin_manager.add_plugin_dir(str(temp_file))
        finally:
            # Clean up
            temp_file.unlink()

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    @patch('nexusml.core.pipeline.plugins.pkgutil.iter_modules')
    def test_discover_plugins(self, mock_iter_modules, mock_import_module):
        """
        Test discovering plugins.
        """
        # Create a temporary directory for testing
        temp_dir = Path("temp_plugin_dir")
        temp_dir.mkdir(exist_ok=True)

        try:
            # Add the plugin directory
            self.plugin_manager.add_plugin_dir(str(temp_dir))

            # Set up the mock module
            mock_module = MagicMock()
            mock_module.__name__ = "temp_plugin_dir"
            mock_import_module.return_value = mock_module

            # Set up the mock iter_modules
            mock_iter_modules.return_value = [
                (None, "plugin1", False),
                (None, "plugin2", True),
            ]

            # Set up the mock plugin modules
            mock_plugin1 = MagicMock()
            mock_plugin1.__name__ = "temp_plugin_dir.plugin1"
            mock_plugin1.register_components = MagicMock()

            mock_plugin2 = MagicMock()
            mock_plugin2.__name__ = "temp_plugin_dir.plugin2"
            mock_plugin2.register_components = MagicMock()

            # Set up the import_module side effect
            def import_module_side_effect(name):
                if name == "temp_plugin_dir":
                    return mock_module
                elif name == "temp_plugin_dir.plugin1":
                    return mock_plugin1
                elif name == "temp_plugin_dir.plugin2":
                    return mock_plugin2
                else:
                    raise ImportError(f"No module named '{name}'")

            mock_import_module.side_effect = import_module_side_effect

            # Discover plugins
            self.plugin_manager.discover_plugins()

            # Verify that the plugins were discovered and registered
            mock_import_module.assert_any_call("temp_plugin_dir")
            mock_import_module.assert_any_call("temp_plugin_dir.plugin1")
            mock_import_module.assert_any_call("temp_plugin_dir.plugin2")
            mock_plugin1.register_components.assert_called_once_with(self.registry)
            mock_plugin2.register_components.assert_called_once_with(self.registry)

            # Verify that the plugins were added to the loaded plugins set
            self.assertEqual(len(self.plugin_manager.loaded_plugins), 2)
            self.assertIn("temp_plugin_dir.plugin1", self.plugin_manager.loaded_plugins)
            self.assertIn("temp_plugin_dir.plugin2", self.plugin_manager.loaded_plugins)
        finally:
            # Clean up
            temp_dir.rmdir()

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    def test_load_plugin(self, mock_import_module):
        """
        Test loading a specific plugin.
        """
        # Set up the mock plugin module
        mock_plugin = MagicMock()
        mock_plugin.__name__ = "test_plugin"
        mock_plugin.register_components = MagicMock()
        mock_import_module.return_value = mock_plugin

        # Load the plugin
        self.plugin_manager.load_plugin("test_plugin")

        # Verify that the plugin was loaded and registered
        mock_import_module.assert_called_once_with("test_plugin")
        mock_plugin.register_components.assert_called_once_with(self.registry)

        # Verify that the plugin was added to the loaded plugins set
        self.assertEqual(len(self.plugin_manager.loaded_plugins), 1)
        self.assertIn("test_plugin", self.plugin_manager.loaded_plugins)

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    def test_load_plugin_already_loaded(self, mock_import_module):
        """
        Test loading a plugin that is already loaded.
        """
        # Add the plugin to the loaded plugins set
        self.plugin_manager.loaded_plugins.add("test_plugin")

        # Load the plugin
        self.plugin_manager.load_plugin("test_plugin")

        # Verify that the plugin was not loaded again
        mock_import_module.assert_not_called()

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    def test_load_plugin_import_error(self, mock_import_module):
        """
        Test that loading a plugin that cannot be imported raises an error.
        """
        # Set up the mock import_module to raise an ImportError
        mock_import_module.side_effect = ImportError("Test error")

        # Try to load the plugin
        with self.assertRaises(PluginError):
            self.plugin_manager.load_plugin("test_plugin")

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    def test_load_plugin_no_register_components(self, mock_import_module):
        """
        Test that loading a plugin without a register_components function raises an error.
        """
        # Set up the mock plugin module without a register_components function
        mock_plugin = MagicMock()
        mock_plugin.__name__ = "test_plugin"
        mock_import_module.return_value = mock_plugin

        # Try to load the plugin
        with self.assertRaises(PluginError):
            self.plugin_manager.load_plugin("test_plugin")

    @patch('nexusml.core.pipeline.plugins.importlib.import_module')
    def test_load_plugin_register_error(self, mock_import_module):
        """
        Test that errors during plugin registration are properly handled.
        """
        # Set up the mock plugin module with a register_components function that raises an error
        mock_plugin = MagicMock()
        mock_plugin.__name__ = "test_plugin"
        mock_plugin.register_components = MagicMock(side_effect=ValueError("Test error"))
        mock_import_module.return_value = mock_plugin

        # Try to load the plugin
        with self.assertRaises(PluginError):
            self.plugin_manager.load_plugin("test_plugin")

    def test_get_loaded_plugins(self):
        """
        Test getting a list of loaded plugin names.
        """
        # Add some plugins to the loaded plugins set
        self.plugin_manager.loaded_plugins.add("plugin1")
        self.plugin_manager.loaded_plugins.add("plugin2")
        self.plugin_manager.loaded_plugins.add("plugin3")

        # Get the loaded plugins
        loaded_plugins = self.plugin_manager.get_loaded_plugins()

        # Verify that the loaded plugins list is correct
        self.assertEqual(len(loaded_plugins), 3)
        self.assertIn("plugin1", loaded_plugins)
        self.assertIn("plugin2", loaded_plugins)
        self.assertIn("plugin3", loaded_plugins)


if __name__ == '__main__':
    unittest.main()