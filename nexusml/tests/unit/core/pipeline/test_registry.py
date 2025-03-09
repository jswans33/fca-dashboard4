"""
Tests for Component Registry Module

This module contains tests for the ComponentRegistry class, ensuring that
components can be properly registered and retrieved.
"""

import unittest
from unittest.mock import MagicMock, patch

from nexusml.core.pipeline.registry import ComponentRegistry, ComponentRegistryError


class TestComponentRegistry(unittest.TestCase):
    """
    Test case for ComponentRegistry class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        self.registry = ComponentRegistry()

    def test_register_and_get(self):
        """
        Test registering and retrieving components.
        """
        # Create mock component classes
        MockComponent1 = MagicMock()
        MockComponent2 = MagicMock()

        # Register components
        self.registry.register("stage", "mock1", MockComponent1)
        self.registry.register("stage", "mock2", MockComponent2)
        self.registry.register("transformer", "mock1", MockComponent1)

        # Retrieve components
        component1 = self.registry.get("stage", "mock1")
        component2 = self.registry.get("stage", "mock2")
        component3 = self.registry.get("transformer", "mock1")
        component4 = self.registry.get("stage", "nonexistent")
        component5 = self.registry.get("nonexistent", "mock1")

        # Verify retrievals
        self.assertEqual(component1, MockComponent1)
        self.assertEqual(component2, MockComponent2)
        self.assertEqual(component3, MockComponent1)
        self.assertIsNone(component4)
        self.assertIsNone(component5)

    def test_register_duplicate(self):
        """
        Test that registering a duplicate component raises an error.
        """
        # Create mock component classes
        MockComponent1 = MagicMock()
        MockComponent2 = MagicMock()

        # Register a component
        self.registry.register("stage", "mock1", MockComponent1)

        # Try to register a duplicate component
        with self.assertRaises(ComponentRegistryError):
            self.registry.register("stage", "mock1", MockComponent2)

    def test_get_all(self):
        """
        Test retrieving all components of a specific type.
        """
        # Create mock component classes
        MockComponent1 = MagicMock()
        MockComponent2 = MagicMock()
        MockComponent3 = MagicMock()

        # Register components
        self.registry.register("stage", "mock1", MockComponent1)
        self.registry.register("stage", "mock2", MockComponent2)
        self.registry.register("transformer", "mock1", MockComponent3)

        # Retrieve all components of a specific type
        stage_components = self.registry.get_all("stage")
        transformer_components = self.registry.get_all("transformer")
        nonexistent_components = self.registry.get_all("nonexistent")

        # Verify retrievals
        self.assertEqual(len(stage_components), 2)
        self.assertEqual(stage_components["mock1"], MockComponent1)
        self.assertEqual(stage_components["mock2"], MockComponent2)
        self.assertEqual(len(transformer_components), 1)
        self.assertEqual(transformer_components["mock1"], MockComponent3)
        self.assertEqual(len(nonexistent_components), 0)

    def test_get_types(self):
        """
        Test retrieving all registered component types.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())
        self.registry.register("transformer", "mock1", MagicMock())
        self.registry.register("pipeline", "mock1", MagicMock())

        # Retrieve all component types
        types = self.registry.get_types()

        # Verify retrievals
        self.assertEqual(len(types), 3)
        self.assertIn("stage", types)
        self.assertIn("transformer", types)
        self.assertIn("pipeline", types)

    def test_get_names(self):
        """
        Test retrieving all registered component names for a specific type.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())
        self.registry.register("stage", "mock2", MagicMock())
        self.registry.register("transformer", "mock1", MagicMock())

        # Retrieve all component names for a specific type
        stage_names = self.registry.get_names("stage")
        transformer_names = self.registry.get_names("transformer")
        nonexistent_names = self.registry.get_names("nonexistent")

        # Verify retrievals
        self.assertEqual(len(stage_names), 2)
        self.assertIn("mock1", stage_names)
        self.assertIn("mock2", stage_names)
        self.assertEqual(len(transformer_names), 1)
        self.assertIn("mock1", transformer_names)
        self.assertEqual(len(nonexistent_names), 0)

    def test_has_type(self):
        """
        Test checking if a component type is registered.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())

        # Check if component types are registered
        has_stage = self.registry.has_type("stage")
        has_transformer = self.registry.has_type("transformer")

        # Verify checks
        self.assertTrue(has_stage)
        self.assertFalse(has_transformer)

    def test_has_component(self):
        """
        Test checking if a component is registered.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())

        # Check if components are registered
        has_component1 = self.registry.has_component("stage", "mock1")
        has_component2 = self.registry.has_component("stage", "mock2")
        has_component3 = self.registry.has_component("transformer", "mock1")

        # Verify checks
        self.assertTrue(has_component1)
        self.assertFalse(has_component2)
        self.assertFalse(has_component3)

    def test_register_from_module(self):
        """
        Test registering components from a module.
        """
        # Create a mock module with a register_components function
        mock_module = MagicMock()
        mock_module.__name__ = "mock_module"
        mock_module.register_components = MagicMock()

        # Register components from the module
        self.registry.register_from_module(mock_module)

        # Verify that the register_components function was called
        mock_module.register_components.assert_called_once_with(self.registry)

    def test_register_from_module_no_function(self):
        """
        Test that registering components from a module without a register_components function raises an error.
        """
        # Create a mock module without a register_components function
        mock_module = MagicMock()
        mock_module.__name__ = "mock_module"
        
        # Explicitly configure hasattr to return False for register_components
        type(mock_module).register_components = MagicMock(side_effect=AttributeError)
        
        # Try to register components from the module
        with self.assertRaises(ComponentRegistryError):
            self.registry.register_from_module(mock_module)

    def test_register_from_module_error(self):
        """
        Test that errors during component registration are properly handled.
        """
        # Create a mock module with a register_components function that raises an error
        mock_module = MagicMock()
        mock_module.__name__ = "mock_module"
        mock_module.register_components = MagicMock(side_effect=ValueError("Test error"))

        # Try to register components from the module
        with self.assertRaises(ComponentRegistryError):
            self.registry.register_from_module(mock_module)

    def test_clear(self):
        """
        Test clearing all registered components.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())
        self.registry.register("transformer", "mock1", MagicMock())

        # Clear all registered components
        self.registry.clear()

        # Verify that all components are cleared
        self.assertEqual(len(self.registry.get_types()), 0)
        self.assertEqual(len(self.registry.get_all("stage")), 0)
        self.assertEqual(len(self.registry.get_all("transformer")), 0)

    def test_clear_type(self):
        """
        Test clearing all registered components of a specific type.
        """
        # Register components
        self.registry.register("stage", "mock1", MagicMock())
        self.registry.register("stage", "mock2", MagicMock())
        self.registry.register("transformer", "mock1", MagicMock())

        # Clear all registered components of a specific type
        self.registry.clear_type("stage")

        # Verify that components of the specified type are cleared
        self.assertEqual(len(self.registry.get_types()), 2)  # stage and transformer types still exist
        self.assertEqual(len(self.registry.get_all("stage")), 0)
        self.assertEqual(len(self.registry.get_all("transformer")), 1)


if __name__ == '__main__':
    unittest.main()