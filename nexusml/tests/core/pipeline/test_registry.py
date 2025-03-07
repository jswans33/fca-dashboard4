"""
Tests for the ComponentRegistry class.

This module contains tests for the ComponentRegistry class, which is responsible
for registering and retrieving component implementations.
"""

import unittest
from typing import Any, Dict, Type

import pytest

from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.core.pipeline.registry import ComponentRegistry, ComponentRegistryError


class MockDataLoader(DataLoader):
    """Mock implementation of DataLoader for testing."""

    def load_data(self, data_path=None, **kwargs):
        import pandas as pd

        return pd.DataFrame()

    def get_config(self):
        return {}


class AnotherMockDataLoader(DataLoader):
    """Another mock implementation of DataLoader for testing."""

    def load_data(self, data_path=None, **kwargs):
        import pandas as pd

        return pd.DataFrame()

    def get_config(self):
        return {}


class MockDataPreprocessor(DataPreprocessor):
    """Mock implementation of DataPreprocessor for testing."""

    def preprocess(self, data, **kwargs):
        import pandas as pd

        return pd.DataFrame()

    def verify_required_columns(self, data):
        import pandas as pd

        return pd.DataFrame()


class TestComponentRegistry(unittest.TestCase):
    """Test cases for the ComponentRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ComponentRegistry()

    def test_register_component(self):
        """Test registering a component implementation."""
        # Register a component
        self.registry.register(DataLoader, "mock", MockDataLoader)

        # Verify it was registered
        implementations = self.registry.get_implementations(DataLoader)
        self.assertIn("mock", implementations)
        self.assertEqual(implementations["mock"], MockDataLoader)

    def test_register_duplicate_component(self):
        """Test registering a component with a name that already exists."""
        # Register a component
        self.registry.register(DataLoader, "mock", MockDataLoader)

        # Try to register another component with the same name
        with pytest.raises(ComponentRegistryError):
            self.registry.register(DataLoader, "mock", AnotherMockDataLoader)

    def test_register_multiple_components(self):
        """Test registering multiple components of the same type."""
        # Register multiple components
        self.registry.register(DataLoader, "mock1", MockDataLoader)
        self.registry.register(DataLoader, "mock2", AnotherMockDataLoader)

        # Verify they were registered
        implementations = self.registry.get_implementations(DataLoader)
        self.assertIn("mock1", implementations)
        self.assertIn("mock2", implementations)
        self.assertEqual(implementations["mock1"], MockDataLoader)
        self.assertEqual(implementations["mock2"], AnotherMockDataLoader)

    def test_register_different_component_types(self):
        """Test registering components of different types."""
        # Register components of different types
        self.registry.register(DataLoader, "mock_loader", MockDataLoader)
        self.registry.register(
            DataPreprocessor, "mock_preprocessor", MockDataPreprocessor
        )

        # Verify they were registered
        loader_implementations = self.registry.get_implementations(DataLoader)
        preprocessor_implementations = self.registry.get_implementations(
            DataPreprocessor
        )

        self.assertIn("mock_loader", loader_implementations)
        self.assertIn("mock_preprocessor", preprocessor_implementations)

    def test_get_implementation(self):
        """Test getting a specific implementation."""
        # Register a component
        self.registry.register(DataLoader, "mock", MockDataLoader)

        # Get the implementation
        implementation = self.registry.get_implementation(DataLoader, "mock")
        self.assertEqual(implementation, MockDataLoader)

    def test_get_nonexistent_implementation(self):
        """Test getting an implementation that doesn't exist."""
        with pytest.raises(ComponentRegistryError):
            self.registry.get_implementation(DataLoader, "nonexistent")

    def test_get_implementations_empty(self):
        """Test getting implementations when none are registered."""
        implementations = self.registry.get_implementations(DataLoader)
        self.assertEqual(implementations, {})

    def test_get_default_implementation(self):
        """Test getting the default implementation."""
        # Register components
        self.registry.register(DataLoader, "mock1", MockDataLoader)
        self.registry.register(DataLoader, "mock2", AnotherMockDataLoader)

        # Set a default implementation
        self.registry.set_default_implementation(DataLoader, "mock2")

        # Get the default implementation
        implementation = self.registry.get_default_implementation(DataLoader)
        self.assertEqual(implementation, AnotherMockDataLoader)

    def test_get_default_implementation_not_set(self):
        """Test getting the default implementation when none is set."""
        # Register a component
        self.registry.register(DataLoader, "mock", MockDataLoader)

        # Try to get the default implementation
        with pytest.raises(ComponentRegistryError):
            self.registry.get_default_implementation(DataLoader)

    def test_set_nonexistent_default_implementation(self):
        """Test setting a default implementation that doesn't exist."""
        with pytest.raises(ComponentRegistryError):
            self.registry.set_default_implementation(DataLoader, "nonexistent")

    def test_has_implementation(self):
        """Test checking if an implementation exists."""
        # Register a component
        self.registry.register(DataLoader, "mock", MockDataLoader)

        # Check if implementations exist
        self.assertTrue(self.registry.has_implementation(DataLoader, "mock"))
        self.assertFalse(self.registry.has_implementation(DataLoader, "nonexistent"))

    def test_clear_implementations(self):
        """Test clearing all implementations of a component type."""
        # Register components
        self.registry.register(DataLoader, "mock1", MockDataLoader)
        self.registry.register(DataLoader, "mock2", AnotherMockDataLoader)

        # Clear implementations
        self.registry.clear_implementations(DataLoader)

        # Verify they were cleared
        implementations = self.registry.get_implementations(DataLoader)
        self.assertEqual(implementations, {})
