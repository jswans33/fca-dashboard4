"""
Unit tests for the model adapter classes.
"""

import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.pipeline.adapters.model_adapter import (
    LegacyModelBuilderAdapter,
    LegacyModelEvaluatorAdapter,
    LegacyModelSerializerAdapter,
    LegacyModelTrainerAdapter,
    ModelComponentFactory,
)
from nexusml.core.pipeline.components.model_builder import RandomForestModelBuilder
from nexusml.core.pipeline.components.model_evaluator import EnhancedModelEvaluator
from nexusml.core.pipeline.components.model_serializer import PickleModelSerializer
from nexusml.core.pipeline.components.model_trainer import StandardModelTrainer


class TestModelComponentFactory(unittest.TestCase):
    """
    Test cases for the ModelComponentFactory.
    """

    def test_create_model_builder_standard(self):
        """
        Test that create_model_builder creates a RandomForestModelBuilder when use_legacy is False.
        """
        builder = ModelComponentFactory.create_model_builder(use_legacy=False)
        self.assertIsInstance(builder, RandomForestModelBuilder)

    def test_create_model_builder_legacy(self):
        """
        Test that create_model_builder creates a LegacyModelBuilderAdapter when use_legacy is True.
        """
        builder = ModelComponentFactory.create_model_builder(use_legacy=True)
        self.assertIsInstance(builder, LegacyModelBuilderAdapter)

    def test_create_model_trainer_standard(self):
        """
        Test that create_model_trainer creates a StandardModelTrainer when use_legacy is False.
        """
        trainer = ModelComponentFactory.create_model_trainer(use_legacy=False)
        self.assertIsInstance(trainer, StandardModelTrainer)

    def test_create_model_trainer_legacy(self):
        """
        Test that create_model_trainer creates a LegacyModelTrainerAdapter when use_legacy is True.
        """
        trainer = ModelComponentFactory.create_model_trainer(use_legacy=True)
        self.assertIsInstance(trainer, LegacyModelTrainerAdapter)

    def test_create_model_evaluator_standard(self):
        """
        Test that create_model_evaluator creates an EnhancedModelEvaluator when use_legacy is False.
        """
        evaluator = ModelComponentFactory.create_model_evaluator(use_legacy=False)
        self.assertIsInstance(evaluator, EnhancedModelEvaluator)

    def test_create_model_evaluator_legacy(self):
        """
        Test that create_model_evaluator creates a LegacyModelEvaluatorAdapter when use_legacy is True.
        """
        evaluator = ModelComponentFactory.create_model_evaluator(use_legacy=True)
        self.assertIsInstance(evaluator, LegacyModelEvaluatorAdapter)

    def test_create_model_serializer_standard(self):
        """
        Test that create_model_serializer creates a PickleModelSerializer when use_legacy is False.
        """
        serializer = ModelComponentFactory.create_model_serializer(use_legacy=False)
        self.assertIsInstance(serializer, PickleModelSerializer)

    def test_create_model_serializer_legacy(self):
        """
        Test that create_model_serializer creates a LegacyModelSerializerAdapter when use_legacy is True.
        """
        serializer = ModelComponentFactory.create_model_serializer(use_legacy=True)
        self.assertIsInstance(serializer, LegacyModelSerializerAdapter)


class TestLegacyModelSerializerAdapter(unittest.TestCase):
    """
    Test cases for the LegacyModelSerializerAdapter.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create the model serializer
        self.model_serializer = LegacyModelSerializerAdapter()

        # Create a simple model for testing
        self.test_model = Pipeline(
            [("clf", RandomForestClassifier(n_estimators=10, random_state=42))]
        )

        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """
        Clean up test fixtures.
        """
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_save_and_load_model(self):
        """
        Test that save_model and load_model work correctly together.
        """
        # Define the path for the test model
        model_path = self.test_dir / "test_model.pkl"

        # Save the model
        self.model_serializer.save_model(self.test_model, model_path)

        # Check that the file exists
        self.assertTrue(model_path.exists())

        # Load the model
        loaded_model = self.model_serializer.load_model(model_path)

        # Check that the loaded model is a Pipeline
        self.assertIsInstance(loaded_model, Pipeline)

    def test_load_model_nonexistent_file(self):
        """
        Test that load_model raises an error for nonexistent files.
        """
        # Define a path for a nonexistent model
        nonexistent_path = self.test_dir / "nonexistent_model.pkl"

        # Check that loading a nonexistent model raises an error
        with self.assertRaises(IOError):
            self.model_serializer.load_model(nonexistent_path)

    def test_load_model_invalid_file(self):
        """
        Test that load_model raises an error for invalid files.
        """
        # Define the path for an invalid model
        invalid_path = self.test_dir / "invalid_model.pkl"

        # Create an invalid model file (not a pickle file)
        with open(invalid_path, "w") as f:
            f.write("This is not a pickle file")

        # Check that loading an invalid model raises an error
        with self.assertRaises(IOError):
            self.model_serializer.load_model(invalid_path)

    def test_get_name(self):
        """
        Test that get_name returns the correct name.
        """
        self.assertEqual(
            self.model_serializer.get_name(), "LegacyModelSerializerAdapter"
        )

    def test_get_description(self):
        """
        Test that get_description returns the correct description.
        """
        self.assertEqual(
            self.model_serializer.get_description(),
            "Adapter for legacy model serialization",
        )

    def test_validate_config(self):
        """
        Test that validate_config always returns True.
        """
        self.assertTrue(self.model_serializer.validate_config({}))


class TestLegacyModelBuilderAdapter(unittest.TestCase):
    """
    Test cases for the LegacyModelBuilderAdapter.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create the model builder
        self.model_builder = LegacyModelBuilderAdapter()

    def test_get_name(self):
        """
        Test that get_name returns the correct name.
        """
        self.assertEqual(self.model_builder.get_name(), "LegacyModelBuilderAdapter")

    def test_get_description(self):
        """
        Test that get_description returns the correct description.
        """
        self.assertEqual(
            self.model_builder.get_description(),
            "Adapter for legacy model building functions",
        )

    def test_validate_config(self):
        """
        Test that validate_config always returns True.
        """
        self.assertTrue(self.model_builder.validate_config({}))

    @patch("nexusml.core.pipeline.adapters.model_adapter.build_enhanced_model")
    def test_build_model(self, mock_build_enhanced_model):
        """
        Test that build_model calls the legacy function.
        """
        # Configure the mock
        mock_model = Pipeline([("clf", RandomForestClassifier())])
        mock_build_enhanced_model.return_value = mock_model

        # Call the method
        result = self.model_builder.build_model()

        # Check that the legacy function was called
        mock_build_enhanced_model.assert_called_once()

        # Check that the result is the mock model
        self.assertEqual(result, mock_model)


if __name__ == "__main__":
    unittest.main()
