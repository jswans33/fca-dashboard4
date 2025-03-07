"""
Unit tests for the PickleModelSerializer component.
"""

import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.model_serializer import PickleModelSerializer


class TestPickleModelSerializer(unittest.TestCase):
    """
    Test cases for the PickleModelSerializer component.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)
        self.mock_config_provider.config.classification.model_dump.return_value = {
            "serialization": {
                "default_directory": "test_outputs/models",
                "protocol": pickle.HIGHEST_PROTOCOL,
                "compress": True,
                "file_extension": ".pkl",
            }
        }

        # Create the model serializer with the mock config provider
        self.model_serializer = PickleModelSerializer(
            config_provider=self.mock_config_provider
        )

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

    def test_save_model(self):
        """
        Test that the save_model method correctly saves a model to disk.
        """
        # Define the path for the test model
        model_path = self.test_dir / "test_model.pkl"

        # Save the model
        self.model_serializer.save_model(self.test_model, model_path)

        # Check that the file exists
        self.assertTrue(model_path.exists())

        # Check that the file contains a valid model
        with open(model_path, "rb") as f:
            loaded_model = pickle.load(f)

        # Check that the loaded model is a Pipeline
        self.assertIsInstance(loaded_model, Pipeline)

    def test_save_model_with_metadata(self):
        """
        Test that the save_model method correctly saves a model with metadata.
        """
        # Define the path for the test model
        model_path = self.test_dir / "test_model_with_metadata.pkl"
        metadata = {"created_by": "test", "version": "1.0.0"}

        # Save the model with metadata
        self.model_serializer.save_model(
            self.test_model, model_path, save_metadata=True, metadata=metadata
        )

        # Check that the model file exists
        self.assertTrue(model_path.exists())

        # Check that the metadata file exists
        metadata_path = model_path.with_suffix(".meta.json")
        self.assertTrue(metadata_path.exists())

        # Check that the metadata file contains the correct data
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)

        self.assertEqual(loaded_metadata, metadata)

    def test_load_model(self):
        """
        Test that the load_model method correctly loads a model from disk.
        """
        # Define the path for the test model
        model_path = self.test_dir / "test_model_for_loading.pkl"

        # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(self.test_model, f)

        # Load the model
        loaded_model = self.model_serializer.load_model(model_path)

        # Check that the loaded model is a Pipeline
        self.assertIsInstance(loaded_model, Pipeline)

    def test_load_model_with_metadata(self):
        """
        Test that the load_model method correctly loads a model with metadata.
        """
        # Define the path for the test model
        model_path = self.test_dir / "test_model_with_metadata_for_loading.pkl"
        metadata_path = model_path.with_suffix(".meta.json")
        metadata = {"created_by": "test", "version": "1.0.0"}

        # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(self.test_model, f)

        # Save the metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Create a dictionary to store the metadata
        metadata_dict = {}

        # Define a function to capture the metadata
        def store_metadata(key, value):
            metadata_dict[key] = value

        # Patch the model_serializer to store metadata in our dictionary
        with patch.object(self.model_serializer, "config") as mock_config:
            # Configure the mock
            mock_config.get.return_value = {"file_extension": ".pkl"}

            # Load the model with metadata
            loaded_model = self.model_serializer.load_model(
                model_path, load_metadata=True, metadata_callback=store_metadata
            )

            # Check that the loaded model is a Pipeline
            self.assertIsInstance(loaded_model, Pipeline)

            # Check that the metadata was loaded and passed to our callback
            self.assertIn("metadata", metadata_dict)
            self.assertEqual(metadata_dict["metadata"], metadata)

    def test_load_model_nonexistent_file(self):
        """
        Test that the load_model method raises an error for nonexistent files.
        """
        # Define a path for a nonexistent model
        nonexistent_path = self.test_dir / "nonexistent_model.pkl"

        # Check that loading a nonexistent model raises an error
        with self.assertRaises(IOError):
            self.model_serializer.load_model(nonexistent_path)

    def test_load_model_invalid_file(self):
        """
        Test that the load_model method raises an error for invalid files.
        """
        # Define the path for an invalid model
        invalid_path = self.test_dir / "invalid_model.pkl"

        # Create an invalid model file (not a pickle file)
        with open(invalid_path, "w") as f:
            f.write("This is not a pickle file")

        # Check that loading an invalid model raises an error
        with self.assertRaises(IOError):
            self.model_serializer.load_model(invalid_path)

    def test_list_saved_models(self):
        """
        Test that the list_saved_models method correctly lists saved models.
        """
        # Create some test models
        model_names = ["model1", "model2", "model3"]
        for name in model_names:
            model_path = self.test_dir / f"{name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.test_model, f)

        # List the saved models
        models = self.model_serializer.list_saved_models(self.test_dir)

        # Check that all models are listed
        self.assertEqual(len(models), len(model_names))
        for name in model_names:
            self.assertIn(name, models)

    def test_list_saved_models_with_metadata(self):
        """
        Test that the list_saved_models method correctly lists saved models with metadata.
        """
        # Create some test models with metadata
        model_names = ["model1", "model2", "model3"]
        for i, name in enumerate(model_names):
            model_path = self.test_dir / f"{name}.pkl"
            metadata_path = model_path.with_suffix(".meta.json")

            # Save the model
            with open(model_path, "wb") as f:
                pickle.dump(self.test_model, f)

            # Save the metadata
            metadata = {"created_by": "test", "version": f"1.0.{i}"}
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

        # List the saved models
        models = self.model_serializer.list_saved_models(self.test_dir)

        # Check that all models are listed with their metadata
        self.assertEqual(len(models), len(model_names))
        for i, name in enumerate(model_names):
            self.assertIn(name, models)
            self.assertIn("metadata", models[name])
            self.assertEqual(models[name]["metadata"]["version"], f"1.0.{i}")

    def test_delete_model(self):
        """
        Test that the delete_model method correctly deletes a model.
        """
        # Create a test model
        model_path = self.test_dir / "model_to_delete.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.test_model, f)

        # Delete the model
        result = self.model_serializer.delete_model(model_path)

        # Check that the model was deleted
        self.assertTrue(result)
        self.assertFalse(model_path.exists())

    def test_delete_model_with_metadata(self):
        """
        Test that the delete_model method correctly deletes a model with metadata.
        """
        # Create a test model with metadata
        model_path = self.test_dir / "model_with_metadata_to_delete.pkl"
        metadata_path = model_path.with_suffix(".meta.json")

        # Save the model
        with open(model_path, "wb") as f:
            pickle.dump(self.test_model, f)

        # Save the metadata
        metadata = {"created_by": "test", "version": "1.0.0"}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Delete the model
        result = self.model_serializer.delete_model(model_path)

        # Check that the model and metadata were deleted
        self.assertTrue(result)
        self.assertFalse(model_path.exists())
        self.assertFalse(metadata_path.exists())

    def test_delete_nonexistent_model(self):
        """
        Test that the delete_model method returns False for nonexistent models.
        """
        # Define a path for a nonexistent model
        nonexistent_path = self.test_dir / "nonexistent_model.pkl"

        # Try to delete the nonexistent model
        result = self.model_serializer.delete_model(nonexistent_path)

        # Check that the method returned False
        self.assertFalse(result)

    def test_directory_path_handling(self):
        """
        Test that the list_saved_models method handles None directory paths correctly.
        """
        # Create a clean test directory for this test
        clean_dir = tempfile.TemporaryDirectory()

        # Mock the default directory path to point to our clean directory
        with patch.object(self.model_serializer, "config") as mock_config:
            mock_config.get.return_value = {
                "default_directory": clean_dir.name,
                "file_extension": ".pkl",
            }

            # Call the method with None directory
            result = self.model_serializer.list_saved_models(None)

            # Check that the method returned an empty dictionary
            self.assertEqual(result, {})

        # Clean up
        clean_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
