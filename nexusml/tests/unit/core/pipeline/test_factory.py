"""
Tests for Pipeline Factory Module

This module contains tests for the PipelineFactory class, ensuring that
pipelines can be properly created and configured.
"""

import unittest
from unittest.mock import MagicMock, patch

from nexusml.config.manager import ConfigurationManager
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory, PipelineFactoryError
from nexusml.core.pipeline.pipelines.base import BasePipeline
from nexusml.core.pipeline.pipelines.training import TrainingPipeline
from nexusml.core.pipeline.pipelines.prediction import PredictionPipeline
from nexusml.core.pipeline.pipelines.evaluation import EvaluationPipeline
from nexusml.core.pipeline.registry import ComponentRegistry


class TestPipelineFactory(unittest.TestCase):
    """
    Test case for PipelineFactory class.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        self.registry = MagicMock(spec=ComponentRegistry)
        self.container = MagicMock(spec=DIContainer)
        self.config_manager = MagicMock(spec=ConfigurationManager)
        self.factory = PipelineFactory(self.registry, self.container, self.config_manager)

    def test_create_pipeline(self):
        """
        Test creating a pipeline of a specific type.
        """
        # Test creating a training pipeline
        pipeline = self.factory.create_pipeline("training", {"test_key": "test_value"})
        self.assertIsInstance(pipeline, TrainingPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

        # Test creating a prediction pipeline
        pipeline = self.factory.create_pipeline("prediction", {"test_key": "test_value"})
        self.assertIsInstance(pipeline, PredictionPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

        # Test creating an evaluation pipeline
        pipeline = self.factory.create_pipeline("evaluation", {"test_key": "test_value"})
        self.assertIsInstance(pipeline, EvaluationPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

    def test_create_pipeline_unsupported_type(self):
        """
        Test that creating a pipeline of an unsupported type raises an error.
        """
        # Set up the registry to return None for the unsupported type
        self.registry.get.return_value = None

        # Try to create a pipeline of an unsupported type
        with self.assertRaises(PipelineFactoryError):
            self.factory.create_pipeline("unsupported", {})

    def test_create_pipeline_from_registry(self):
        """
        Test creating a pipeline from the registry.
        """
        # Create a mock pipeline class
        mock_pipeline_class = MagicMock(return_value=MagicMock(spec=BasePipeline))

        # Set up the registry to return the mock pipeline class
        self.registry.get.return_value = mock_pipeline_class

        # Create a pipeline of a custom type
        pipeline = self.factory.create_pipeline("custom", {"test_key": "test_value"})

        # Verify that the registry was queried
        self.registry.get.assert_called_once_with("pipeline", "custom")

        # Verify that the pipeline was created with the correct arguments
        mock_pipeline_class.assert_called_once_with(
            config={"test_key": "test_value"}, container=self.container
        )

    def test_register_pipeline_type(self):
        """
        Test registering a new pipeline type.
        """
        # Create a mock pipeline class
        mock_pipeline_class = MagicMock(spec=BasePipeline)

        # Register the pipeline type
        self.factory.register_pipeline_type("custom", mock_pipeline_class)

        # Verify that the pipeline type was registered
        self.registry.register.assert_called_once_with(
            "pipeline", "custom", mock_pipeline_class
        )

        # Verify that the pipeline type is in the factory's pipeline types
        self.assertIn("custom", self.factory._pipeline_types)
        self.assertEqual(self.factory._pipeline_types["custom"], mock_pipeline_class)

    def test_create_training_pipeline(self):
        """
        Test creating a training pipeline.
        """
        # Create a training pipeline
        pipeline = self.factory.create_training_pipeline({"test_key": "test_value"})

        # Verify that the pipeline is a TrainingPipeline
        self.assertIsInstance(pipeline, TrainingPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

    def test_create_prediction_pipeline(self):
        """
        Test creating a prediction pipeline.
        """
        # Create a prediction pipeline
        pipeline = self.factory.create_prediction_pipeline({"test_key": "test_value"})

        # Verify that the pipeline is a PredictionPipeline
        self.assertIsInstance(pipeline, PredictionPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

    def test_create_evaluation_pipeline(self):
        """
        Test creating an evaluation pipeline.
        """
        # Create an evaluation pipeline
        pipeline = self.factory.create_evaluation_pipeline({"test_key": "test_value"})

        # Verify that the pipeline is an EvaluationPipeline
        self.assertIsInstance(pipeline, EvaluationPipeline)
        self.assertEqual(pipeline.config, {"test_key": "test_value"})
        self.assertEqual(pipeline.container, self.container)

    def test_create_pipeline_from_config(self):
        """
        Test creating a pipeline from a configuration file.
        """
        # Set up the config manager to return a configuration
        self.config_manager.load_config.return_value = {
            "pipeline_type": "training",
            "test_key": "test_value",
        }

        # Create a pipeline from a configuration file
        pipeline = self.factory.create_pipeline_from_config("test_config.yml")

        # Verify that the config manager was called
        self.config_manager.load_config.assert_called_once_with("test_config.yml")

        # Verify that the pipeline is a TrainingPipeline
        self.assertIsInstance(pipeline, TrainingPipeline)
        self.assertEqual(
            pipeline.config, {"pipeline_type": "training", "test_key": "test_value"}
        )
        self.assertEqual(pipeline.container, self.container)

    def test_create_pipeline_from_config_no_type(self):
        """
        Test that creating a pipeline from a configuration file without a pipeline type raises an error.
        """
        # Set up the config manager to return a configuration without a pipeline type
        self.config_manager.load_config.return_value = {"test_key": "test_value"}

        # Try to create a pipeline from a configuration file without a pipeline type
        with self.assertRaises(PipelineFactoryError):
            self.factory.create_pipeline_from_config("test_config.yml")

    def test_create_pipeline_from_config_error(self):
        """
        Test that errors during pipeline creation from a configuration file are properly handled.
        """
        # Set up the config manager to raise an error
        self.config_manager.load_config.side_effect = ValueError("Test error")

        # Try to create a pipeline from a configuration file that raises an error
        with self.assertRaises(PipelineFactoryError):
            self.factory.create_pipeline_from_config("test_config.yml")


if __name__ == '__main__':
    unittest.main()