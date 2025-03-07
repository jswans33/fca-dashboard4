"""
Unit tests for the StandardModelTrainer component.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.model_trainer import StandardModelTrainer


class TestStandardModelTrainer(unittest.TestCase):
    """
    Test cases for the StandardModelTrainer component.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)
        self.mock_config_provider.config.classification.model_dump.return_value = {
            "training": {
                "validation_split": 0.2,
                "random_state": 42,
                "stratify": True,
                "verbose": 1,
            }
        }

        # Create the model trainer with the mock config provider
        self.model_trainer = StandardModelTrainer(
            config_provider=self.mock_config_provider
        )

        # Create a simple model for testing
        self.test_model = MagicMock(spec=Pipeline)

        # Create test data
        self.x_train = pd.DataFrame(
            {
                "combined_features": ["text1", "text2", "text3", "text4"],
                "service_life": [10, 20, 30, 40],
            }
        )
        self.y_train = pd.DataFrame(
            {
                "target1": ["class1", "class2", "class1", "class3"],
                "target2": ["class3", "class4", "class3", "class4"],
            }
        )

    def test_train(self):
        """
        Test that the train method correctly trains a model.
        """
        # Train the model
        trained_model = self.model_trainer.train(
            self.test_model, self.x_train, self.y_train
        )

        # Check that the model's fit method was called
        self.test_model.fit.assert_called_once_with(self.x_train, self.y_train)

        # Check that the trained model is the same as the test model
        self.assertEqual(trained_model, self.test_model)

    def test_train_with_validation_data(self):
        """
        Test that the train method correctly trains a model with validation data.
        """
        # Create validation data
        x_val = pd.DataFrame(
            {"combined_features": ["text5", "text6"], "service_life": [50, 60]}
        )
        y_val = pd.DataFrame(
            {"target1": ["class1", "class2"], "target2": ["class3", "class4"]}
        )

        # Train the model with validation data
        trained_model = self.model_trainer.train(
            self.test_model, self.x_train, self.y_train, x_val=x_val, y_val=y_val
        )

        # Check that the model's fit method was called
        self.assertTrue(self.test_model.fit.called)

        # Check that the first two arguments were x_train and y_train
        args, kwargs = self.test_model.fit.call_args
        pd.testing.assert_frame_equal(args[0], self.x_train)
        pd.testing.assert_frame_equal(args[1], self.y_train)

        # Check that x_val and y_val were passed as kwargs
        self.assertIn("x_val", kwargs)
        self.assertIn("y_val", kwargs)
        pd.testing.assert_frame_equal(kwargs["x_val"], x_val)
        pd.testing.assert_frame_equal(kwargs["y_val"], y_val)

        # Check that the trained model is the same as the test model
        self.assertEqual(trained_model, self.test_model)

    def test_train_with_custom_parameters(self):
        """
        Test that the train method correctly trains a model with custom parameters.
        """
        # Train the model with custom parameters
        trained_model = self.model_trainer.train(
            self.test_model,
            self.x_train,
            self.y_train,
            validation_split=0.3,
            random_state=123,
        )

        # Check that the model's fit method was called
        self.assertTrue(self.test_model.fit.called)

        # Check that the first two arguments were x_train and y_train
        args, kwargs = self.test_model.fit.call_args
        pd.testing.assert_frame_equal(args[0], self.x_train)
        pd.testing.assert_frame_equal(args[1], self.y_train)

        # Check that validation_split and random_state were passed as kwargs
        self.assertIn("validation_split", kwargs)
        self.assertIn("random_state", kwargs)
        self.assertEqual(kwargs["validation_split"], 0.3)
        self.assertEqual(kwargs["random_state"], 123)

        # Check that the trained model is the same as the test model
        self.assertEqual(trained_model, self.test_model)

    def test_cross_validate(self):
        """
        Test that the cross_validate method correctly performs cross-validation.
        """
        # Mock the cross_validate function
        with patch(
            "nexusml.core.pipeline.components.model_trainer.cross_validate"
        ) as mock_cv:
            # Configure the mock
            mock_cv.return_value = {
                "fit_time": [0.1, 0.2, 0.3],
                "score_time": [0.01, 0.02, 0.03],
                "test_accuracy": [0.8, 0.9, 0.85],
                "train_accuracy": [0.9, 0.95, 0.92],
            }

            # Call the method
            results = self.model_trainer.cross_validate(
                self.test_model, self.x_train, self.y_train, cv=3, scoring="accuracy"
            )

            # Check that cross_validate was called with the expected arguments
            mock_cv.assert_called_once()
            args, kwargs = mock_cv.call_args
            self.assertEqual(args[0], self.test_model)
            # For DataFrames, we need to check equality differently
            pd.testing.assert_frame_equal(args[1], self.x_train)
            pd.testing.assert_frame_equal(args[2], self.y_train)
            self.assertEqual(kwargs["cv"], 3)
            self.assertEqual(kwargs["scoring"], "accuracy")
            self.assertEqual(kwargs["return_train_score"], True)

            # Check that the results contain the expected keys
            self.assertIn("fit_time", results)
            self.assertIn("score_time", results)
            self.assertIn("test_accuracy", results)
            self.assertIn("train_accuracy", results)


if __name__ == "__main__":
    unittest.main()
