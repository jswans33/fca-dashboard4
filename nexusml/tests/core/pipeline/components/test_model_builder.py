"""
Unit tests for the RandomForestModelBuilder component.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.model_builder import RandomForestModelBuilder


class TestRandomForestModelBuilder(unittest.TestCase):
    """
    Test cases for the RandomForestModelBuilder component.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)
        self.mock_config_provider.config.classification.model_dump.return_value = {
            "model": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "class_weight": "balanced",
                    "random_state": 42,
                }
            }
        }

        # Create the model builder with the mock config provider
        self.model_builder = RandomForestModelBuilder(
            config_provider=self.mock_config_provider
        )

    def test_build_model(self):
        """
        Test that the build_model method creates a valid pipeline.
        """
        # Build the model
        model = self.model_builder.build_model()

        # Check that the model is a Pipeline
        self.assertIsInstance(model, Pipeline)

        # Check that the pipeline has the expected steps
        self.assertIn("preprocessor", model.named_steps)
        self.assertIn("clf", model.named_steps)

        # Check that the preprocessor has the expected transformers
        preprocessor = model.named_steps["preprocessor"]
        self.assertEqual(len(preprocessor.transformers), 2)
        self.assertEqual(preprocessor.transformers[0][0], "text")
        self.assertEqual(preprocessor.transformers[1][0], "numeric")

    def test_optimize_hyperparameters(self):
        """
        Test that the optimize_hyperparameters method works correctly.
        """
        # Create a mock model and data
        mock_model = MagicMock(spec=Pipeline)
        mock_x_train = pd.DataFrame(
            {"combined_features": ["text1", "text2"], "service_life": [10, 20]}
        )
        mock_y_train = pd.DataFrame(
            {"target1": ["class1", "class2"], "target2": ["class3", "class4"]}
        )

        # Mock the GridSearchCV
        with patch(
            "nexusml.core.pipeline.components.model_builder.GridSearchCV"
        ) as mock_grid_search:
            # Configure the mock
            mock_grid_search.return_value.fit.return_value = None
            mock_grid_search.return_value.best_estimator_ = mock_model
            mock_grid_search.return_value.best_params_ = {"param1": "value1"}
            mock_grid_search.return_value.best_score_ = 0.95

            # Call the method
            result = self.model_builder.optimize_hyperparameters(
                mock_model, mock_x_train, mock_y_train
            )

            # Check that GridSearchCV was called with the expected arguments
            mock_grid_search.assert_called_once()
            args, kwargs = mock_grid_search.call_args
            self.assertEqual(args[0], mock_model)
            self.assertIn("param_grid", kwargs)
            self.assertIn("cv", kwargs)
            self.assertIn("scoring", kwargs)

            # Check that the result is the best estimator
            self.assertEqual(result, mock_model)


if __name__ == "__main__":
    unittest.main()
