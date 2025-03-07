"""
Unit tests for the EnhancedModelEvaluator component.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.model_evaluator import EnhancedModelEvaluator


class TestEnhancedModelEvaluator(unittest.TestCase):
    """
    Test cases for the EnhancedModelEvaluator component.
    """

    def setUp(self):
        """
        Set up test fixtures.
        """
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)
        self.mock_config_provider.config.classification.model_dump.return_value = {
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "detailed_report": True,
                "confusion_matrix": True,
                "other_category_analysis": True,
            }
        }

        # Create the model evaluator with the mock config provider
        self.model_evaluator = EnhancedModelEvaluator(
            config_provider=self.mock_config_provider
        )

        # Create a simple model for testing
        self.test_model = Pipeline(
            [("clf", RandomForestClassifier(n_estimators=10, random_state=42))]
        )

        # Create test data
        self.x_test = pd.DataFrame(
            {
                "combined_features": ["text1", "text2", "text3", "text4"],
                "service_life": [10, 20, 30, 40],
            }
        )
        self.y_test = pd.DataFrame(
            {
                "target1": ["class1", "class2", "class1", "class3"],
                "target2": ["class3", "class4", "class3", "class4"],
            }
        )

        # Mock the model's predict method
        self.test_model.predict = MagicMock(
            return_value=pd.DataFrame(
                {
                    "target1": ["class1", "class2", "class1", "class3"],
                    "target2": ["class3", "class4", "class3", "class4"],
                }
            )
        )

    def test_evaluate(self):
        """
        Test that the evaluate method correctly evaluates a model.
        """
        # Evaluate the model
        metrics = self.model_evaluator.evaluate(
            self.test_model, self.x_test, self.y_test
        )

        # Check that the metrics dictionary contains the expected keys
        self.assertIn("target1", metrics)
        self.assertIn("target2", metrics)
        self.assertIn("overall", metrics)

        # Check that each target's metrics contain the expected keys
        for target in ["target1", "target2"]:
            self.assertIn("accuracy", metrics[target])
            self.assertIn("precision_macro", metrics[target])
            self.assertIn("recall_macro", metrics[target])
            self.assertIn("f1_macro", metrics[target])
            self.assertIn("classification_report", metrics[target])
            self.assertIn("confusion_matrix", metrics[target])
            self.assertIn("per_class", metrics[target])

        # Check that the overall metrics contain the expected keys
        self.assertIn("accuracy_mean", metrics["overall"])
        self.assertIn("f1_macro_mean", metrics["overall"])
        self.assertIn("precision_macro_mean", metrics["overall"])
        self.assertIn("recall_macro_mean", metrics["overall"])

    def test_evaluate_with_different_types(self):
        """
        Test that the evaluate method handles different types of inputs correctly.
        """
        # Create test data with numpy arrays instead of pandas Series
        y_test_numpy = pd.DataFrame(
            {
                "target1": np.array(["class1", "class2", "class1", "class3"]),
                "target2": np.array(["class3", "class4", "class3", "class4"]),
            }
        )

        # Mock the model's predict method to return numpy arrays
        self.test_model.predict = MagicMock(
            return_value=np.array(
                [
                    ["class1", "class3"],
                    ["class2", "class4"],
                    ["class1", "class3"],
                    ["class3", "class4"],
                ]
            )
        )

        # Evaluate the model
        metrics = self.model_evaluator.evaluate(
            self.test_model, self.x_test, y_test_numpy
        )

        # Check that the metrics dictionary contains the expected keys
        self.assertIn("target1", metrics)
        self.assertIn("target2", metrics)
        self.assertIn("overall", metrics)

    def test_analyze_predictions(self):
        """
        Test that the analyze_predictions method correctly analyzes predictions.
        """
        # Create predictions
        y_pred = pd.DataFrame(
            {
                "target1": ["class1", "class2", "class1", "class3"],
                "target2": ["class3", "class4", "class3", "class4"],
            }
        )

        # Analyze the predictions
        analysis = self.model_evaluator.analyze_predictions(
            self.test_model, self.x_test, self.y_test, y_pred
        )

        # Check that the analysis dictionary contains the expected keys
        self.assertIn("target1", analysis)
        self.assertIn("target2", analysis)

        # Check that each target's analysis contains the expected keys
        for target in ["target1", "target2"]:
            self.assertIn("class_distribution", analysis[target])

    def test_analyze_predictions_with_other_category(self):
        """
        Test that the analyze_predictions method correctly analyzes predictions with 'Other' category.
        """
        # Create test data with 'Other' category
        y_test_with_other = pd.DataFrame(
            {
                "target1": ["class1", "class2", "Other", "class3"],
                "target2": ["class3", "class4", "class3", "Other"],
            }
        )

        # Create predictions with 'Other' category
        y_pred_with_other = pd.DataFrame(
            {
                "target1": ["class1", "class2", "class1", "Other"],
                "target2": ["Other", "class4", "class3", "class4"],
            }
        )

        # Analyze the predictions
        analysis = self.model_evaluator.analyze_predictions(
            self.test_model, self.x_test, y_test_with_other, y_pred_with_other
        )

        # Check that the analysis dictionary contains the expected keys
        self.assertIn("target1", analysis)
        self.assertIn("target2", analysis)

        # Check that each target's analysis contains the 'Other' category analysis
        for target in ["target1", "target2"]:
            self.assertIn("other_category", analysis[target])
            self.assertIn("precision", analysis[target]["other_category"])
            self.assertIn("recall", analysis[target]["other_category"])
            self.assertIn("f1_score", analysis[target]["other_category"])

    def test_calculate_metrics(self):
        """
        Test that the _calculate_metrics method correctly calculates metrics.
        """
        # Create test data
        y_true = pd.Series(["class1", "class2", "class1", "class3"])
        y_pred = pd.Series(["class1", "class2", "class1", "class3"])

        # Calculate metrics
        metrics = self.model_evaluator._calculate_metrics(y_true, y_pred)

        # Check that the metrics dictionary contains the expected keys
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class", metrics)

        # Check that the accuracy is 1.0 (perfect prediction)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_calculate_metrics_with_different_types(self):
        """
        Test that the _calculate_metrics method handles different types of inputs correctly.
        """
        # Create test data with numpy arrays
        y_true = np.array(["class1", "class2", "class1", "class3"])
        y_pred = np.array(["class1", "class2", "class1", "class3"])

        # Calculate metrics
        metrics = self.model_evaluator._calculate_metrics(y_true, y_pred)

        # Check that the metrics dictionary contains the expected keys
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertIn("classification_report", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("per_class", metrics)

        # Check that the accuracy is 1.0 (perfect prediction)
        self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
