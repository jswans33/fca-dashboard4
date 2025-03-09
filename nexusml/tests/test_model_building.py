"""
Tests for the model building components.

This module contains tests for the model building components in the NexusML suite.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

from nexusml.core.model_building import (
    BaseModelBuilder,
    BaseConfigurableModelBuilder,
    BaseModelTrainer,
    BaseConfigurableModelTrainer,
    BaseHyperparameterOptimizer,
    BaseModelEvaluator,
    BaseModelSerializer,
    RandomForestBuilder,
    GradientBoostingBuilder,
    EnsembleBuilder,
)
from nexusml.core.model_training import (
    StandardModelTrainer,
    CrossValidationTrainer,
    GridSearchOptimizer,
    RandomizedSearchOptimizer,
)


class TestBaseModelBuilder(unittest.TestCase):
    """Tests for the BaseModelBuilder class."""
    
    def test_init(self):
        """Test initialization."""
        builder = BaseModelBuilder(name="TestBuilder", description="Test description")
        self.assertEqual(builder.get_name(), "TestBuilder")
        self.assertEqual(builder.get_description(), "Test description")
    
    def test_get_default_parameters(self):
        """Test get_default_parameters method."""
        builder = BaseModelBuilder()
        self.assertEqual(builder.get_default_parameters(), {})
    
    def test_get_param_grid(self):
        """Test get_param_grid method."""
        builder = BaseModelBuilder()
        self.assertEqual(builder.get_param_grid(), {})
    
    def test_build_model_not_implemented(self):
        """Test build_model method raises NotImplementedError."""
        builder = BaseModelBuilder()
        with self.assertRaises(NotImplementedError):
            builder.build_model()
    
    def test_optimize_hyperparameters(self):
        """Test optimize_hyperparameters method."""
        builder = BaseModelBuilder()
        model = MagicMock(spec=Pipeline)
        x_train = pd.DataFrame({"A": [1, 2, 3]})
        y_train = pd.DataFrame({"B": [4, 5, 6]})
        
        result = builder.optimize_hyperparameters(model, x_train, y_train)
        
        self.assertIs(result, model)


class TestBaseModelTrainer(unittest.TestCase):
    """Tests for the BaseModelTrainer class."""
    
    def test_init(self):
        """Test initialization."""
        trainer = BaseModelTrainer(name="TestTrainer", description="Test description")
        self.assertEqual(trainer.get_name(), "TestTrainer")
        self.assertEqual(trainer.get_description(), "Test description")
    
    def test_train(self):
        """Test train method."""
        trainer = BaseModelTrainer()
        model = MagicMock(spec=Pipeline)
        x_train = pd.DataFrame({"A": [1, 2, 3]})
        y_train = pd.DataFrame({"B": [4, 5, 6]})
        
        result = trainer.train(model, x_train, y_train)
        
        self.assertIs(result, model)
        model.fit.assert_called_once_with(x_train, y_train)
    
    @patch("nexusml.core.model_building.base.cross_validate")
    def test_cross_validate(self, mock_cross_validate):
        """Test cross_validate method."""
        trainer = BaseModelTrainer()
        model = MagicMock(spec=Pipeline)
        x = pd.DataFrame({"A": [1, 2, 3]})
        y = pd.DataFrame({"B": [4, 5, 6]})
        
        # Mock the cross_validate function
        mock_cross_validate.return_value = {
            "train_score": np.array([0.8, 0.9]),
            "test_score": np.array([0.7, 0.8]),
            "fit_time": np.array([0.1, 0.2]),
            "score_time": np.array([0.01, 0.02]),
        }
        
        result = trainer.cross_validate(model, x, y, cv=2)
        
        mock_cross_validate.assert_called_once_with(
            model, x, y, cv=2, scoring="accuracy", return_train_score=True
        )
        
        self.assertEqual(result["train_score"], [0.8, 0.9])
        self.assertEqual(result["test_score"], [0.7, 0.8])
        self.assertEqual(result["fit_time"], [0.1, 0.2])
        self.assertEqual(result["score_time"], [0.01, 0.02])


class TestBaseHyperparameterOptimizer(unittest.TestCase):
    """Tests for the BaseHyperparameterOptimizer class."""
    
    def test_init(self):
        """Test initialization."""
        optimizer = BaseHyperparameterOptimizer(name="TestOptimizer", description="Test description")
        self.assertEqual(optimizer.get_name(), "TestOptimizer")
        self.assertEqual(optimizer.get_description(), "Test description")
    
    @patch("nexusml.core.model_building.base.GridSearchCV")
    def test_optimize(self, mock_grid_search_cv):
        """Test optimize method."""
        optimizer = BaseHyperparameterOptimizer()
        model = MagicMock(spec=Pipeline)
        x_train = pd.DataFrame({"A": [1, 2, 3]})
        y_train = pd.DataFrame({"B": [4, 5, 6]})
        param_grid = {"param1": [1, 2, 3]}
        
        # Mock the GridSearchCV class
        mock_grid_search = MagicMock()
        mock_grid_search_cv.return_value = mock_grid_search
        mock_grid_search.best_params_ = {"param1": 2}
        mock_grid_search.best_score_ = 0.9
        mock_grid_search.best_estimator_ = model
        
        result = optimizer.optimize(model, x_train, y_train, param_grid=param_grid)
        
        mock_grid_search_cv.assert_called_once_with(
            model, param_grid=param_grid, cv=3, scoring="f1_macro", verbose=1
        )
        mock_grid_search.fit.assert_called_once_with(x_train, y_train)
        
        self.assertIs(result, model)
        self.assertEqual(optimizer.get_best_params(), {"param1": 2})
        self.assertEqual(optimizer.get_best_score(), 0.9)
    
    def test_get_best_params_not_optimized(self):
        """Test get_best_params method when not optimized."""
        optimizer = BaseHyperparameterOptimizer()
        with self.assertRaises(ValueError):
            optimizer.get_best_params()
    
    def test_get_best_score_not_optimized(self):
        """Test get_best_score method when not optimized."""
        optimizer = BaseHyperparameterOptimizer()
        with self.assertRaises(ValueError):
            optimizer.get_best_score()


class TestBaseModelEvaluator(unittest.TestCase):
    """Tests for the BaseModelEvaluator class."""
    
    def test_init(self):
        """Test initialization."""
        evaluator = BaseModelEvaluator(name="TestEvaluator", description="Test description")
        self.assertEqual(evaluator.get_name(), "TestEvaluator")
        self.assertEqual(evaluator.get_description(), "Test description")
    
    @patch("sklearn.metrics.accuracy_score")
    @patch("sklearn.metrics.f1_score")
    @patch("sklearn.metrics.classification_report")
    def test_evaluate(self, mock_classification_report, mock_f1_score, mock_accuracy_score):
        """Test evaluate method."""
        evaluator = BaseModelEvaluator()
        model = MagicMock(spec=Pipeline)
        x_test = pd.DataFrame({"A": [1, 2, 3]})
        y_test = pd.DataFrame({"B": [4, 5, 6], "C": [7, 8, 9]})
        
        # Mock the model.predict method
        model.predict.return_value = pd.DataFrame({"B": [4, 5, 6], "C": [7, 8, 9]})
        
        # Mock the metric functions
        mock_accuracy_score.return_value = 0.9
        mock_f1_score.return_value = 0.8
        mock_classification_report.return_value = "Classification Report"
        
        result = evaluator.evaluate(model, x_test, y_test)
        
        model.predict.assert_called_once_with(x_test)
        
        self.assertEqual(result["B"]["accuracy"], 0.9)
        self.assertEqual(result["B"]["f1_macro"], 0.8)
        self.assertEqual(result["B"]["classification_report"], "Classification Report")
        
        self.assertEqual(result["C"]["accuracy"], 0.9)
        self.assertEqual(result["C"]["f1_macro"], 0.8)
        self.assertEqual(result["C"]["classification_report"], "Classification Report")
        
        self.assertEqual(result["overall"]["accuracy_mean"], 0.9)
        self.assertEqual(result["overall"]["f1_macro_mean"], 0.8)
    
    def test_analyze_predictions(self):
        """Test analyze_predictions method."""
        evaluator = BaseModelEvaluator()
        model = MagicMock(spec=Pipeline)
        x_test = pd.DataFrame({"A": [1, 2, 3]})
        y_test = pd.DataFrame({"B": ["a", "b", "Other"]})
        y_pred = pd.DataFrame({"B": ["a", "Other", "Other"]})
        
        result = evaluator.analyze_predictions(model, x_test, y_test, y_pred)
        
        self.assertEqual(result["B"]["true_positives"], 1)
        self.assertEqual(result["B"]["false_positives"], 0)
        self.assertEqual(result["B"]["true_negatives"], 1)
        self.assertEqual(result["B"]["false_negatives"], 1)
        self.assertEqual(result["B"]["precision"], 1.0)
        self.assertEqual(result["B"]["recall"], 0.5)
        self.assertEqual(result["B"]["f1_score"], 2/3)
        
        self.assertEqual(result["B"]["other_category"]["accuracy"], 1.0)
        self.assertEqual(result["B"]["other_category"]["true_positives"], 1)
        self.assertEqual(result["B"]["other_category"]["false_positives"], 1)
        self.assertEqual(result["B"]["other_category"]["false_negatives"], 0)
        self.assertEqual(result["B"]["other_category"]["precision"], 0.5)
        self.assertEqual(result["B"]["other_category"]["recall"], 1.0)
        self.assertEqual(result["B"]["other_category"]["f1_score"], 2/3)


class TestBaseModelSerializer(unittest.TestCase):
    """Tests for the BaseModelSerializer class."""
    
    def test_init(self):
        """Test initialization."""
        serializer = BaseModelSerializer(name="TestSerializer", description="Test description")
        self.assertEqual(serializer.get_name(), "TestSerializer")
        self.assertEqual(serializer.get_description(), "Test description")
    
    @patch("pickle.dump")
    @patch("os.makedirs")
    def test_save_model(self, mock_makedirs, mock_dump):
        """Test save_model method."""
        serializer = BaseModelSerializer()
        model = MagicMock(spec=Pipeline)
        path = "test_model.pkl"
        
        serializer.save_model(model, path)
        
        mock_makedirs.assert_called_once_with(os.path.dirname(path), exist_ok=True)
        mock_dump.assert_called_once()
    
    @patch("pickle.load")
    @patch("os.path.exists")
    def test_load_model(self, mock_exists, mock_load):
        """Test load_model method."""
        serializer = BaseModelSerializer()
        model = MagicMock(spec=Pipeline)
        path = "test_model.pkl"
        
        # Mock the os.path.exists function
        mock_exists.return_value = True
        
        # Mock the pickle.load function
        mock_load.return_value = model
        
        result = serializer.load_model(path)
        
        mock_exists.assert_called_once_with(path)
        mock_load.assert_called_once()
        
        self.assertIs(result, model)
    
    def test_load_model_not_found(self):
        """Test load_model method when file not found."""
        serializer = BaseModelSerializer()
        path = "nonexistent_model.pkl"
        
        # Ensure the file doesn't exist
        import os
        if os.path.exists(path):
            os.remove(path)
        
        # Test that loading a nonexistent file raises FileNotFoundError
        with self.assertRaises(IOError):
            serializer.load_model(path)


class TestRandomForestBuilder(unittest.TestCase):
    """Tests for the RandomForestBuilder class."""
    
    def test_init(self):
        """Test initialization."""
        builder = RandomForestBuilder(name="TestBuilder", description="Test description")
        self.assertEqual(builder.get_name(), "TestBuilder")
        self.assertEqual(builder.get_description(), "Test description")
    
    def test_get_default_parameters(self):
        """Test get_default_parameters method."""
        builder = RandomForestBuilder()
        params = builder.get_default_parameters()
        
        self.assertIn("tfidf", params)
        self.assertIn("random_forest", params)
        
        self.assertIn("max_features", params["tfidf"])
        self.assertIn("ngram_range", params["tfidf"])
        
        self.assertIn("n_estimators", params["random_forest"])
        self.assertIn("class_weight", params["random_forest"])
    
    def test_get_param_grid(self):
        """Test get_param_grid method."""
        builder = RandomForestBuilder()
        param_grid = builder.get_param_grid()
        
        self.assertIn("preprocessor__text__tfidf__max_features", param_grid)
        self.assertIn("preprocessor__text__tfidf__ngram_range", param_grid)
        self.assertIn("clf__estimator__n_estimators", param_grid)
        self.assertIn("clf__estimator__min_samples_leaf", param_grid)
    
    def test_validate_config(self):
        """Test validate_config method."""
        builder = RandomForestBuilder()
        
        # Valid config
        config = {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
            },
            "random_forest": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
                "random_state": 42,
            },
        }
        
        self.assertTrue(builder.validate_config(config))
        
        # Invalid config - missing tfidf section
        invalid_config1 = {
            "random_forest": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
                "random_state": 42,
            },
        }
        
        self.assertFalse(builder.validate_config(invalid_config1))
        
        # Invalid config - missing random_forest section
        invalid_config2 = {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
            },
        }
        
        self.assertFalse(builder.validate_config(invalid_config2))
        
        # Invalid config - missing required parameter in tfidf section
        invalid_config3 = {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
            },
            "random_forest": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
                "random_state": 42,
            },
        }
        
        self.assertFalse(builder.validate_config(invalid_config3))
        
        # Invalid config - missing required parameter in random_forest section
        invalid_config4 = {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
            },
            "random_forest": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
            },
        }
        
        self.assertFalse(builder.validate_config(invalid_config4))
    
    def test_build_model(self):
        """Test build_model method."""
        builder = RandomForestBuilder()
        
        model = builder.build_model()
        
        self.assertIsInstance(model, Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], "preprocessor")
        self.assertEqual(model.steps[1][0], "clf")


if __name__ == "__main__":
    unittest.main()