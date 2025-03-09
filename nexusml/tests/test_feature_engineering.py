"""
Tests for the feature engineering components.

This module contains tests for the feature engineering components in the NexusML suite.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

from nexusml.core.feature_engineering import (
    # Base classes
    BaseFeatureTransformer,
    BaseColumnTransformer,
    BaseFeatureEngineer,
    
    # Transformers
    TextCombiner,
    NumericCleaner,
    HierarchyBuilder,
    ColumnMapper,
    
    # Config-driven feature engineer
    ConfigDrivenFeatureEngineer,
    
    # Registry
    register_transformer,
    create_transformer,
    get_transformer_class,
)


class TestBaseFeatureTransformer(unittest.TestCase):
    """Tests for the BaseFeatureTransformer class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = BaseFeatureTransformer(name="TestTransformer")
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertFalse(transformer._is_fitted)
    
    def test_fit(self):
        """Test fit method."""
        transformer = BaseFeatureTransformer()
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = transformer.fit(df)
        
        self.assertTrue(transformer._is_fitted)
        self.assertIs(result, transformer)
    
    def test_transform(self):
        """Test transform method."""
        transformer = BaseFeatureTransformer()
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # The base implementation should return a copy of the input
        self.assertIsNot(result, df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_transform_not_fitted(self):
        """Test transform method when not fitted."""
        transformer = BaseFeatureTransformer()
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            transformer.transform(df)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        transformer = BaseFeatureTransformer()
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = transformer.fit_transform(df)
        
        self.assertTrue(transformer._is_fitted)
        self.assertIsNot(result, df)
        pd.testing.assert_frame_equal(result, df)
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transformer = BaseFeatureTransformer()
        
        result = transformer.get_feature_names()
        
        self.assertEqual(result, [])


class TestBaseColumnTransformer(unittest.TestCase):
    """Tests for the BaseColumnTransformer class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = BaseColumnTransformer(
            input_columns=["A", "B"],
            output_columns=["C", "D"],
            name="TestTransformer"
        )
        
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.input_columns, ["A", "B"])
        self.assertEqual(transformer.output_columns, ["C", "D"])
        self.assertFalse(transformer._is_fitted)
    
    def test_init_default_output_columns(self):
        """Test initialization with default output columns."""
        transformer = BaseColumnTransformer(input_columns=["A", "B"])
        
        self.assertEqual(transformer.output_columns, ["A", "B"])
    
    def test_get_input_columns(self):
        """Test get_input_columns method."""
        transformer = BaseColumnTransformer(input_columns=["A", "B"])
        
        result = transformer.get_input_columns()
        
        self.assertEqual(result, ["A", "B"])
    
    def test_get_output_columns(self):
        """Test get_output_columns method."""
        transformer = BaseColumnTransformer(
            input_columns=["A", "B"],
            output_columns=["C", "D"]
        )
        
        result = transformer.get_output_columns()
        
        self.assertEqual(result, ["C", "D"])
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transformer = BaseColumnTransformer(
            input_columns=["A", "B"],
            output_columns=["C", "D"]
        )
        
        result = transformer.get_feature_names()
        
        self.assertEqual(result, ["C", "D"])
    
    def test_transform(self):
        """Test transform method."""
        # Create a subclass that implements _transform
        class TestTransformer(BaseColumnTransformer):
            def _transform(self, X):
                X = X.copy()
                X["C"] = X["A"] + X["B"]
                return X
        
        transformer = TestTransformer(input_columns=["A", "B"], output_columns=["C"])
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), [5, 7, 9])
    
    def test_transform_missing_columns(self):
        """Test transform method with missing columns."""
        transformer = BaseColumnTransformer(input_columns=["A", "B"])
        df = pd.DataFrame({"C": [1, 2, 3]})
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        with self.assertRaises(ValueError):
            transformer.transform(df)
    
    def test_handle_missing_columns(self):
        """Test _handle_missing_columns method."""
        # Create a subclass that implements _handle_missing_columns
        class TestTransformer(BaseColumnTransformer):
            def _handle_missing_columns(self, X, missing_columns):
                X = X.copy()
                X["C"] = 0
                return X
        
        transformer = TestTransformer(input_columns=["A", "B"], output_columns=["C"])
        df = pd.DataFrame({"D": [1, 2, 3]})
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), [0, 0, 0])


class TestTextCombiner(unittest.TestCase):
    """Tests for the TextCombiner class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = TextCombiner(
            columns=["A", "B"],
            separator=" - ",
            new_column="C",
            name="TestTransformer"
        )
        
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.input_columns, ["A", "B"])
        self.assertEqual(transformer.output_columns, ["C"])
        self.assertEqual(transformer.separator, " - ")
        self.assertEqual(transformer.new_column, "C")
    
    def test_transform(self):
        """Test transform method."""
        transformer = TextCombiner(
            columns=["A", "B"],
            separator=" - ",
            new_column="C"
        )
        
        df = pd.DataFrame({
            "A": ["a", "b", "c"],
            "B": ["d", "e", "f"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["a - d", "b - e", "c - f"])
    
    def test_transform_missing_columns(self):
        """Test transform method with missing columns."""
        transformer = TextCombiner(
            columns=["A", "B"],
            separator=" - ",
            new_column="C"
        )
        
        df = pd.DataFrame({
            "A": ["a", "b", "c"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["a", "b", "c"])
    
    def test_transform_all_missing_columns(self):
        """Test transform method with all columns missing."""
        transformer = TextCombiner(
            columns=["A", "B"],
            separator=" - ",
            new_column="C"
        )
        
        df = pd.DataFrame({
            "D": ["a", "b", "c"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["", "", ""])


class TestNumericCleaner(unittest.TestCase):
    """Tests for the NumericCleaner class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = NumericCleaner(
            column="A",
            new_name="B",
            fill_value=0,
            dtype="float",
            name="TestTransformer"
        )
        
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.input_columns, ["A"])
        self.assertEqual(transformer.output_columns, ["B"])
        self.assertEqual(transformer.column, "A")
        self.assertEqual(transformer.new_name, "B")
        self.assertEqual(transformer.fill_value, 0)
        self.assertEqual(transformer.dtype, "float")
    
    def test_transform_float(self):
        """Test transform method with float dtype."""
        transformer = NumericCleaner(
            column="A",
            new_name="B",
            fill_value=0,
            dtype="float"
        )
        
        df = pd.DataFrame({
            "A": [1, 2, None, "4"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["B"].tolist(), [1.0, 2.0, 0.0, 4.0])
    
    def test_transform_int(self):
        """Test transform method with int dtype."""
        transformer = NumericCleaner(
            column="A",
            new_name="B",
            fill_value=0,
            dtype="int"
        )
        
        df = pd.DataFrame({
            "A": [1.5, 2.7, None, "4.2"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["B"].tolist(), [1, 2, 0, 4])
    
    def test_transform_missing_column(self):
        """Test transform method with missing column."""
        transformer = NumericCleaner(
            column="A",
            new_name="B",
            fill_value=0,
            dtype="float"
        )
        
        df = pd.DataFrame({
            "C": [1, 2, 3]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["B"].tolist(), [0.0, 0.0, 0.0])


class TestHierarchyBuilder(unittest.TestCase):
    """Tests for the HierarchyBuilder class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = HierarchyBuilder(
            parent_columns=["A", "B"],
            new_column="C",
            separator="/",
            name="TestTransformer"
        )
        
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.input_columns, ["A", "B"])
        self.assertEqual(transformer.output_columns, ["C"])
        self.assertEqual(transformer.parent_columns, ["A", "B"])
        self.assertEqual(transformer.new_column, "C")
        self.assertEqual(transformer.separator, "/")
    
    def test_transform(self):
        """Test transform method."""
        transformer = HierarchyBuilder(
            parent_columns=["A", "B"],
            new_column="C",
            separator="/"
        )
        
        df = pd.DataFrame({
            "A": ["a", "b", "c"],
            "B": ["d", "e", "f"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["a/d", "b/e", "c/f"])
    
    def test_transform_missing_columns(self):
        """Test transform method with missing columns."""
        transformer = HierarchyBuilder(
            parent_columns=["A", "B"],
            new_column="C",
            separator="/"
        )
        
        df = pd.DataFrame({
            "A": ["a", "b", "c"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["a", "b", "c"])
    
    def test_transform_all_missing_columns(self):
        """Test transform method with all columns missing."""
        transformer = HierarchyBuilder(
            parent_columns=["A", "B"],
            new_column="C",
            separator="/"
        )
        
        df = pd.DataFrame({
            "D": ["a", "b", "c"]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["C"].tolist(), ["", "", ""])


class TestColumnMapper(unittest.TestCase):
    """Tests for the ColumnMapper class."""
    
    def test_init(self):
        """Test initialization."""
        transformer = ColumnMapper(
            mappings=[
                {"source": "A", "target": "B"},
                {"source": "C", "target": "D"}
            ],
            name="TestTransformer"
        )
        
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.input_columns, ["A", "C"])
        self.assertEqual(transformer.output_columns, ["B", "D"])
        self.assertEqual(transformer.mappings, [
            {"source": "A", "target": "B"},
            {"source": "C", "target": "D"}
        ])
    
    def test_transform(self):
        """Test transform method."""
        transformer = ColumnMapper(
            mappings=[
                {"source": "A", "target": "B"},
                {"source": "C", "target": "D"}
            ]
        )
        
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "C": [4, 5, 6]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["B"].tolist(), [1, 2, 3])
        self.assertEqual(result["D"].tolist(), [4, 5, 6])
    
    def test_transform_missing_columns(self):
        """Test transform method with missing columns."""
        transformer = ColumnMapper(
            mappings=[
                {"source": "A", "target": "B"},
                {"source": "C", "target": "D"}
            ]
        )
        
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "E": [4, 5, 6]
        })
        
        # Fit the transformer
        transformer.fit(df)
        
        # Transform the data
        result = transformer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        self.assertEqual(result["B"].tolist(), [1, 2, 3])
        self.assertNotIn("D", result.columns)


class TestBaseFeatureEngineer(unittest.TestCase):
    """Tests for the BaseFeatureEngineer class."""
    
    def test_init(self):
        """Test initialization."""
        transformer1 = MagicMock()
        transformer2 = MagicMock()
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2],
            name="TestFeatureEngineer"
        )
        
        self.assertEqual(feature_engineer.name, "TestFeatureEngineer")
        self.assertEqual(feature_engineer.transformers, [transformer1, transformer2])
        self.assertFalse(feature_engineer._is_fitted)
    
    def test_fit(self):
        """Test fit method."""
        transformer1 = MagicMock()
        transformer1.fit_transform.return_value = pd.DataFrame({"A": [1, 2, 3]})
        
        transformer2 = MagicMock()
        transformer2.fit_transform.return_value = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2]
        )
        
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = feature_engineer.fit(df)
        
        self.assertTrue(feature_engineer._is_fitted)
        self.assertIs(result, feature_engineer)
        transformer1.fit_transform.assert_called_once()
        transformer2.fit_transform.assert_called_once()
    
    def test_transform(self):
        """Test transform method."""
        transformer1 = MagicMock()
        transformer1.transform.return_value = pd.DataFrame({"A": [1, 2, 3]})
        
        transformer2 = MagicMock()
        transformer2.transform.return_value = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2]
        )
        
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        # Fit the feature engineer
        feature_engineer._is_fitted = True
        
        # Transform the data
        result = feature_engineer.transform(df)
        
        # Check the result
        self.assertIsNot(result, df)
        transformer1.transform.assert_called_once()
        transformer2.transform.assert_called_once()
    
    def test_transform_not_fitted(self):
        """Test transform method when not fitted."""
        feature_engineer = BaseFeatureEngineer()
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        with self.assertRaises(ValueError):
            feature_engineer.transform(df)
    
    def test_fit_transform(self):
        """Test fit_transform method."""
        transformer1 = MagicMock()
        transformer1.fit_transform.return_value = pd.DataFrame({"A": [1, 2, 3]})
        
        transformer2 = MagicMock()
        transformer2.fit_transform.return_value = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2]
        )
        
        df = pd.DataFrame({"A": [1, 2, 3]})
        
        result = feature_engineer.fit_transform(df)
        
        self.assertTrue(feature_engineer._is_fitted)
        self.assertIsNot(result, df)
        transformer1.fit_transform.assert_called_once()
        transformer2.fit_transform.assert_called_once()
    
    def test_get_transformers(self):
        """Test get_transformers method."""
        transformer1 = MagicMock()
        transformer2 = MagicMock()
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2]
        )
        
        result = feature_engineer.get_transformers()
        
        self.assertEqual(result, [transformer1, transformer2])
    
    def test_add_transformer(self):
        """Test add_transformer method."""
        transformer1 = MagicMock()
        transformer2 = MagicMock()
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1]
        )
        
        feature_engineer.add_transformer(transformer2)
        
        self.assertEqual(feature_engineer.transformers, [transformer1, transformer2])
    
    def test_get_feature_names(self):
        """Test get_feature_names method."""
        transformer1 = MagicMock()
        transformer1.get_feature_names.return_value = ["A", "B"]
        
        transformer2 = MagicMock()
        transformer2.get_feature_names.return_value = ["C", "D"]
        
        feature_engineer = BaseFeatureEngineer(
            transformers=[transformer1, transformer2]
        )
        
        result = feature_engineer.get_feature_names()
        
        self.assertEqual(result, ["C", "D"])


class TestConfigDrivenFeatureEngineer(unittest.TestCase):
    """Tests for the ConfigDrivenFeatureEngineer class."""
    
    @patch("nexusml.core.feature_engineering.config_driven.create_transformer")
    def test_create_transformers_from_config(self, mock_create_transformer):
        """Test create_transformers_from_config method."""
        # Mock the create_transformer function
        mock_transformer1 = MagicMock()
        mock_transformer2 = MagicMock()
        mock_create_transformer.side_effect = [mock_transformer1, mock_transformer2]
        
        # Create a configuration
        config = {
            "transformers": [
                {
                    "type": "text_combiner",
                    "columns": ["A", "B"],
                    "separator": " - ",
                    "new_column": "C"
                },
                {
                    "type": "numeric_cleaner",
                    "column": "D",
                    "new_name": "E",
                    "fill_value": 0,
                    "dtype": "float"
                }
            ]
        }
        
        # Create a feature engineer
        feature_engineer = ConfigDrivenFeatureEngineer(config=config)
        
        # Check the transformers
        self.assertEqual(feature_engineer.transformers, [mock_transformer1, mock_transformer2])
        
        # Check the create_transformer calls
        mock_create_transformer.assert_any_call(
            "text_combiner",
            columns=["A", "B"],
            separator=" - ",
            new_column="C"
        )
        
        mock_create_transformer.assert_any_call(
            "numeric_cleaner",
            column="D",
            new_name="E",
            fill_value=0,
            dtype="float"
        )


class TestTransformerRegistry(unittest.TestCase):
    """Tests for the transformer registry."""
    
    def test_register_and_get_transformer(self):
        """Test registering and getting a transformer."""
        # Define a test transformer
        class TestTransformer(BaseFeatureTransformer):
            pass
        
        # Use a unique name for the transformer
        import uuid
        unique_name = f"test_transformer_{uuid.uuid4().hex}"
        
        # Register the transformer
        register_transformer(unique_name, TestTransformer)
        
        # Get the transformer class
        transformer_class = get_transformer_class(unique_name)
        
        # Check the result
        self.assertIs(transformer_class, TestTransformer)
    
    def test_create_transformer(self):
        """Test creating a transformer."""
        # Define a test transformer
        class TestTransformer(BaseFeatureTransformer):
            def __init__(self, name="TestTransformer", value=None):
                super().__init__(name)
                self.value = value
        
        # Use a unique name for the transformer
        import uuid
        unique_name = f"test_transformer_create_{uuid.uuid4().hex}"
        
        # Register the transformer
        register_transformer(unique_name, TestTransformer)
        
        # Create a transformer
        transformer = create_transformer(unique_name, value=42)
        
        # Check the result
        self.assertIsInstance(transformer, TestTransformer)
        self.assertEqual(transformer.name, "TestTransformer")
        self.assertEqual(transformer.value, 42)


if __name__ == "__main__":
    unittest.main()