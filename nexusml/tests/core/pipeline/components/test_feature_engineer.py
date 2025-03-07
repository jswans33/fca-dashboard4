"""
Tests for the StandardFeatureEngineer component.

This module contains tests for the StandardFeatureEngineer component,
which is responsible for feature engineering in the NexusML pipeline.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.feature_engineer import StandardFeatureEngineer


class TestStandardFeatureEngineer(unittest.TestCase):
    """Tests for the StandardFeatureEngineer component."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)

        # Create a mock configuration
        self.mock_config = MagicMock()
        self.mock_config.feature_engineering = MagicMock()
        self.mock_config.feature_engineering.model_dump.return_value = {
            "text_combinations": [
                {
                    "name": "combined_features",
                    "columns": ["Asset Category", "Equip Name ID", "Title"],
                    "separator": " ",
                }
            ],
            "numeric_columns": [
                {
                    "name": "Service Life",
                    "new_name": "service_life",
                    "fill_value": 0,
                    "dtype": "float",
                }
            ],
            "hierarchies": [
                {
                    "new_col": "Equipment_Type",
                    "parents": ["Asset Category", "Equip Name ID"],
                    "separator": "-",
                }
            ],
            "column_mappings": [
                {"source": "System Type ID", "target": "Uniformat_Class"}
            ],
        }

        # Set the mock configuration as the return value for the config property
        self.mock_config_provider.config = self.mock_config

        # Create the feature engineer with the mock configuration provider
        self.feature_engineer = StandardFeatureEngineer(
            config_provider=self.mock_config_provider
        )

        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame(
            {
                "Asset Category": ["Chiller", "Boiler", "Pump"],
                "Equip Name ID": ["Centrifugal", "Hot Water", "Circulation"],
                "Title": ["Main Chiller", "Heating Boiler", "Condenser Pump"],
                "System Type ID": ["H", "H", "P"],
                "Service Life": [15.0, 20.0, np.nan],
            }
        )

    def test_initialization(self):
        """Test initialization of the StandardFeatureEngineer."""
        self.assertEqual(self.feature_engineer.get_name(), "StandardFeatureEngineer")
        self.assertEqual(
            self.feature_engineer.get_description(),
            "Standard feature engineer using unified configuration",
        )
        self.assertEqual(
            self.feature_engineer.config,
            self.mock_config.feature_engineering.model_dump(),
        )

    def test_engineer_features(self):
        """Test the engineer_features method."""
        # Mock the _build_pipeline method to return a simple pipeline
        with patch.object(
            self.feature_engineer, "_build_pipeline"
        ) as mock_build_pipeline:
            # Create a mock pipeline that adds a 'test_feature' column
            mock_pipeline = MagicMock()
            mock_pipeline.fit.return_value = mock_pipeline
            mock_pipeline.transform.return_value = self.sample_data.assign(
                test_feature="test"
            )
            mock_build_pipeline.return_value = mock_pipeline

            # Call engineer_features
            result = self.feature_engineer.engineer_features(self.sample_data)

            # Check that the pipeline was built and used
            mock_build_pipeline.assert_called_once()
            mock_pipeline.fit.assert_called_once_with(self.sample_data)
            mock_pipeline.transform.assert_called_once_with(self.sample_data)

            # Check that the result has the expected column
            self.assertIn("test_feature", result.columns)
            self.assertEqual(result["test_feature"].iloc[0], "test")

    def test_fit_transform(self):
        """Test the fit and transform methods separately."""
        # Mock the _build_pipeline method to return a simple pipeline
        with patch.object(
            self.feature_engineer, "_build_pipeline"
        ) as mock_build_pipeline:
            # Create a mock pipeline that adds a 'test_feature' column
            mock_pipeline = MagicMock()
            mock_pipeline.fit.return_value = mock_pipeline
            mock_pipeline.transform.return_value = self.sample_data.assign(
                test_feature="test"
            )
            mock_build_pipeline.return_value = mock_pipeline

            # Call fit
            result = self.feature_engineer.fit(self.sample_data)

            # Check that the pipeline was built and fitted
            mock_build_pipeline.assert_called_once()
            mock_pipeline.fit.assert_called_once_with(self.sample_data)

            # Check that fit returns self
            self.assertEqual(result, self.feature_engineer)

            # Call transform
            result = self.feature_engineer.transform(self.sample_data)

            # Check that the pipeline was used to transform
            mock_pipeline.transform.assert_called_once_with(self.sample_data)

            # Check that the result has the expected column
            self.assertIn("test_feature", result.columns)
            self.assertEqual(result["test_feature"].iloc[0], "test")

    def test_build_pipeline(self):
        """Test the _build_pipeline method."""
        # Call _build_pipeline
        pipeline = self.feature_engineer._build_pipeline()

        # Check that the pipeline has the expected steps
        self.assertGreater(len(pipeline.steps), 0)

        # Check that the pipeline can be fitted and used to transform
        # Create a copy of the data to avoid modifying the original
        data_copy = self.sample_data.copy()
        # Fit the pipeline first
        pipeline.fit(data_copy)
        # Then transform (this avoids the FutureWarning)
        result = pipeline.transform(data_copy)

        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
