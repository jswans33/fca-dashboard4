"""
Tests for the feature engineering adapters.

This module contains tests for the feature engineering adapters,
which maintain backward compatibility with the existing code.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.adapters.feature_adapter import (
    GenericFeatureEngineerAdapter,
    enhanced_masterformat_mapping_adapter,
)


class TestGenericFeatureEngineerAdapter(unittest.TestCase):
    """Tests for the GenericFeatureEngineerAdapter."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)

        # Create a mock configuration
        self.mock_config = MagicMock()
        self.mock_config_provider.config = self.mock_config

        # Create the adapter with the mock configuration provider
        with patch(
            "nexusml.core.pipeline.adapters.feature_adapter.StandardFeatureEngineer"
        ) as mock_feature_engineer_class:
            # Create a mock StandardFeatureEngineer
            self.mock_feature_engineer = MagicMock()
            mock_feature_engineer_class.return_value = self.mock_feature_engineer

            # Create the adapter
            self.adapter = GenericFeatureEngineerAdapter(
                config_provider=self.mock_config_provider
            )

        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame(
            {
                "Asset Category": ["Chiller", "Boiler", "Pump"],
                "Equip Name ID": ["Centrifugal", "Hot Water", "Circulation"],
                "Title": ["Main Chiller", "Heating Boiler", "Condenser Pump"],
                "System Type ID": ["H", "H", "P"],
                "Precon System": [
                    "Chiller Plant",
                    "Heating Water Boiler Plant",
                    "Domestic Water Plant",
                ],
                "Operations System": ["CHW", "HHW", "DHW"],
                "Sub System Type": ["Primary", "Secondary", "Tertiary"],
                "Sub System ID": ["P", "S", "T"],
                "Sub System Class": ["Class 1", "Class 2", "Class 3"],
                "Drawing Abbreviation": ["CH", "BL", "PU"],
                "Equipment Size": [500, 1000, 100],
                "Unit": ["tons", "MBH", "GPM"],
                "Service Life": [15.0, 20.0, np.nan],
            }
        )

    def test_enhance_features(self):
        """Test the enhance_features method."""
        # Create a modified DataFrame that includes both the enhanced column
        # and the legacy column mappings
        enhanced_data = self.sample_data.copy()
        enhanced_data["enhanced"] = True
        enhanced_data["Equipment_Category"] = enhanced_data["Asset Category"]
        enhanced_data["Uniformat_Class"] = enhanced_data["System Type ID"]
        enhanced_data["System_Type"] = enhanced_data["Precon System"]
        enhanced_data["Equipment_Subcategory"] = enhanced_data["Equip Name ID"]

        # Set up the mock feature engineer to return the modified DataFrame
        self.mock_feature_engineer.engineer_features.return_value = enhanced_data

        # Call enhance_features
        result = self.adapter.enhance_features(self.sample_data)

        # Check that the feature engineer was used
        self.mock_feature_engineer.engineer_features.assert_called_once()

        # Check that the result has the expected column
        self.assertIn("enhanced", result.columns)
        self.assertTrue(result["enhanced"].iloc[0])

        # Check that the legacy column mappings were applied
        self.assertIn("Equipment_Category", result.columns)
        self.assertIn("Uniformat_Class", result.columns)
        self.assertIn("System_Type", result.columns)
        self.assertIn("Equipment_Subcategory", result.columns)

    def test_enhance_features_fallback(self):
        """Test the enhance_features method with fallback to legacy implementation."""
        # Set up the mock feature engineer to raise an exception
        self.mock_feature_engineer.engineer_features.side_effect = ValueError(
            "Test error"
        )

        # Call enhance_features
        result = self.adapter.enhance_features(self.sample_data)

        # Check that the feature engineer was used
        self.mock_feature_engineer.engineer_features.assert_called_once()

        # Check that the legacy implementation was used as fallback
        self.assertIn("combined_features", result.columns)
        self.assertIn("size_feature", result.columns)
        self.assertIn("service_life", result.columns)

        # Check that the combined_features column has the expected format
        self.assertTrue(
            result["combined_features"].iloc[0].startswith("Chiller Centrifugal")
        )

    def test_create_hierarchical_categories(self):
        """Test the create_hierarchical_categories method."""
        # Set up the mock feature engineer to return a modified DataFrame
        hierarchical_data = self.sample_data.copy()
        hierarchical_data["Equipment_Type"] = (
            hierarchical_data["Asset Category"]
            + "-"
            + hierarchical_data["Equip Name ID"]
        )
        hierarchical_data["System_Subtype"] = (
            hierarchical_data["Precon System"]
            + "-"
            + hierarchical_data["Operations System"]
        )
        self.mock_feature_engineer.engineer_features.return_value = hierarchical_data

        # Call create_hierarchical_categories
        result = self.adapter.create_hierarchical_categories(self.sample_data)

        # Check that the feature engineer was used
        self.mock_feature_engineer.engineer_features.assert_called_once()

        # Check that the result has the expected columns
        self.assertIn("Equipment_Type", result.columns)
        self.assertIn("System_Subtype", result.columns)

        # Check that the hierarchical columns have the expected format
        self.assertEqual(result["Equipment_Type"].iloc[0], "Chiller-Centrifugal")
        self.assertEqual(result["System_Subtype"].iloc[0], "Chiller Plant-CHW")

    def test_create_hierarchical_categories_fallback(self):
        """Test the create_hierarchical_categories method with fallback to legacy implementation."""
        # Set up the mock feature engineer to raise an exception
        self.mock_feature_engineer.engineer_features.side_effect = ValueError(
            "Test error"
        )

        # Call create_hierarchical_categories
        result = self.adapter.create_hierarchical_categories(self.sample_data)

        # Check that the feature engineer was used
        self.mock_feature_engineer.engineer_features.assert_called_once()

        # Check that the legacy implementation was used as fallback
        self.assertIn("Equipment_Type", result.columns)
        self.assertIn("System_Subtype", result.columns)

        # Check that the hierarchical columns have the expected format
        self.assertEqual(result["Equipment_Type"].iloc[0], "Chiller-Centrifugal")
        self.assertEqual(result["System_Subtype"].iloc[0], "Chiller Plant-CHW")


class TestEnhancedMasterformatMappingAdapter(unittest.TestCase):
    """Tests for the enhanced_masterformat_mapping_adapter function."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a patch for the ConfigurationProvider
        self.config_provider_patch = patch(
            "nexusml.core.pipeline.adapters.feature_adapter.ConfigurationProvider"
        )
        self.mock_config_provider_class = self.config_provider_patch.start()

        # Create a mock configuration provider
        self.mock_config_provider = MagicMock(spec=ConfigurationProvider)
        self.mock_config_provider_class.return_value = self.mock_config_provider

        # Create a mock configuration
        self.mock_config = MagicMock()
        self.mock_config_provider.config = self.mock_config

        # Set up mock masterformat mappings
        self.mock_equipment_mappings = MagicMock()
        self.mock_equipment_mappings.root = {
            "Centrifugal": "23 64 16",  # Centrifugal Water Chillers
            "Hot Water": "23 52 23",  # Cast-Iron Boilers
            "Circulation": "22 11 23",  # Domestic Water Pumps
        }

        self.mock_system_mappings = MagicMock()
        self.mock_system_mappings.root = {
            "H": {
                "Chiller Plant": "23 64 00",  # Commercial Water Chillers
                "Heating Water Boiler Plant": "23 52 00",  # Heating Boilers
            },
            "P": {
                "Domestic Water Plant": "22 11 00",  # Facility Water Distribution
            },
        }

        # Set the mock mappings on the mock configuration
        self.mock_config.masterformat_equipment = self.mock_equipment_mappings
        self.mock_config.masterformat_primary = self.mock_system_mappings

    def tearDown(self):
        """Tear down test fixtures."""
        self.config_provider_patch.stop()

    def test_equipment_specific_mapping(self):
        """Test mapping with equipment-specific mapping."""
        # Call the adapter with equipment subcategory
        result = enhanced_masterformat_mapping_adapter(
            uniformat_class="H",
            system_type="Chiller Plant",
            equipment_category="Chiller",
            equipment_subcategory="Centrifugal",
        )

        # Check that the result is the expected code
        self.assertEqual(result, "23 64 16")

    def test_system_type_mapping(self):
        """Test mapping with system-type mapping."""
        # Call the adapter with system type
        result = enhanced_masterformat_mapping_adapter(
            uniformat_class="H",
            system_type="Chiller Plant",
            equipment_category="Chiller",
            equipment_subcategory="Unknown",
        )

        # Check that the result is the expected code
        self.assertEqual(result, "23 64 00")

    def test_fallback_mapping(self):
        """Test mapping with fallback mapping."""
        # Call the adapter with unknown system type
        result = enhanced_masterformat_mapping_adapter(
            uniformat_class="H",
            system_type="Unknown",
            equipment_category="Chiller",
            equipment_subcategory="Unknown",
        )

        # Check that the result is the expected code
        self.assertEqual(result, "23 00 00")

    def test_default_mapping(self):
        """Test mapping with default mapping."""
        # Call the adapter with unknown uniformat class
        result = enhanced_masterformat_mapping_adapter(
            uniformat_class="Unknown",
            system_type="Unknown",
            equipment_category="Unknown",
            equipment_subcategory="Unknown",
        )

        # Check that the result is the expected code
        self.assertEqual(result, "00 00 00")

    def test_configuration_error(self):
        """Test mapping with configuration error."""
        # Set up the mock configuration provider to raise an exception
        self.mock_config_provider.config = None

        # Call the adapter
        result = enhanced_masterformat_mapping_adapter(
            uniformat_class="H",
            system_type="Chiller Plant",
            equipment_category="Chiller",
            equipment_subcategory="Centrifugal",
        )

        # Check that the result is from the legacy implementation
        self.assertEqual(result, "23 64 00")


if __name__ == "__main__":
    unittest.main()
