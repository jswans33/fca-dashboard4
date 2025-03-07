"""
Feature Engineering Example

This example demonstrates how to use the new feature engineering components
in the NexusML suite. It shows both the new StandardFeatureEngineer and
the adapter for backward compatibility.
"""

import logging

import numpy as np
import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.adapters.feature_adapter import (
    GenericFeatureEngineerAdapter,
    enhanced_masterformat_mapping_adapter,
)
from nexusml.core.pipeline.components.feature_engineer import StandardFeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create a sample DataFrame for demonstration."""
    return pd.DataFrame(
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


def example_standard_feature_engineer():
    """Example using the StandardFeatureEngineer directly."""
    logger.info("Example: Using StandardFeatureEngineer directly")

    # Create a sample DataFrame
    data = create_sample_data()
    logger.info(f"Sample data shape: {data.shape}")

    # Create a StandardFeatureEngineer
    feature_engineer = StandardFeatureEngineer()
    logger.info(f"Created {feature_engineer.get_name()}")

    # Engineer features
    result = feature_engineer.engineer_features(data)
    logger.info(f"Engineered features. Result shape: {result.shape}")

    # Display the result
    logger.info("Result columns:")
    for col in result.columns:
        logger.info(f"  - {col}")

    # Display a sample row
    logger.info("Sample row:")
    for col, value in result.iloc[0].items():
        logger.info(f"  - {col}: {value}")

    return result


def example_adapter():
    """Example using the GenericFeatureEngineerAdapter for backward compatibility."""
    logger.info(
        "Example: Using GenericFeatureEngineerAdapter for backward compatibility"
    )

    # Create a sample DataFrame
    data = create_sample_data()
    logger.info(f"Sample data shape: {data.shape}")

    # Create a GenericFeatureEngineerAdapter
    adapter = GenericFeatureEngineerAdapter()
    logger.info("Created GenericFeatureEngineerAdapter")

    # Enhance features (backward-compatible API)
    result = adapter.enhance_features(data)
    logger.info(f"Enhanced features. Result shape: {result.shape}")

    # Create hierarchical categories (backward-compatible API)
    result = adapter.create_hierarchical_categories(result)
    logger.info(f"Created hierarchical categories. Result shape: {result.shape}")

    # Display the result
    logger.info("Result columns:")
    for col in result.columns:
        logger.info(f"  - {col}")

    # Display a sample row
    logger.info("Sample row:")
    for col, value in result.iloc[0].items():
        logger.info(f"  - {col}: {value}")

    return result


def example_masterformat_mapping():
    """Example using the enhanced_masterformat_mapping_adapter."""
    logger.info("Example: Using enhanced_masterformat_mapping_adapter")

    # Create a sample DataFrame
    data = create_sample_data()
    logger.info(f"Sample data shape: {data.shape}")

    # Map each row to MasterFormat
    masterformat_codes = []
    for _, row in data.iterrows():
        code = enhanced_masterformat_mapping_adapter(
            uniformat_class=row["System Type ID"],
            system_type=row["Precon System"],
            equipment_category=row["Asset Category"],
            equipment_subcategory=row["Equip Name ID"],
        )
        masterformat_codes.append(code)

    # Add the MasterFormat codes to the DataFrame
    data["MasterFormat"] = masterformat_codes
    logger.info("Added MasterFormat codes")

    # Display the result
    logger.info("MasterFormat codes:")
    for i, (_, row) in enumerate(data.iterrows()):
        logger.info(
            f"  - {row['Asset Category']} {row['Equip Name ID']} "
            f"({row['System Type ID']}, {row['Precon System']}): "
            f"{row['MasterFormat']}"
        )

    return data


def main():
    """Run the examples."""
    logger.info("Starting feature engineering examples")

    # Example using StandardFeatureEngineer directly
    standard_result = example_standard_feature_engineer()
    logger.info("-" * 80)

    # Example using GenericFeatureEngineerAdapter
    adapter_result = example_adapter()
    logger.info("-" * 80)

    # Example using enhanced_masterformat_mapping_adapter
    mapping_result = example_masterformat_mapping()
    logger.info("-" * 80)

    logger.info("Examples completed")


if __name__ == "__main__":
    main()
