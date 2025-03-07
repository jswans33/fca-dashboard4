"""
Feature Engineering Adapter Module

This module provides adapter classes that maintain backward compatibility
with the existing feature engineering code while delegating to the new
components that use the configuration system.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.components.feature_engineer import StandardFeatureEngineer

# Set up logging
logger = logging.getLogger(__name__)


class GenericFeatureEngineerAdapter:
    """
    Adapter for the GenericFeatureEngineer class.

    This adapter maintains backward compatibility with the existing
    GenericFeatureEngineer class while delegating to the new
    StandardFeatureEngineer that uses the configuration system.
    """

    def __init__(self, config_provider: Optional[ConfigurationProvider] = None):
        """
        Initialize the GenericFeatureEngineerAdapter.

        Args:
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        self._feature_engineer = StandardFeatureEngineer(
            name="GenericFeatureEngineerAdapter",
            description="Adapter for GenericFeatureEngineer",
            config_provider=config_provider,
        )
        logger.info("Initialized GenericFeatureEngineerAdapter")

    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature engineering with hierarchical structure and more granular categories.

        This method delegates to the StandardFeatureEngineer while maintaining
        the same API as the original enhance_features function.

        Args:
            df: Input dataframe with raw features.

        Returns:
            DataFrame with enhanced features.
        """
        try:
            logger.info(f"Enhancing features for DataFrame with shape: {df.shape}")

            # Create a copy of the DataFrame to avoid modifying the original
            result = df.copy()

            # Apply the original column mappings for backward compatibility
            self._apply_legacy_column_mappings(result)

            # Use the StandardFeatureEngineer to engineer features
            result = self._feature_engineer.engineer_features(result)

            logger.info(f"Features enhanced successfully. Output shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Error enhancing features: {str(e)}")
            # Fall back to the original implementation
            logger.warning("Falling back to legacy implementation")
            return self._legacy_enhance_features(df)

    def create_hierarchical_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hierarchical category structure to better handle "Other" categories.

        This method delegates to the StandardFeatureEngineer while maintaining
        the same API as the original create_hierarchical_categories function.

        Args:
            df: Input dataframe with basic features.

        Returns:
            DataFrame with hierarchical category features.
        """
        try:
            logger.info(
                f"Creating hierarchical categories for DataFrame with shape: {df.shape}"
            )

            # Create a copy of the DataFrame to avoid modifying the original
            result = df.copy()

            # Check if the required columns exist
            required_columns = [
                "Asset Category",
                "Equip Name ID",
                "Precon System",
                "Operations System",
            ]
            missing_columns = [
                col for col in required_columns if col not in result.columns
            ]

            if missing_columns:
                logger.warning(
                    f"Missing required columns for hierarchical categories: {missing_columns}. "
                    f"Falling back to legacy implementation."
                )
                return self._legacy_create_hierarchical_categories(df)

            # Use the StandardFeatureEngineer to create hierarchical categories
            # The HierarchyBuilder transformer should handle this
            result = self._feature_engineer.engineer_features(result)

            logger.info(
                f"Hierarchical categories created successfully. Output shape: {result.shape}"
            )
            return result
        except Exception as e:
            logger.error(f"Error creating hierarchical categories: {str(e)}")
            # Fall back to the original implementation
            logger.warning("Falling back to legacy implementation")
            return self._legacy_create_hierarchical_categories(df)

    def _apply_legacy_column_mappings(self, df: pd.DataFrame) -> None:
        """
        Apply the original column mappings for backward compatibility.

        Args:
            df: DataFrame to modify in-place.
        """
        try:
            # Extract primary classification columns
            if (
                "Asset Category" in df.columns
                and "Equipment_Category" not in df.columns
            ):
                df["Equipment_Category"] = df["Asset Category"]

            if "System Type ID" in df.columns and "Uniformat_Class" not in df.columns:
                df["Uniformat_Class"] = df["System Type ID"]

            if "Precon System" in df.columns and "System_Type" not in df.columns:
                df["System_Type"] = df["Precon System"]

            # Create subcategory field for more granular classification
            if (
                "Equip Name ID" in df.columns
                and "Equipment_Subcategory" not in df.columns
            ):
                df["Equipment_Subcategory"] = df["Equip Name ID"]

            # Add equipment size and unit as features
            if (
                "Equipment Size" in df.columns
                and "Unit" in df.columns
                and "size_feature" not in df.columns
            ):
                df["size_feature"] = (
                    df["Equipment Size"].astype(str) + " " + df["Unit"].astype(str)
                )

            # Add service life as a feature
            if "Service Life" in df.columns and "service_life" not in df.columns:
                df["service_life"] = df["Service Life"].fillna(0).astype(float)

            logger.debug("Applied legacy column mappings")
        except Exception as e:
            logger.error(f"Error applying legacy column mappings: {str(e)}")

    def _legacy_enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy implementation of enhance_features for fallback.

        Args:
            df: Input dataframe with raw features.

        Returns:
            DataFrame with enhanced features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()

        # Extract primary classification columns
        result["Equipment_Category"] = result["Asset Category"]
        result["Uniformat_Class"] = result["System Type ID"]
        result["System_Type"] = result["Precon System"]

        # Create subcategory field for more granular classification
        result["Equipment_Subcategory"] = result["Equip Name ID"]

        # Combine fields for rich text features
        result["combined_features"] = (
            result["Asset Category"]
            + " "
            + result["Equip Name ID"]
            + " "
            + result["Sub System Type"]
            + " "
            + result["Sub System ID"]
            + " "
            + result["Title"]
            + " "
            + result["Precon System"]
            + " "
            + result["Operations System"]
            + " "
            + result["Sub System Class"]
            + " "
            + result["Drawing Abbreviation"]
        )

        # Add equipment size and unit as features
        result["size_feature"] = (
            result["Equipment Size"].astype(str) + " " + result["Unit"].astype(str)
        )

        # Add service life as a feature
        result["service_life"] = result["Service Life"].fillna(0).astype(float)

        # Fill NaN values
        result["combined_features"] = result["combined_features"].fillna("")

        return result

    def _legacy_create_hierarchical_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Legacy implementation of create_hierarchical_categories for fallback.

        Args:
            df: Input dataframe with basic features.

        Returns:
            DataFrame with hierarchical category features.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        result = df.copy()

        # Create Equipment Type - a more detailed category than Equipment_Category
        result["Equipment_Type"] = (
            result["Asset Category"] + "-" + result["Equip Name ID"]
        )

        # Create System Subtype - a more detailed category than System_Type
        result["System_Subtype"] = (
            result["Precon System"] + "-" + result["Operations System"]
        )

        return result


def enhanced_masterformat_mapping_adapter(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None,
) -> str:
    """
    Adapter for the enhanced_masterformat_mapping function.

    This adapter maintains backward compatibility with the existing
    enhanced_masterformat_mapping function while delegating to the new
    configuration system.

    Args:
        uniformat_class: Uniformat classification.
        system_type: System type.
        equipment_category: Equipment category.
        equipment_subcategory: Equipment subcategory.

    Returns:
        MasterFormat classification code.
    """
    try:
        logger.debug(
            f"Mapping MasterFormat for uniformat_class={uniformat_class}, "
            f"system_type={system_type}, equipment_category={equipment_category}, "
            f"equipment_subcategory={equipment_subcategory}"
        )

        # Try to get configuration
        try:
            config_provider = ConfigurationProvider()
            config = config_provider.config

            # Try equipment-specific mapping first
            if equipment_subcategory and config.masterformat_equipment:
                equipment_mappings = config.masterformat_equipment.root
                if equipment_subcategory in equipment_mappings:
                    masterformat_code = equipment_mappings[equipment_subcategory]
                    logger.debug(
                        f"Mapped to MasterFormat code: {masterformat_code} (equipment-specific)"
                    )
                    return masterformat_code

            # Then try system-type mapping
            if config.masterformat_primary:
                system_mappings = config.masterformat_primary.root
                if uniformat_class in system_mappings:
                    uniformat_mappings = system_mappings[uniformat_class]
                    if system_type in uniformat_mappings:
                        masterformat_code = uniformat_mappings[system_type]
                        logger.debug(
                            f"Mapped to MasterFormat code: {masterformat_code} (system-type)"
                        )
                        return masterformat_code

            # Try fallback mappings
            fallbacks = {
                "H": "23 00 00",  # Heating, Ventilating, and Air Conditioning (HVAC)
                "P": "22 00 00",  # Plumbing
                "SM": "23 00 00",  # HVAC
                "R": "11 40 00",  # Foodservice Equipment (Refrigeration)
            }

            if uniformat_class in fallbacks:
                masterformat_code = fallbacks[uniformat_class]
                logger.debug(
                    f"Mapped to MasterFormat code: {masterformat_code} (fallback)"
                )
                return masterformat_code

            # No match found, return default
            logger.debug(f"No mapping found, returning default code: 00 00 00")
            return "00 00 00"

        except Exception as e:
            logger.error(f"Error accessing configuration: {str(e)}")
            # Fall back to the original implementation
            logger.warning(
                "Falling back to legacy implementation due to configuration error"
            )
            return _legacy_enhanced_masterformat_mapping(
                uniformat_class, system_type, equipment_category, equipment_subcategory
            )

    except Exception as e:
        logger.error(f"Error in enhanced_masterformat_mapping_adapter: {str(e)}")
        # Fall back to the original implementation
        logger.warning("Falling back to legacy implementation")
        return _legacy_enhanced_masterformat_mapping(
            uniformat_class, system_type, equipment_category, equipment_subcategory
        )


def _legacy_enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None,
) -> str:
    """
    Legacy implementation of enhanced_masterformat_mapping for fallback.

    Args:
        uniformat_class: Uniformat classification.
        system_type: System type.
        equipment_category: Equipment category.
        equipment_subcategory: Equipment subcategory.

    Returns:
        MasterFormat classification code.
    """
    # Primary mapping
    primary_mapping = {
        "H": {
            "Chiller Plant": "23 64 00",  # Commercial Water Chillers
            "Cooling Tower Plant": "23 65 00",  # Cooling Towers
            "Heating Water Boiler Plant": "23 52 00",  # Heating Boilers
            "Steam Boiler Plant": "23 52 33",  # Steam Heating Boilers
            "Air Handling Units": "23 73 00",  # Indoor Central-Station Air-Handling Units
        },
        "P": {
            "Domestic Water Plant": "22 11 00",  # Facility Water Distribution
            "Medical/Lab Gas Plant": "22 63 00",  # Gas Systems for Laboratory and Healthcare Facilities
            "Sanitary Equipment": "22 13 00",  # Facility Sanitary Sewerage
        },
        "SM": {
            "Air Handling Units": "23 74 00",  # Packaged Outdoor HVAC Equipment
            "SM Accessories": "23 33 00",  # Air Duct Accessories
            "SM Equipment": "23 30 00",  # HVAC Air Distribution
        },
    }

    # Secondary mapping for specific equipment types that were in "Other"
    equipment_specific_mapping = {
        "Heat Exchanger": "23 57 00",  # Heat Exchangers for HVAC
        "Water Softener": "22 31 00",  # Domestic Water Softeners
        "Humidifier": "23 84 13",  # Humidifiers
        "Radiant Panel": "23 83 16",  # Radiant-Heating Hydronic Piping
        "Make-up Air Unit": "23 74 23",  # Packaged Outdoor Heating-Only Makeup Air Units
        "Energy Recovery Ventilator": "23 72 00",  # Air-to-Air Energy Recovery Equipment
        "DI/RO Equipment": "22 31 16",  # Deionized-Water Piping
        "Bypass Filter Feeder": "23 25 00",  # HVAC Water Treatment
        "Grease Interceptor": "22 13 23",  # Sanitary Waste Interceptors
        "Heat Trace": "23 05 33",  # Heat Tracing for HVAC Piping
        "Dust Collector": "23 35 16",  # Engine Exhaust Systems
        "Venturi VAV Box": "23 36 00",  # Air Terminal Units
        "Water Treatment Controller": "23 25 13",  # Water Treatment for Closed-Loop Hydronic Systems
        "Polishing System": "23 25 00",  # HVAC Water Treatment
        "Ozone Generator": "22 67 00",  # Processed Water Systems for Laboratory and Healthcare Facilities
    }

    # Try equipment-specific mapping first
    if equipment_subcategory in equipment_specific_mapping:
        return equipment_specific_mapping[equipment_subcategory]

    # Then try primary mapping
    if (
        uniformat_class in primary_mapping
        and system_type in primary_mapping[uniformat_class]
    ):
        return primary_mapping[uniformat_class][system_type]

    # Refined fallback mappings by Uniformat class
    fallbacks = {
        "H": "23 00 00",  # Heating, Ventilating, and Air Conditioning (HVAC)
        "P": "22 00 00",  # Plumbing
        "SM": "23 00 00",  # HVAC
        "R": "11 40 00",  # Foodservice Equipment (Refrigeration)
    }

    return fallbacks.get(uniformat_class, "00 00 00")  # Return unknown if no match
