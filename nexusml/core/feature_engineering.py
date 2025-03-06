"""
Feature Engineering Module

This module handles feature engineering for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on feature transformations.
"""

import json
from typing import Dict, Optional, Tuple

import pandas as pd

from nexusml.config import get_project_root


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with hierarchical structure and more granular categories

    Args:
        df (pd.DataFrame): Input dataframe with raw features

    Returns:
        pd.DataFrame: DataFrame with enhanced features
    """
    # Extract primary classification columns
    df["Equipment_Category"] = df["Asset Category"]
    df["Uniformat_Class"] = df["System Type ID"]
    df["System_Type"] = df["Precon System"]

    # Create subcategory field for more granular classification
    df["Equipment_Subcategory"] = df["Equip Name ID"]

    # Combine fields for rich text features
    df["combined_features"] = (
        df["Asset Category"]
        + " "
        + df["Equip Name ID"]
        + " "
        + df["Sub System Type"]
        + " "
        + df["Sub System ID"]
        + " "
        + df["Title"]
        + " "
        + df["Precon System"]
        + " "
        + df["Operations System"]
        + " "
        + df["Sub System Class"]
        + " "
        + df["Drawing Abbreviation"]
    )

    # Add equipment size and unit as features
    df["size_feature"] = df["Equipment Size"].astype(str) + " " + df["Unit"].astype(str)

    # Add service life as a feature
    df["service_life"] = df["Service Life"].fillna(0).astype(float)

    # Fill NaN values
    df["combined_features"] = df["combined_features"].fillna("")

    return df


def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical category structure to better handle "Other" categories

    Args:
        df (pd.DataFrame): Input dataframe with basic features

    Returns:
        pd.DataFrame: DataFrame with hierarchical category features
    """
    # Create Equipment Type - a more detailed category than Equipment_Category
    df["Equipment_Type"] = df["Asset Category"] + "-" + df["Equip Name ID"]

    # Create System Subtype - a more detailed category than System_Type
    df["System_Subtype"] = df["Precon System"] + "-" + df["Operations System"]

    return df


def load_masterformat_mappings() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Load MasterFormat mappings from JSON files.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Primary and equipment-specific mappings
    """
    root = get_project_root()

    try:
        with open(root / "config" / "mappings" / "masterformat_primary.json") as f:
            primary_mapping = json.load(f)

        with open(root / "config" / "mappings" / "masterformat_equipment.json") as f:
            equipment_specific_mapping = json.load(f)

        return primary_mapping, equipment_specific_mapping
    except Exception as e:
        print(f"Warning: Could not load MasterFormat mappings: {e}")
        # Return empty mappings if files cannot be loaded
        return {}, {}


def enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None,
) -> str:
    """
    Enhanced mapping with better handling of specialty equipment types

    Args:
        uniformat_class (str): Uniformat classification
        system_type (str): System type
        equipment_category (str): Equipment category
        equipment_subcategory (Optional[str]): Equipment subcategory

    Returns:
        str: MasterFormat classification code
    """
    # Load mappings from JSON files
    primary_mapping, equipment_specific_mapping = load_masterformat_mappings()

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
