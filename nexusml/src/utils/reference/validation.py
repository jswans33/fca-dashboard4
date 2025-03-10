"""
Reference Data Validation

This module provides validation functions for reference data sources.
"""

from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd

from nexusml.core.reference.base import ReferenceDataSource
from nexusml.core.reference.classification import ClassificationDataSource
from nexusml.core.reference.equipment import EquipmentTaxonomyDataSource
from nexusml.core.reference.glossary import GlossaryDataSource
from nexusml.core.reference.manufacturer import ManufacturerDataSource
from nexusml.core.reference.service_life import ServiceLifeDataSource

# Type alias for DataFrame to help with type checking
DataFrame = pd.DataFrame


def validate_classification_data(
    source: ClassificationDataSource, source_type: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate a classification data source.

    Args:
        source: Classification data source
        source_type: Type of classification (omniclass, uniformat, masterformat)
        config: Configuration dictionary

    Returns:
        Dictionary with validation results
    """
    result = {
        "loaded": source.data is not None,
        "issues": [],
        "stats": {},
    }

    if not result["loaded"]:
        result["issues"].append("Data not loaded")
        return result

    # Check if data is a DataFrame
    if not isinstance(source.data, pd.DataFrame):
        result["issues"].append(f"Expected DataFrame, got {type(source.data).__name__}")
        return result

    # Check required columns
    required_columns = ["code", "name", "description"]
    column_mappings = config.get("column_mappings", {}).get(source_type, {})
    missing_columns = [
        col for col in required_columns if col not in source.data.columns
    ]

    if missing_columns:
        result["issues"].append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    # Check for nulls in key columns
    for col in [c for c in required_columns if c in source.data.columns]:
        null_count = source.data[col].isna().sum()
        if null_count > 0:
            result["issues"].append(f"Column '{col}' has {null_count} null values")

    # Check for duplicates in code column
    if "code" in source.data.columns:
        duplicate_count = source.data["code"].duplicated().sum()
        if duplicate_count > 0:
            result["issues"].append(f"Found {duplicate_count} duplicate codes")

    # Add statistics
    try:
        # Cast source.data to DataFrame to help with type checking
        df = cast(DataFrame, source.data)

        result["stats"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }
    except Exception as e:
        result["issues"].append(f"Error calculating statistics: {e}")
        result["stats"] = {"error": str(e)}

    return result


def validate_glossary_data(source: GlossaryDataSource) -> Dict[str, Any]:
    """
    Validate a glossary data source.

    Args:
        source: Glossary data source

    Returns:
        Dictionary with validation results
    """
    result = {
        "loaded": source.data is not None,
        "issues": [],
        "stats": {},
    }

    if not result["loaded"]:
        result["issues"].append("Data not loaded")
        return result

    # Check if data is a dictionary
    if not isinstance(source.data, dict):
        result["issues"].append(
            f"Expected dictionary, got {type(source.data).__name__}"
        )
        return result

    # Check for empty values
    empty_values = 0
    data_dict = source.data if isinstance(source.data, dict) else {}

    for v in data_dict.values():
        if not v:
            empty_values += 1

    if empty_values > 0:
        result["issues"].append(f"Found {empty_values} empty definitions")

    # Add statistics
    data_len = len(data_dict)
    total_key_length = 0
    total_value_length = 0

    for k, v in data_dict.items():
        total_key_length += len(k)
        total_value_length += len(str(v))

    result["stats"] = {
        "entry_count": data_len,
        "avg_key_length": total_key_length / max(1, data_len),
        "avg_value_length": total_value_length / max(1, data_len),
    }

    return result


def validate_manufacturer_data(source: ManufacturerDataSource) -> Dict[str, Any]:
    """
    Validate a manufacturer data source.

    Args:
        source: Manufacturer data source

    Returns:
        Dictionary with validation results
    """
    result = {
        "loaded": source.data is not None,
        "issues": [],
        "stats": {},
    }

    if not result["loaded"]:
        result["issues"].append("Data not loaded")
        return result

    # Check if data is a list
    if not isinstance(source.data, list):
        result["issues"].append(f"Expected list, got {type(source.data).__name__}")
        return result

    # Check required fields in each manufacturer entry
    required_fields = ["name", "products"]
    missing_fields = {}
    empty_products = 0
    total_products = 0
    valid_entries = 0

    for i, manuf in enumerate(source.data):
        if not isinstance(manuf, dict):
            result["issues"].append(f"Entry {i} is not a dictionary")
            continue

        valid_entries += 1

        for field in required_fields:
            if field not in manuf:
                missing_fields[field] = missing_fields.get(field, 0) + 1

        # Check for empty product lists
        if isinstance(manuf.get("products"), list):
            if not manuf["products"]:
                empty_products += 1
            total_products += len(manuf["products"])

    for field, count in missing_fields.items():
        result["issues"].append(f"Field '{field}' missing in {count} entries")

    if empty_products > 0:
        result["issues"].append(
            f"Found {empty_products} entries with empty product lists"
        )

    # Add statistics
    data_len = len(source.data)
    result["stats"] = {
        "manufacturer_count": data_len,
        "valid_entries": valid_entries,
        "avg_products_per_manufacturer": total_products / max(1, valid_entries),
    }

    return result


def validate_service_life_data(source: ServiceLifeDataSource) -> Dict[str, Any]:
    """
    Validate a service life data source.

    Args:
        source: Service life data source

    Returns:
        Dictionary with validation results
    """
    result = {
        "loaded": source.data is not None,
        "issues": [],
        "stats": {},
    }

    if not result["loaded"]:
        result["issues"].append("Data not loaded")
        return result

    # Check if data is a DataFrame
    if not isinstance(source.data, pd.DataFrame):
        result["issues"].append(f"Expected DataFrame, got {type(source.data).__name__}")
        return result

    # Check required columns
    column_mappings = source.column_mappings
    required_columns = ["equipment_type", "median_years"]

    # Map internal column names to actual DataFrame columns
    required_df_columns = []

    # Ensure column_mappings is a dictionary
    if isinstance(column_mappings, dict) and column_mappings:
        for col in required_columns:
            # Use a safer approach to find the mapped column
            mapped_col = col
            for k, v in column_mappings.items():
                if v == col:
                    mapped_col = k
                    break
            required_df_columns.append(mapped_col)
    else:
        # If column_mappings is not a dictionary, use the original column names
        required_df_columns = required_columns

    missing_columns = [
        col for col in required_df_columns if col not in source.data.columns
    ]

    if missing_columns:
        result["issues"].append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    # Check for nulls in key columns
    for col in [c for c in required_df_columns if c in source.data.columns]:
        null_count = source.data[col].isna().sum()
        if null_count > 0:
            result["issues"].append(f"Column '{col}' has {null_count} null values")

    # Check for negative service life values
    try:
        # Cast source.data to DataFrame to help with type checking
        df = cast(DataFrame, source.data)

        # Get column names as a list to avoid iteration issues
        column_names = list(df.columns)

        # Find columns with 'year' in the name
        year_columns = []
        for c in column_names:
            if isinstance(c, str) and "year" in c.lower():
                year_columns.append(c)

        for col in year_columns:
            try:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    result["issues"].append(
                        f"Column '{col}' has {neg_count} negative values"
                    )
            except Exception as e:
                result["issues"].append(
                    f"Error checking negative values in column '{col}': {e}"
                )
    except Exception as e:
        result["issues"].append(f"Error checking for negative service life values: {e}")

    # Add statistics
    try:
        # Cast source.data to DataFrame to help with type checking
        df = cast(DataFrame, source.data)

        result["stats"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }

        if "median_years" in df.columns:
            result["stats"]["avg_service_life"] = df["median_years"].mean()
            result["stats"]["min_service_life"] = df["median_years"].min()
            result["stats"]["max_service_life"] = df["median_years"].max()
    except Exception as e:
        result["issues"].append(f"Error calculating statistics: {e}")
        result["stats"] = {"error": str(e)}

    return result


def validate_equipment_taxonomy_data(
    source: EquipmentTaxonomyDataSource,
) -> Dict[str, Any]:
    """
    Validate an equipment taxonomy data source.

    Args:
        source: Equipment taxonomy data source

    Returns:
        Dictionary with validation results
    """
    result = {
        "loaded": source.data is not None,
        "issues": [],
        "stats": {},
    }

    if not result["loaded"]:
        result["issues"].append("Data not loaded")
        return result

    # Check if data is a DataFrame
    if not isinstance(source.data, pd.DataFrame):
        result["issues"].append(f"Expected DataFrame, got {type(source.data).__name__}")
        return result

    # Print actual column names for debugging
    print("Equipment taxonomy columns:", list(source.data.columns))

    # Handle BOM character in first column name
    if source.data.columns[0].startswith("\ufeff"):
        # Create a copy of the DataFrame with fixed column names
        fixed_columns = list(source.data.columns)
        fixed_columns[0] = fixed_columns[0].replace("\ufeff", "")
        source.data.columns = fixed_columns
        print("Fixed first column name, new columns:", list(source.data.columns))

    # Check required columns based on actual CSV columns
    required_columns = [
        "Asset Category",
        "equipment_id",
        "trade",
        "title",
        "drawing_abbreviation",
        "service_life",
    ]

    # Case-insensitive column check
    available_columns = [col.lower() for col in source.data.columns]
    missing_columns = [
        col
        for col in required_columns
        if col.lower() not in available_columns
        and col.replace(" ", "_").lower() not in available_columns
    ]

    if missing_columns:
        result["issues"].append(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    # Check for nulls in key columns - case-insensitive
    for req_col in required_columns:
        # Find the actual column name in the DataFrame (case-insensitive)
        actual_col = None
        for df_col in source.data.columns:
            if (
                df_col.lower() == req_col.lower()
                or df_col.lower() == req_col.replace(" ", "_").lower()
            ):
                actual_col = df_col
                break

        if actual_col:
            null_count = source.data[actual_col].isna().sum()
            if null_count > 0:
                result["issues"].append(
                    f"Column '{actual_col}' has {null_count} null values"
                )

    # Check for negative service life values - case-insensitive
    try:
        # Cast source.data to DataFrame to help with type checking
        df = cast(DataFrame, source.data)

        # Use the actual column name from the CSV
        service_life_col = "service_life"

        if service_life_col:
            try:
                # Convert to numeric, coercing errors to NaN
                service_life = pd.to_numeric(df[service_life_col], errors="coerce")

                # Check for negative values
                neg_count = (service_life < 0).sum()
                if neg_count > 0:
                    result["issues"].append(
                        f"Column '{service_life_col}' has {neg_count} negative values"
                    )

                # Check for non-numeric values
                non_numeric = df[service_life_col].isna() != service_life.isna()
                non_numeric_count = non_numeric.sum()
                if non_numeric_count > 0:
                    result["issues"].append(
                        f"Column '{service_life_col}' has {non_numeric_count} non-numeric values"
                    )
            except Exception as e:
                result["issues"].append(
                    f"Error checking '{service_life_col}' column: {e}"
                )
    except Exception as e:
        result["issues"].append(f"Error validating service life values: {e}")

    # Add statistics
    try:
        # Cast source.data to DataFrame to help with type checking
        df = cast(DataFrame, source.data)

        result["stats"] = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
        }

        # Use the actual column names from the CSV
        category_col = "Asset Category"
        title_col = "title"
        service_life_col = "service_life"

        # Count unique categories
        if category_col:
            result["stats"]["category_count"] = df[category_col].nunique()

        # Count unique equipment types
        if title_col:
            result["stats"]["equipment_type_count"] = df[title_col].nunique()

        # Calculate average service life if available
        if service_life_col:
            try:
                service_life = pd.to_numeric(df[service_life_col], errors="coerce")
                result["stats"]["avg_service_life"] = service_life.mean()
                result["stats"]["min_service_life"] = service_life.min()
                result["stats"]["max_service_life"] = service_life.max()
            except Exception as e:
                result["issues"].append(
                    f"Error calculating service life statistics: {e}"
                )
    except Exception as e:
        result["issues"].append(f"Error calculating statistics: {e}")
        result["stats"] = {"error": str(e)}

    return result
