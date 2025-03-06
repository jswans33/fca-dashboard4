"""
Data Mapper Module

This module handles mapping between staging data and the ML model input format.
"""

from typing import Any, Dict, List, Optional

import pandas as pd


class DataMapper:
    """
    Maps data between different formats, specifically from staging data to ML model input
    and from ML model output to master database fields.
    """

    def __init__(self, column_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize the data mapper with an optional column mapping.

        Args:
            column_mapping: Dictionary mapping staging columns to model input columns
        """
        # Default mapping from staging columns to model input columns
        self.column_mapping = column_mapping or {
            # Map fake data columns to model input columns
            "category_name": "category_name",
            "equipment_tag": "equipment_tag",
            "manufacturer": "manufacturer",
            "model": "model",
            "omniclass_code": "omniclass_code",
            "uniformat_code": "uniformat_code",
            "masterformat_code": "masterformat_code",
            "mcaa_system_category": "mcaa_system_category",
            "building_name": "building_name",
            "initial_cost": "initial_cost",
            "condition_score": "condition_score",
            "CategoryID": "CategoryID",
            "OmniClassID": "OmniClassID",
            "UniFormatID": "UniFormatID",
            "MasterFormatID": "MasterFormatID",
            "MCAAID": "MCAAID",
            "LocationID": "LocationID",
        }

        # Required fields with default values
        self.required_fields = {
            "category_name": "Unknown Equipment",
            "mcaa_system_category": "Unknown System",
            "equipment_tag": "UNKNOWN-TAG",
        }

        # Numeric fields with default values
        self.numeric_fields = {"condition_score": 3.0, "initial_cost": 0.0}

    def map_staging_to_model_input(self, staging_df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps staging data columns to the format expected by the ML model.

        Args:
            staging_df: DataFrame from staging table

        Returns:
            DataFrame with columns mapped to what the ML model expects
        """
        # Create a new DataFrame for the model input
        model_df = pd.DataFrame()

        # Map columns
        for target_col, source_col in self.column_mapping.items():
            if source_col in staging_df.columns:
                model_df[target_col] = staging_df[source_col]
            else:
                # Use empty values for missing columns
                model_df[target_col] = ""

        # Fill required fields with defaults if missing
        for field, default in self.required_fields.items():
            if field not in model_df.columns or model_df[field].isna().all():
                model_df[field] = default

        # Handle numeric fields - convert to proper numeric values
        for field, default in self.numeric_fields.items():
            if field in model_df.columns:
                # Convert to numeric, coercing errors to NaN
                model_df[field] = pd.to_numeric(model_df[field], errors="coerce")
                # Fill NaN values with default
                model_df[field] = model_df[field].fillna(default)
            else:
                # Create the field with default value if missing
                model_df[field] = default

        # Create required columns for the ML model
        if (
            "service_life" not in model_df.columns
            and "Service Life" in model_df.columns
        ):
            model_df["service_life"] = pd.to_numeric(
                model_df["Service Life"], errors="coerce"
            ).fillna(20.0)

        return model_df

    def map_predictions_to_master_db(
        self, predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Maps model predictions to master database fields.

        Args:
            predictions: Dictionary of predictions from the ML model

        Returns:
            Dictionary with fields mapped to master DB structure
        """
        # Map to Equipment and Equipment_Categories tables
        equipment_data = {
            "category_name": predictions.get("category_name", "Unknown"),
            "uniformat_code": predictions.get("uniformat_code", ""),
            "mcaa_system_category": predictions.get("mcaa_system_category", ""),
            "MasterFormat_Class": predictions.get("MasterFormat_Class", ""),
            "equipment_tag": predictions.get(
                "equipment_tag", ""
            ),  # Required NOT NULL field
        }

        # Add classification IDs directly from the data
        if "omniclass_code" in predictions:
            equipment_data["OmniClass_ID"] = predictions["omniclass_code"]
        if "uniformat_code" in predictions:
            equipment_data["Uniformat_ID"] = predictions["uniformat_code"]
        if "masterformat_code" in predictions:
            equipment_data["MasterFormat_ID"] = predictions["masterformat_code"]

        # Use CategoryID directly from the data if available
        if "CategoryID" in predictions:
            equipment_data["CategoryID"] = predictions["CategoryID"]
        else:
            # Map CategoryID (foreign key to Equipment_Categories)
            equipment_data["CategoryID"] = self._map_to_category_id(
                equipment_data["category_name"]
            )

        # Use LocationID directly from the data if available
        if "LocationID" in predictions:
            equipment_data["LocationID"] = predictions["LocationID"]
        else:
            # Default LocationID if not provided
            equipment_data["LocationID"] = 1  # Default location ID

        return equipment_data

    def _map_to_category_id(self, equipment_category: str) -> int:
        """
        Maps an equipment category name to a CategoryID for the master database.
        In a real implementation, this would query the Equipment_Categories table.

        Args:
            equipment_category: The equipment category name

        Returns:
            CategoryID as an integer
        """
        # This is a placeholder. In a real implementation, this would query
        # the Equipment_Categories table or use a mapping dictionary.
        # For now, we'll use a simple hash function to generate a positive integer
        category_hash = hash(equipment_category) % 10000
        return abs(category_hash) + 1  # Ensure positive and non-zero


def map_staging_to_model_input(staging_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps staging data columns to the format expected by the ML model.

    Args:
        staging_df: DataFrame from staging table

    Returns:
        DataFrame with columns mapped to what the ML model expects
    """
    mapper = DataMapper()
    return mapper.map_staging_to_model_input(staging_df)


def map_predictions_to_master_db(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps model predictions to master database fields.

    Args:
        predictions: Dictionary of predictions from the ML model

    Returns:
        Dictionary with fields mapped to master DB structure
    """
    mapper = DataMapper()
    return mapper.map_predictions_to_master_db(predictions)
