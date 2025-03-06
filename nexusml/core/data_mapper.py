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
            "Asset Name": "Asset Name",
            "Asset Tag": "Asset Tag",
            "Trade": "Trade",
            "Manufacturer": "Manufacturer",
            "Model Number": "Model Number",
            "Size": "Size",
            "Unit": "Unit",
            "Motor HP": "Motor HP",
            "System Category": "System Category",
            "Sub System Type": "Sub System Type",
            "Sub System Classification": "Sub System Classification",
            "Service Life": "Service Life",
        }

        # Required fields with default values
        self.required_fields = {
            "Asset Name": "Unknown Equipment",
            "System Category": "Unknown System",
            "Trade": "H",  # Default to HVAC
        }

        # Numeric fields with default values
        self.numeric_fields = {"Service Life": 20.0, "Motor HP": 0.0, "Size": 0.0}

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
            "Equipment_Category": predictions.get("Equipment_Category", "Unknown"),
            "Uniformat_Class": predictions.get("Uniformat_Class", ""),
            "System_Type": predictions.get("System_Type", ""),
            "MasterFormat_Class": predictions.get("MasterFormat_Class", ""),
            "EquipmentTag": predictions.get("Asset Tag", ""),  # Required NOT NULL field
        }

        # Add classification IDs from EAV
        if "OmniClass_ID" in predictions:
            equipment_data["OmniClass_ID"] = predictions["OmniClass_ID"]
        if "Uniformat_ID" in predictions:
            equipment_data["Uniformat_ID"] = predictions["Uniformat_ID"]
        if "MasterFormat_ID" in predictions:
            equipment_data["MasterFormat_ID"] = predictions["MasterFormat_ID"]

        # Map CategoryID (foreign key to Equipment_Categories)
        equipment_data["CategoryID"] = self._map_to_category_id(
            equipment_data["Equipment_Category"]
        )

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
