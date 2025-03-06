"""
Dynamic Field Mapper

This module provides a flexible way to map input fields to the expected format
for the ML model, regardless of the exact column names in the input data.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml


class DynamicFieldMapper:
    """Maps input data fields to model fields using flexible pattern matching."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the mapper with a configuration file.

        Args:
            config_path: Path to the configuration YAML file.
                         If None, uses the default path.
        """
        self.config_path = config_path
        self.load_config()

    def load_config(self) -> None:
        """Load the field mapping configuration."""
        if self.config_path is None:
            # Use default path
            config_path = (
                Path(__file__).resolve().parent.parent
                / "config"
                / "classification_config.yml"
            )
        else:
            config_path = Path(self.config_path)

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.field_mappings = self.config.get("input_field_mappings", [])
        self.classification_targets = self.config.get("classification_targets", [])

    def get_best_match(
        self, available_columns: List[str], target_field: str
    ) -> Optional[str]:
        """
        Find the best matching column for a target field.

        Args:
            available_columns: List of available column names
            target_field: Target field name to match

        Returns:
            Best matching column name or None if no match found
        """
        # First try exact match
        if target_field in available_columns:
            return target_field

        # Then try pattern matching from config
        for mapping in self.field_mappings:
            if mapping["target"] == target_field:
                for pattern in mapping["patterns"]:
                    if pattern in available_columns:
                        return pattern

                    # Try case-insensitive matching
                    pattern_lower = pattern.lower()
                    for col in available_columns:
                        if col.lower() == pattern_lower:
                            return col

        # No match found
        return None

    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map input dataframe columns to the format expected by the ML model.

        Args:
            df: Input DataFrame with arbitrary column names

        Returns:
            DataFrame with columns mapped to what the model expects
        """
        result_df = pd.DataFrame()
        available_columns = df.columns.tolist()

        # Get required fields for the model from feature_config.yml
        try:
            feature_config_path = (
                Path(__file__).resolve().parent.parent / "config" / "feature_config.yml"
            )
            with open(feature_config_path, "r") as f:
                feature_config = yaml.safe_load(f)

            # Extract text combination fields
            required_fields = []
            for combo in feature_config.get("text_combinations", []):
                required_fields.extend(combo.get("columns", []))

            # Add numeric fields
            for num_col in feature_config.get("numeric_columns", []):
                required_fields.append(num_col.get("name"))

            # Add hierarchy parent fields
            for hierarchy in feature_config.get("hierarchies", []):
                required_fields.extend(hierarchy.get("parents", []))

            # Add source fields from column mappings
            for mapping in feature_config.get("column_mappings", []):
                required_fields.append(mapping.get("source"))

            # Remove duplicates
            required_fields = list(set([f for f in required_fields if f]))

        except Exception as e:
            print(f"Warning: Could not load feature config: {e}")
            # Default required fields if feature config can't be loaded
            required_fields = [
                "Asset Category",
                "Equip Name ID",
                "System Type ID",
                "Precon System",
                "Sub System Type",
                "Service Life",
            ]

        # Map each required field
        for field in required_fields:
            best_match = self.get_best_match(available_columns, field)

            if best_match:
                # Copy the column with the new name
                result_df[field] = df[best_match]
            else:
                # Create empty column if no match found
                result_df[field] = ""

        return result_df

    def get_classification_targets(self) -> List[str]:
        """
        Get the list of classification targets.

        Returns:
            List of classification target names
        """
        return [target["name"] for target in self.classification_targets]

    def get_required_db_fields(self) -> Dict[str, Dict]:
        """
        Get the mapping of classification targets to database fields.

        Returns:
            Dictionary mapping classification names to DB field info
        """
        result = {}
        for target in self.classification_targets:
            if target.get("required", False) and "master_db" in target:
                result[target["name"]] = target["master_db"]

        return result
