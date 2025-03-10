"""
Equipment Taxonomy Reference Data Source

This module provides the EquipmentTaxonomyDataSource class for accessing
equipment taxonomy data.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from pandas import DataFrame, Series

from nexusml.core.reference.base import ReferenceDataSource


class EquipmentTaxonomyDataSource(ReferenceDataSource):
    """
    Equipment taxonomy data source.

    This class provides access to equipment taxonomy data, including:
    - Equipment categories and types
    - Service life information
    - Maintenance requirements
    - System classifications
    """

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """
        Initialize the equipment taxonomy data source.

        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
        """
        super().__init__(config, base_path)
        self.source_key = "equipment_taxonomy"
        self.column_mappings = self.config.get("column_mappings", {}).get(
            "equipment_taxonomy", {}
        )

    def load(self) -> None:
        """
        Load equipment taxonomy data from CSV files.

        Searches for CSV files in the configured path and loads them into a DataFrame.
        """
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: Equipment taxonomy path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No equipment taxonomy files found matching pattern {pattern} in {path}"
                )
                return

            # Read and combine all CSV files
            dfs = []
            for file in csv_files:
                try:
                    df = pd.read_csv(file)
                    # Standardize column names based on mapping
                    if self.column_mappings:
                        df = df.rename(
                            columns={
                                v: k
                                for k, v in self.column_mappings.items()
                                if v in df.columns
                            }
                        )
                    dfs.append(df)
                except Exception as e:
                    print(
                        f"Warning: Could not read equipment taxonomy file {file}: {e}"
                    )

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(self.data)} equipment taxonomy entries")
            else:
                print("Warning: No equipment taxonomy data loaded")

        except Exception as e:
            print(f"Error loading equipment taxonomy data: {e}")

    def _find_equipment_match(self, search_term: str) -> Optional[Series]:
        """
        Find equipment matching the search term.

        Args:
            search_term: Term to search for

        Returns:
            Matching equipment row or None if not found
        """
        if self.data is None or not search_term:
            return None

        search_term_lower = search_term.lower()

        # Search columns in priority order
        search_columns = [
            "Asset Category",
            "title",
            "equipment_id",
            "drawing_abbreviation",
        ]

        # Try exact matches first
        for col in search_columns:
            if col not in self.data.columns:
                continue

            exact_matches = self.data[self.data[col].str.lower() == search_term_lower]
            if not exact_matches.empty:
                return exact_matches.iloc[0]

        # Try partial matches
        for col in search_columns:
            if col not in self.data.columns:
                continue

            for _, row in self.data.iterrows():
                if pd.isna(row[col]):
                    continue

                col_value = str(row[col]).lower()
                if search_term_lower in col_value or col_value in search_term_lower:
                    return row

        return None

    def get_equipment_info(self, equipment_type: str) -> Optional[Dict[str, Any]]:
        """
        Get equipment information for a given equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Dictionary with equipment information or None if not found
        """
        match = self._find_equipment_match(equipment_type)
        if match is not None:
            # Convert to a standard Python dict with str keys
            return {str(k): v for k, v in match.to_dict().items()}
        return None

    def get_service_life(self, equipment_type: str) -> Dict[str, Any]:
        """
        Get service life information for an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Dictionary with service life information
        """
        equipment_info = self.get_equipment_info(equipment_type)

        if equipment_info is None or "service_life" not in equipment_info:
            return self._get_default_service_life()

        try:
            service_life = float(equipment_info["service_life"])
            return {
                "median_years": service_life,
                "min_years": service_life * 0.7,  # Estimate
                "max_years": service_life * 1.3,  # Estimate
                "source": "equipment_taxonomy",
            }
        except (ValueError, TypeError):
            return self._get_default_service_life()

    def _get_default_service_life(self) -> Dict[str, Any]:
        """
        Get default service life information.

        Returns:
            Dictionary with default service life values
        """
        default_life = self.config.get("defaults", {}).get("service_life", 15.0)
        return {
            "median_years": default_life,
            "min_years": default_life * 0.7,  # Estimate
            "max_years": default_life * 1.3,  # Estimate
            "source": "equipment_taxonomy_default",
        }

    def get_maintenance_hours(self, equipment_type: str) -> Optional[float]:
        """
        Get maintenance hours for an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Maintenance hours or None if not found
        """
        equipment_info = self.get_equipment_info(equipment_type)

        if equipment_info is None or "service_maintenance_hrs" not in equipment_info:
            return None

        try:
            return float(equipment_info["service_maintenance_hrs"])
        except (ValueError, TypeError):
            return None

    def _filter_by_column(self, column: str, value: str) -> List[Dict[str, Any]]:
        """
        Filter equipment by a specific column value.

        Args:
            column: Column name to filter on
            value: Value to match

        Returns:
            List of matching equipment dictionaries
        """
        if self.data is None or column not in self.data.columns:
            return []

        value_lower = value.lower()
        matches = self.data[self.data[column].str.lower() == value_lower]

        if matches.empty:
            return []

        # Convert to list of dicts with string keys
        return [
            {str(k): v for k, v in record.items()}
            for record in matches.to_dict(orient="records")
        ]

    def get_equipment_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all equipment in a specific category.

        Args:
            category: Asset category

        Returns:
            List of equipment dictionaries
        """
        return self._filter_by_column("Asset Category", category)

    def get_equipment_by_system(self, system_type: str) -> List[Dict[str, Any]]:
        """
        Get all equipment in a specific system type.

        Args:
            system_type: System type

        Returns:
            List of equipment dictionaries
        """
        # Try different system columns in order
        for col in ["system_type_id", "sub_system_type", "sub_system_id"]:
            results = self._filter_by_column(col, system_type)
            if results:
                return results

        return []
