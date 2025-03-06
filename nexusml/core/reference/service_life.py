"""
Service Life Reference Data Sources

This module provides classes for service life data sources:
- ServiceLifeDataSource (base class)
- ASHRAEDataSource
- EnergizeDenverDataSource
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from nexusml.core.reference.base import ReferenceDataSource


class ServiceLifeDataSource(ReferenceDataSource):
    """Base class for service life data sources (ASHRAE, Energize Denver)."""

    def __init__(self, config: Dict[str, Any], base_path: Path, source_key: str):
        """
        Initialize the service life data source.

        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            source_key: Key identifying this data source in the config
        """
        super().__init__(config, base_path)
        self.source_key = source_key
        self.column_mappings = self.config.get("column_mappings", {}).get(
            "service_life", {}
        )

    def get_service_life(self, equipment_type: str) -> Dict[str, Any]:
        """
        Get service life information for an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Dictionary with service life information
        """
        if self.data is None or equipment_type is None:
            return self._get_default_service_life()

        equipment_type_lower = equipment_type.lower()
        equipment_col = self.column_mappings.get("equipment_type")

        if not equipment_col or equipment_col not in self.data.columns:
            return self._get_default_service_life()

        # Try exact match
        match = self.data[self.data[equipment_col].str.lower() == equipment_type_lower]

        # If no exact match, try partial match
        if match.empty and self.data is not None:
            for idx, row in self.data.iterrows():
                if (
                    equipment_type_lower in str(row[equipment_col]).lower()
                    or str(row[equipment_col]).lower() in equipment_type_lower
                ):
                    match = self.data.iloc[[idx]]
                    break

        if match.empty:
            return self._get_default_service_life()

        row = match.iloc[0]

        return {
            "median_years": row.get(
                self.column_mappings.get("median_years"),
                self.config.get("defaults", {}).get("service_life", 15.0),
            ),
            "min_years": row.get(self.column_mappings.get("min_years"), 0.0),
            "max_years": row.get(self.column_mappings.get("max_years"), 0.0),
            "source": row.get(self.column_mappings.get("source"), self.source_key),
        }

    def _get_default_service_life(self) -> Dict[str, Any]:
        """Get default service life information."""
        default_life = self.config.get("defaults", {}).get("service_life", 15.0)
        return {
            "median_years": default_life,
            "min_years": default_life * 0.7,  # Estimate
            "max_years": default_life * 1.3,  # Estimate
            "source": f"{self.source_key}_default",
        }


class ASHRAEDataSource(ServiceLifeDataSource):
    """ASHRAE service life data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the ASHRAE data source."""
        super().__init__(config, base_path, "ashrae")

    def load(self) -> None:
        """Load ASHRAE service life data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: ASHRAE path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No ASHRAE files found matching pattern {pattern} in {path}"
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
                    # Add source column if not present
                    if "source" not in df.columns:
                        df["source"] = "ASHRAE"
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read ASHRAE file {file}: {e}")

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(self.data)} ASHRAE service life entries")
            else:
                print("Warning: No ASHRAE data loaded")

        except Exception as e:
            print(f"Error loading ASHRAE data: {e}")


class EnergizeDenverDataSource(ServiceLifeDataSource):
    """Energize Denver service life data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the Energize Denver data source."""
        super().__init__(config, base_path, "energize_denver")

    def load(self) -> None:
        """Load Energize Denver service life data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: Energize Denver path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No Energize Denver files found matching pattern {pattern} in {path}"
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
                    # Add source column if not present
                    if "source" not in df.columns:
                        df["source"] = "Energize Denver"
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read Energize Denver file {file}: {e}")

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(f"Loaded {len(self.data)} Energize Denver service life entries")
            else:
                print("Warning: No Energize Denver data loaded")

        except Exception as e:
            print(f"Error loading Energize Denver data: {e}")
