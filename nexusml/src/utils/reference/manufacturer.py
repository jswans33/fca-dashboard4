"""
Manufacturer Reference Data Sources

This module provides classes for manufacturer data sources:
- ManufacturerDataSource (base class)
- SMACNADataSource
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from nexusml.core.reference.base import ReferenceDataSource


class ManufacturerDataSource(ReferenceDataSource):
    """Base class for manufacturer data sources (SMACNA)."""

    def __init__(self, config: Dict[str, Any], base_path: Path, source_key: str):
        """
        Initialize the manufacturer data source.

        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            source_key: Key identifying this data source in the config
        """
        super().__init__(config, base_path)
        self.source_key = source_key

    def find_manufacturers_by_product(self, product: str) -> List[Dict[str, Any]]:
        """
        Find manufacturers that produce a specific product.

        Args:
            product: Product description or keyword

        Returns:
            List of manufacturer information dictionaries
        """
        if self.data is None or product is None:
            return []

        product_lower = product.lower()
        results = []

        if isinstance(self.data, list):
            for manufacturer in self.data:
                if "products" in manufacturer and isinstance(
                    manufacturer["products"], list
                ):
                    for prod in manufacturer["products"]:
                        if product_lower in prod.lower():
                            results.append(manufacturer)
                            break

        return results

    def find_products_by_manufacturer(self, manufacturer: str) -> List[str]:
        """
        Find products made by a specific manufacturer.

        Args:
            manufacturer: Manufacturer name

        Returns:
            List of product descriptions
        """
        if self.data is None or manufacturer is None:
            return []

        manufacturer_lower = manufacturer.lower()

        if isinstance(self.data, list):
            for manuf in self.data:
                if manufacturer_lower in manuf.get("name", "").lower():
                    return manuf.get("products", [])

        return []


class SMACNADataSource(ManufacturerDataSource):
    """SMACNA manufacturer data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the SMACNA data source."""
        super().__init__(config, base_path, "smacna")

    def load(self) -> None:
        """Load SMACNA manufacturer data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: SMACNA path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            json_files = list(path.glob(pattern))

            if not json_files:
                print(
                    f"Warning: No SMACNA files found matching pattern {pattern} in {path}"
                )
                return

            # Parse JSON files for manufacturer data
            manufacturers = []
            for file in json_files:
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Process manufacturer data
                    # Assuming format with manufacturer name, representative, and products
                    for item in data:
                        manufacturer = {
                            "name": item.get("Manufacturer", ""),
                            "representative": item.get("Representative", ""),
                            "products": item.get("Product_Description", "").split(", "),
                        }
                        manufacturers.append(manufacturer)

                except Exception as e:
                    print(f"Warning: Could not read SMACNA file {file}: {e}")

            if manufacturers:
                self.data = manufacturers
                print(f"Loaded {len(self.data)} SMACNA manufacturers")
            else:
                print("Warning: No SMACNA data loaded")

        except Exception as e:
            print(f"Error loading SMACNA data: {e}")
