"""
Reference Manager

This module provides the main facade for accessing all reference data sources.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from nexusml.core.reference.base import ReferenceDataSource
from nexusml.core.reference.classification import (
    MasterFormatDataSource,
    OmniClassDataSource,
    UniformatDataSource,
)
from nexusml.core.reference.equipment import EquipmentTaxonomyDataSource
from nexusml.core.reference.glossary import (
    MCAAAbbrDataSource,
    MCAAGlossaryDataSource,
)
from nexusml.core.reference.manufacturer import SMACNADataSource
from nexusml.core.reference.service_life import (
    ASHRAEDataSource,
    EnergizeDenverDataSource,
)


class ReferenceManager:
    """
    Unified manager for all reference data sources.

    This class follows the Facade pattern to provide a simple interface
    to the complex subsystem of reference data sources.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the reference manager.

        Args:
            config_path: Path to the reference configuration file.
                         If None, uses the default path.
        """
        self.config = self._load_config(config_path)
        self.base_path = self._get_base_path()

        # Initialize data sources
        self.omniclass = OmniClassDataSource(self.config, self.base_path)
        self.uniformat = UniformatDataSource(self.config, self.base_path)
        self.masterformat = MasterFormatDataSource(self.config, self.base_path)
        self.mcaa_glossary = MCAAGlossaryDataSource(self.config, self.base_path)
        self.mcaa_abbreviations = MCAAAbbrDataSource(self.config, self.base_path)
        self.smacna = SMACNADataSource(self.config, self.base_path)
        self.ashrae = ASHRAEDataSource(self.config, self.base_path)
        self.energize_denver = EnergizeDenverDataSource(self.config, self.base_path)
        self.equipment_taxonomy = EquipmentTaxonomyDataSource(
            self.config, self.base_path
        )

        # List of all data sources for batch operations
        self.data_sources = [
            self.omniclass,
            self.uniformat,
            self.masterformat,
            self.mcaa_glossary,
            self.mcaa_abbreviations,
            self.smacna,
            self.ashrae,
            self.energize_denver,
            self.equipment_taxonomy,
        ]

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load the reference configuration.

        Args:
            config_path: Path to the configuration file

        Returns:
            Configuration dictionary
        """
        try:
            if not config_path:
                # Use default path
                root_dir = Path(__file__).resolve().parent.parent.parent
                config_path = str(root_dir / "config" / "reference_config.yml")

            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load reference config: {e}")
            return {}

    def _get_base_path(self) -> Path:
        """
        Get the base path for resolving relative paths.

        Returns:
            Base path
        """
        return Path(__file__).resolve().parent.parent.parent.parent

    def load_all(self) -> None:
        """Load all reference data sources."""
        for source in self.data_sources:
            source.load()

    def get_omniclass_description(self, code: str) -> Optional[str]:
        """
        Get the OmniClass description for a code.

        Args:
            code: OmniClass code

        Returns:
            Description or None if not found
        """
        return self.omniclass.get_description(code)

    def get_uniformat_description(self, code: str) -> Optional[str]:
        """
        Get the Uniformat description for a code.

        Args:
            code: Uniformat code

        Returns:
            Description or None if not found
        """
        return self.uniformat.get_description(code)

    def get_masterformat_description(self, code: str) -> Optional[str]:
        """
        Get the MasterFormat description for a code.

        Args:
            code: MasterFormat code

        Returns:
            Description or None if not found
        """
        return self.masterformat.get_description(code)

    def get_term_definition(self, term: str) -> Optional[str]:
        """
        Get the definition for a term from the MCAA glossary.

        Args:
            term: Term to look up

        Returns:
            Definition or None if not found
        """
        return self.mcaa_glossary.get_definition(term)

    def get_abbreviation_meaning(self, abbr: str) -> Optional[str]:
        """
        Get the meaning of an abbreviation from the MCAA abbreviations.

        Args:
            abbr: Abbreviation to look up

        Returns:
            Meaning or None if not found
        """
        return self.mcaa_abbreviations.get_definition(abbr)

    def find_manufacturers_by_product(self, product: str) -> List[Dict[str, Any]]:
        """
        Find manufacturers that produce a specific product.

        Args:
            product: Product description or keyword

        Returns:
            List of manufacturer information dictionaries
        """
        return self.smacna.find_manufacturers_by_product(product)

    def get_service_life(self, equipment_type: str) -> Dict[str, Any]:
        """
        Get service life information for an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Dictionary with service life information
        """
        # Try ASHRAE first, then Energize Denver, then Equipment Taxonomy
        ashrae_result = self.ashrae.get_service_life(equipment_type)
        if ashrae_result.get("source") != "ashrae_default":
            return ashrae_result

        energize_denver_result = self.energize_denver.get_service_life(equipment_type)
        if energize_denver_result.get("source") != "energize_denver_default":
            return energize_denver_result

        return self.equipment_taxonomy.get_service_life(equipment_type)

    def get_equipment_info(self, equipment_type: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Dictionary with equipment information or None if not found
        """
        return self.equipment_taxonomy.get_equipment_info(equipment_type)

    def get_equipment_maintenance_hours(self, equipment_type: str) -> Optional[float]:
        """
        Get maintenance hours for an equipment type.

        Args:
            equipment_type: Equipment type description

        Returns:
            Maintenance hours or None if not found
        """
        return self.equipment_taxonomy.get_maintenance_hours(equipment_type)

    def get_equipment_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all equipment in a specific category.

        Args:
            category: Asset category

        Returns:
            List of equipment dictionaries
        """
        return self.equipment_taxonomy.get_equipment_by_category(category)

    def get_equipment_by_system(self, system_type: str) -> List[Dict[str, Any]]:
        """
        Get all equipment in a specific system type.

        Args:
            system_type: System type

        Returns:
            List of equipment dictionaries
        """
        return self.equipment_taxonomy.get_equipment_by_system(system_type)

    def validate_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all reference data sources to ensure data quality.

        This method checks:
        1. If data is loaded
        2. If required columns exist
        3. If data has the expected structure
        4. Basic data quality checks (nulls, duplicates, etc.)

        Returns:
            Dictionary with validation results for each data source
        """
        from nexusml.core.reference.validation import (
            validate_classification_data,
            validate_equipment_taxonomy_data,
            validate_glossary_data,
            validate_manufacturer_data,
            validate_service_life_data,
        )

        # Load data if not already loaded
        for source in self.data_sources:
            if source.data is None:
                source.load()

        results = {}

        # Validate classification data sources
        results["omniclass"] = validate_classification_data(
            self.omniclass, "omniclass", self.config
        )
        results["uniformat"] = validate_classification_data(
            self.uniformat, "uniformat", self.config
        )
        results["masterformat"] = validate_classification_data(
            self.masterformat, "masterformat", self.config
        )

        # Validate glossary data sources
        results["mcaa_glossary"] = validate_glossary_data(self.mcaa_glossary)
        results["mcaa_abbreviations"] = validate_glossary_data(self.mcaa_abbreviations)

        # Validate manufacturer data sources
        results["smacna"] = validate_manufacturer_data(self.smacna)

        # Validate service life data sources
        results["ashrae"] = validate_service_life_data(self.ashrae)
        results["energize_denver"] = validate_service_life_data(self.energize_denver)

        # Validate equipment taxonomy data
        results["equipment_taxonomy"] = validate_equipment_taxonomy_data(
            self.equipment_taxonomy
        )

        return results

    def enrich_equipment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich equipment data with reference information.

        Args:
            df: DataFrame with equipment data

        Returns:
            Enriched DataFrame
        """
        result_df = df.copy()

        # Add OmniClass descriptions if omniclass_code column exists
        if "omniclass_code" in result_df.columns:
            result_df["omniclass_description"] = result_df["omniclass_code"].apply(
                lambda x: self.get_omniclass_description(x) if pd.notna(x) else None
            )

        # Add Uniformat descriptions if uniformat_code column exists
        if "uniformat_code" in result_df.columns:
            result_df["uniformat_description"] = result_df["uniformat_code"].apply(
                lambda x: self.get_uniformat_description(x) if pd.notna(x) else None
            )

        # Add MasterFormat descriptions if masterformat_code column exists
        if "masterformat_code" in result_df.columns:
            result_df["masterformat_description"] = result_df[
                "masterformat_code"
            ].apply(
                lambda x: self.get_masterformat_description(x) if pd.notna(x) else None
            )

        # Add service life information if equipment_type column exists
        if "equipment_type" in result_df.columns:
            service_life_info = result_df["equipment_type"].apply(self.get_service_life)

            result_df["service_life_median"] = service_life_info.apply(
                lambda x: x.get("median_years")
            )
            result_df["service_life_min"] = service_life_info.apply(
                lambda x: x.get("min_years")
            )
            result_df["service_life_max"] = service_life_info.apply(
                lambda x: x.get("max_years")
            )
            result_df["service_life_source"] = service_life_info.apply(
                lambda x: x.get("source")
            )

            # Add maintenance hours from equipment taxonomy
            result_df["maintenance_hours"] = result_df["equipment_type"].apply(
                lambda x: (
                    self.get_equipment_maintenance_hours(x) if pd.notna(x) else None
                )
            )

            # Add equipment taxonomy information
            def safe_get_equipment_attribute(equip_type: Any, attribute: str) -> Any:
                """Safely get an attribute from equipment info."""
                if pd.isna(equip_type):
                    return None

                info = self.get_equipment_info(equip_type)
                if info is None:
                    return None

                return info.get(attribute)

            result_df["equipment_category"] = result_df["equipment_type"].apply(
                lambda x: safe_get_equipment_attribute(x, "Asset Category")
            )

            result_df["equipment_abbreviation"] = result_df["equipment_type"].apply(
                lambda x: safe_get_equipment_attribute(x, "Drawing Abbreviation")
            )

            result_df["equipment_trade"] = result_df["equipment_type"].apply(
                lambda x: safe_get_equipment_attribute(x, "Trade")
            )

        return result_df
