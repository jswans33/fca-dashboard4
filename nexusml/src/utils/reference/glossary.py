"""
Glossary Reference Data Sources

This module provides classes for glossary data sources:
- GlossaryDataSource (base class)
- MCAAGlossaryDataSource
- MCAAAbbrDataSource
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, Optional

from nexusml.core.reference.base import ReferenceDataSource


class GlossaryDataSource(ReferenceDataSource):
    """Base class for glossary data sources (MCAA glossary, abbreviations)."""

    def __init__(self, config: Dict[str, Any], base_path: Path, source_key: str):
        """
        Initialize the glossary data source.

        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            source_key: Key identifying this data source in the config
        """
        super().__init__(config, base_path)
        self.source_key = source_key

    def get_definition(self, term: str) -> Optional[str]:
        """
        Get the definition for a term.

        Args:
            term: Term to look up

        Returns:
            Definition or None if not found
        """
        if self.data is None or term is None:
            return None

        term_lower = term.lower()
        if term_lower in self.data:
            return self.data[term_lower]

        # Try partial matches
        if self.data is not None and isinstance(self.data, dict):
            for key, value in self.data.items():
                if term_lower in key or key in term_lower:
                    return value

        return None


class MCAAGlossaryDataSource(GlossaryDataSource):
    """MCAA glossary data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the MCAA glossary data source."""
        super().__init__(config, base_path, "mcaa_glossary")

    def load(self) -> None:
        """Load MCAA glossary data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: MCAA glossary path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No MCAA glossary files found matching pattern {pattern} in {path}"
                )
                return

            # Parse CSV files for glossary terms
            glossary = {}
            for file in csv_files:
                try:
                    with open(file, "r", encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        # Skip header row
                        next(reader, None)

                        for row in reader:
                            if len(row) >= 2:
                                term, definition = row[0], row[1]
                                glossary[term.lower()] = definition.strip()

                except Exception as e:
                    print(f"Warning: Could not read MCAA glossary file {file}: {e}")

            if glossary:
                self.data = glossary
                print(f"Loaded {len(self.data)} MCAA glossary terms")
            else:
                print("Warning: No MCAA glossary data loaded")

        except Exception as e:
            print(f"Error loading MCAA glossary data: {e}")


class MCAAAbbrDataSource(GlossaryDataSource):
    """MCAA abbreviations data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the MCAA abbreviations data source."""
        super().__init__(config, base_path, "mcaa_abbreviations")

    def load(self) -> None:
        """Load MCAA abbreviations data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: MCAA abbreviations path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No MCAA abbreviations files found matching pattern {pattern} in {path}"
                )
                return

            # Parse CSV files for abbreviations
            abbreviations = {}
            for file in csv_files:
                try:
                    with open(file, "r", encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        # Skip header row
                        next(reader, None)

                        for row in reader:
                            if len(row) >= 2:
                                abbr, meaning = row[0], row[1]
                                abbreviations[abbr.lower()] = meaning.strip()

                except Exception as e:
                    print(
                        f"Warning: Could not read MCAA abbreviations file {file}: {e}"
                    )

            if abbreviations:
                self.data = abbreviations
                print(f"Loaded {len(self.data)} MCAA abbreviations")
            else:
                print("Warning: No MCAA abbreviations data loaded")

        except Exception as e:
            print(f"Error loading MCAA abbreviations data: {e}")
