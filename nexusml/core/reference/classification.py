"""
Classification Reference Data Sources

This module provides classes for classification data sources:
- ClassificationDataSource (base class)
- OmniClassDataSource
- UniformatDataSource
- MasterFormatDataSource
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from nexusml.core.reference.base import ReferenceDataSource


class ClassificationDataSource(ReferenceDataSource):
    """Base class for classification data sources (OmniClass, Uniformat, MasterFormat)."""

    def __init__(self, config: Dict[str, Any], base_path: Path, source_key: str):
        """
        Initialize the classification data source.

        Args:
            config: Configuration dictionary
            base_path: Base path for resolving relative paths
            source_key: Key identifying this data source in the config
        """
        super().__init__(config, base_path)
        self.source_key = source_key
        self.column_mappings = self.config.get("column_mappings", {}).get(
            source_key, {}
        )
        self.hierarchy_config = self.config.get("hierarchies", {}).get(source_key, {})

    def get_parent_code(self, code: str) -> Optional[str]:
        """
        Get the parent code for a classification code.

        Args:
            code: Classification code

        Returns:
            Parent code or None if at top level
        """
        if not code:
            return None

        separator = self.hierarchy_config.get("separator", "-")
        levels = self.hierarchy_config.get("levels", 3)

        parts = code.split(separator)
        if len(parts) <= 1:
            return None

        # For codes like "23-70-00", create parent by setting last non-zero segment to 00
        for i in range(len(parts) - 1, 0, -1):
            if parts[i] != "00" and parts[i] != "0000":
                parts[i] = "00" if len(parts[i]) == 2 else "0000"
                return separator.join(parts)

        return None

    def get_description(self, code: str) -> Optional[str]:
        """
        Get the description for a classification code.

        Args:
            code: Classification code

        Returns:
            Description or None if not found
        """
        if self.data is None or code is None:
            return None

        code_col = self.column_mappings.get("code")
        desc_col = self.column_mappings.get("description")

        if (
            not code_col
            or not desc_col
            or code_col not in self.data.columns
            or desc_col not in self.data.columns
        ):
            return None

        match = self.data[self.data[code_col] == code]
        if match.empty:
            return None

        return match.iloc[0][desc_col]

    def find_similar_codes(self, code: str, n: int = 5) -> List[str]:
        """
        Find similar classification codes.

        Args:
            code: Classification code
            n: Number of similar codes to return

        Returns:
            List of similar codes
        """
        if self.data is None or code is None:
            return []

        parent = self.get_parent_code(code)
        if not parent:
            return []

        code_col = self.column_mappings.get("code")
        if not code_col or code_col not in self.data.columns:
            return []

        separator = self.hierarchy_config.get("separator", "-")

        # Get siblings (codes with same parent)
        siblings = self.data[
            self.data[code_col].str.startswith(parent.replace(separator + "00", ""))
        ]

        if siblings.empty:
            return []

        return siblings[code_col].tolist()[:n]


class OmniClassDataSource(ClassificationDataSource):
    """OmniClass taxonomy data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the OmniClass data source."""
        super().__init__(config, base_path, "omniclass")

    def load(self) -> None:
        """Load OmniClass taxonomy data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: OmniClass path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No OmniClass files found matching pattern {pattern} in {path}"
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
                    print(f"Warning: Could not read OmniClass file {file}: {e}")

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(
                    f"Loaded {len(self.data)} OmniClass entries from {len(csv_files)} files"
                )
            else:
                print("Warning: No OmniClass data loaded")

        except Exception as e:
            print(f"Error loading OmniClass data: {e}")


class UniformatDataSource(ClassificationDataSource):
    """Uniformat classification data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the Uniformat data source."""
        super().__init__(config, base_path, "uniformat")
        self.keywords_data = None

    def load(self) -> None:
        """Load Uniformat classification data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: Uniformat path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No Uniformat files found matching pattern {pattern} in {path}"
                )
                return

            # Read and combine all CSV files
            dfs = []
            for file in csv_files:
                try:
                    # Skip the keywords file, we'll handle it separately
                    if "uniformat-keywords.csv" in str(file):
                        self._load_keywords(file)
                        continue

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
                    print(f"Warning: Could not read Uniformat file {file}: {e}")

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(
                    f"Loaded {len(self.data)} Uniformat entries from {len(csv_files)} files"
                )
            else:
                print("Warning: No Uniformat data loaded")

        except Exception as e:
            print(f"Error loading Uniformat data: {e}")

    def _load_keywords(self, file_path: Path) -> None:
        """
        Load Uniformat keywords data from a CSV file.

        Args:
            file_path: Path to the keywords CSV file
        """
        try:
            self.keywords_data = pd.read_csv(file_path)
            print(
                f"Loaded {len(self.keywords_data)} Uniformat keywords from {file_path}"
            )
        except Exception as e:
            print(f"Warning: Could not read Uniformat keywords file {file_path}: {e}")
            self.keywords_data = None

    def find_codes_by_keyword(
        self, keyword: str, max_results: int = 10
    ) -> List[Dict[str, str]]:
        """
        Find Uniformat codes by keyword.

        Args:
            keyword: Keyword to search for
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries with Uniformat code, title, and MasterFormat number
        """
        if self.keywords_data is None:
            return []

        # Case-insensitive search
        keyword = keyword.lower()

        # Search for the keyword in the Keyword column
        matches = self.keywords_data[
            self.keywords_data["Keyword"].str.lower().str.contains(keyword, na=False)
        ]

        if matches.empty:
            return []

        # Limit the number of results
        matches = matches.head(max_results)

        # Convert to list of dictionaries
        results = []
        for _, row in matches.iterrows():
            results.append(
                {
                    "uniformat_code": (
                        row["UF Number"] if pd.notna(row["UF Number"]) else ""
                    ),
                    "masterformat_code": (
                        row["MF Number"] if pd.notna(row["MF Number"]) else ""
                    ),
                    "keyword": row["Keyword"],
                }
            )

        return results


class MasterFormatDataSource(ClassificationDataSource):
    """MasterFormat classification data source."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """Initialize the MasterFormat data source."""
        super().__init__(config, base_path, "masterformat")

    def load(self) -> None:
        """Load MasterFormat classification data."""
        path = self.get_path(self.source_key)
        if not path or not path.exists():
            print(f"Warning: MasterFormat path not found: {path}")
            return

        try:
            pattern = self.get_file_pattern(self.source_key)
            csv_files = list(path.glob(pattern))

            if not csv_files:
                print(
                    f"Warning: No MasterFormat files found matching pattern {pattern} in {path}"
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
                    print(f"Warning: Could not read MasterFormat file {file}: {e}")

            if dfs:
                self.data = pd.concat(dfs, ignore_index=True)
                print(
                    f"Loaded {len(self.data)} MasterFormat entries from {len(csv_files)} files"
                )
            else:
                print("Warning: No MasterFormat data loaded")

        except Exception as e:
            print(f"Error loading MasterFormat data: {e}")
