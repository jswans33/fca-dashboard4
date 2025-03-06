"""
Base Reference Data Source

This module provides the abstract base class for all reference data sources.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class ReferenceDataSource(ABC):
    """Abstract base class for all reference data sources."""

    def __init__(self, config: Dict[str, Any], base_path: Path):
        """
        Initialize the reference data source.

        Args:
            config: Configuration dictionary for this data source
            base_path: Base path for resolving relative paths
        """
        self.config = config
        self.base_path = base_path
        self.data = None

    @abstractmethod
    def load(self) -> None:
        """Load the reference data."""
        pass

    def get_path(self, key: str) -> Optional[Path]:
        """
        Get the absolute path for a configuration key.

        Args:
            key: Configuration key for the path

        Returns:
            Absolute path or None if not found
        """
        path = self.config.get("paths", {}).get(key, "")
        if not path:
            return None

        if not os.path.isabs(path):
            return self.base_path / path
        return Path(path)

    def get_file_pattern(self, key: str) -> str:
        """
        Get the file pattern for a data source.

        Args:
            key: Data source key

        Returns:
            File pattern string
        """
        return self.config.get("file_patterns", {}).get(key, "*")
