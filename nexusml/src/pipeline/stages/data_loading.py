"""
Data Loading Stage Module

This module provides implementations of the DataLoadingStage interface for
loading data from various sources.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from nexusml.config.manager import ConfigurationManager
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import BaseDataLoadingStage


class CSVDataLoadingStage(BaseDataLoadingStage):
    """
    Implementation of DataLoadingStage for loading data from CSV files.
    """

    def __init__(
        self,
        name: str = "CSVDataLoading",
        description: str = "Loads data from a CSV file",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the CSV data loading stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading data configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file. If None, uses the default path from config.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        # Use default path if none provided
        if data_path is None:
            # Try to get the default path from the configuration
            data_config = self.config_manager.get_data_config()
            training_data_config = getattr(data_config, "training_data", {})
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.csv"
            )
            data_path = str(
                Path(__file__).resolve().parent.parent.parent.parent / default_path
            )

        # Get encoding from config or kwargs
        encoding = kwargs.get(
            "encoding",
            self.config.get(
                "encoding",
                getattr(
                    getattr(self.config_manager.get_data_config(), "training_data", {}),
                    "encoding",
                    "utf-8",
                ),
            ),
        )
        fallback_encoding = kwargs.get(
            "fallback_encoding",
            self.config.get(
                "fallback_encoding",
                getattr(
                    getattr(self.config_manager.get_data_config(), "training_data", {}),
                    "fallback_encoding",
                    "latin1",
                ),
            ),
        )

        # Remove data_path from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if "data_path" in kwargs_copy:
            del kwargs_copy["data_path"]

        try:
            # Read CSV file using pandas
            df = pd.read_csv(data_path, encoding=encoding, **kwargs_copy)
        except UnicodeDecodeError:
            # Try with a different encoding if the primary one fails
            print(
                f"Warning: Failed to read with {encoding} encoding. Trying {fallback_encoding}."
            )
            df = pd.read_csv(data_path, encoding=fallback_encoding, **kwargs_copy)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {data_path}. Please provide a valid path."
            )

        return df


class ExcelDataLoadingStage(BaseDataLoadingStage):
    """
    Implementation of DataLoadingStage for loading data from Excel files.
    """

    def __init__(
        self,
        name: str = "ExcelDataLoading",
        description: str = "Loads data from an Excel file",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the Excel data loading stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading data configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from an Excel file.

        Args:
            data_path: Path to the Excel file. If None, uses the default path from config.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid.
        """
        # Use default path if none provided
        if data_path is None:
            # Try to get the default path from the configuration
            data_config = self.config_manager.get_data_config()
            training_data_config = getattr(data_config, "training_data", {})
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.xlsx"
            )
            data_path = str(
                Path(__file__).resolve().parent.parent.parent.parent / default_path
            )

        # Get sheet name from kwargs or config
        sheet_name = kwargs.get("sheet_name", self.config.get("sheet_name", 0))

        # Remove data_path from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if "data_path" in kwargs_copy:
            del kwargs_copy["data_path"]
        if "sheet_name" in kwargs_copy:
            del kwargs_copy["sheet_name"]

        try:
            # Read Excel file using pandas
            df = pd.read_excel(data_path, sheet_name=sheet_name, **kwargs_copy)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Data file not found at {data_path}. Please provide a valid path."
            )
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")

        return df


class SQLiteDataLoadingStage(BaseDataLoadingStage):
    """
    Implementation of DataLoadingStage for loading data from SQLite databases.
    """

    def __init__(
        self,
        name: str = "SQLiteDataLoading",
        description: str = "Loads data from a SQLite database",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the SQLite data loading stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading data configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a SQLite database.

        Args:
            data_path: Path to the SQLite database file. If None, uses the default path from config.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the database file cannot be found.
            ValueError: If the data format is invalid.
        """
        import sqlite3

        # Use default path if none provided
        if data_path is None:
            # Try to get the default path from the configuration
            data_config = self.config_manager.get_data_config()
            training_data_config = getattr(data_config, "training_data", {})
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.db"
            )
            data_path = str(
                Path(__file__).resolve().parent.parent.parent.parent / default_path
            )

        # Get query from kwargs or config
        query = kwargs.get("query", self.config.get("query", "SELECT * FROM equipment"))

        # Remove data_path from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if "data_path" in kwargs_copy:
            del kwargs_copy["data_path"]
        if "query" in kwargs_copy:
            del kwargs_copy["query"]

        try:
            # Connect to the database
            conn = sqlite3.connect(data_path)

            # Read data using pandas
            df = pd.read_sql_query(query, conn, **kwargs_copy)

            # Close the connection
            conn.close()
        except sqlite3.Error as e:
            raise ValueError(f"SQLite error: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Database file not found at {data_path}. Please provide a valid path."
            )

        return df


class ConfigurableDataLoadingStage(BaseDataLoadingStage):
    """
    Implementation of DataLoadingStage that can load data from various sources
    based on configuration.
    """

    def __init__(
        self,
        name: str = "ConfigurableDataLoading",
        description: str = "Loads data from various sources based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the configurable data loading stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading data configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self._loaders = {
            "csv": CSVDataLoadingStage(config=config, config_manager=config_manager),
            "excel": ExcelDataLoadingStage(
                config=config, config_manager=config_manager
            ),
            "sqlite": SQLiteDataLoadingStage(
                config=config, config_manager=config_manager
            ),
        }

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # This stage doesn't require any data from the context
        return True

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a source determined by the file extension or configuration.

        Args:
            data_path: Path to the data file. If None, uses the default path from config.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the data file cannot be found.
            ValueError: If the data format is invalid or unsupported.
        """
        # Use default path if none provided
        if data_path is None:
            # Try to get the default path from the configuration
            data_config = self.config_manager.get_data_config()
            training_data_config = getattr(data_config, "training_data", {})
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.csv"
            )
            data_path = str(
                Path(__file__).resolve().parent.parent.parent.parent / default_path
            )

        # Get the file extension
        file_ext = os.path.splitext(data_path)[1].lower()

        # Determine the loader to use
        loader_type = kwargs.get("loader_type", self.config.get("loader_type", None))
        if loader_type is None:
            # Determine loader type from file extension
            if file_ext == ".csv":
                loader_type = "csv"
            elif file_ext in (".xls", ".xlsx"):
                loader_type = "excel"
            elif file_ext == ".db":
                loader_type = "sqlite"
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")

        # Get the appropriate loader
        if loader_type not in self._loaders:
            raise ValueError(f"Unsupported loader type: {loader_type}")

        loader = self._loaders[loader_type]

        # Remove loader_type from kwargs to avoid confusion
        kwargs_copy = kwargs.copy()
        if "loader_type" in kwargs_copy:
            del kwargs_copy["loader_type"]

        # Load the data using the selected loader
        return loader.load_data(data_path, **kwargs_copy)
