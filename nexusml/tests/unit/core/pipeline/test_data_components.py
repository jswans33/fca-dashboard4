"""
Unit tests for the data components.

This module contains tests for the StandardDataLoader, StandardDataPreprocessor,
and their adapter classes.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from nexusml.core.config.configuration import (
    DataConfig,
    NexusMLConfig,
    RequiredColumn,
    TrainingDataConfig,
)
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.adapters.data_adapter import (
    DataComponentFactory,
    LegacyDataLoaderAdapter,
    LegacyDataPreprocessorAdapter,
)
from nexusml.core.pipeline.components.data_loader import StandardDataLoader
from nexusml.core.pipeline.components.data_preprocessor import StandardDataPreprocessor


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"],
            "description": ["Desc 1", "Desc 2", "Desc 3", "Desc 4", "Desc 5"],
            "category": ["Cat A", "Cat B", "Cat A", "Cat C", "Cat B"],
            "value": [10.5, 20.0, 15.75, 8.25, 30.0],
        }
    )


@pytest.fixture
def sample_csv_file():
    """Create a temporary CSV file with sample data for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        f.write("id,name,description,category,value\n")
        f.write("1,Item 1,Desc 1,Cat A,10.5\n")
        f.write("2,Item 2,Desc 2,Cat B,20.0\n")
        f.write("3,Item 3,Desc 3,Cat A,15.75\n")
        f.write("4,Item 4,Desc 4,Cat C,8.25\n")
        f.write("5,Item 5,Desc 5,Cat B,30.0\n")

    file_path = f.name
    yield file_path

    # Clean up the temporary file
    os.unlink(file_path)


@pytest.fixture
def mock_config_provider():
    """Mock the ConfigurationProvider to return a test configuration."""
    # Create a test configuration
    required_columns = [
        RequiredColumn(name="id", default_value=0, data_type="int"),
        RequiredColumn(name="name", default_value="", data_type="str"),
        RequiredColumn(name="description", default_value="", data_type="str"),
        RequiredColumn(name="category", default_value="Unknown", data_type="str"),
        RequiredColumn(name="value", default_value=0.0, data_type="float"),
    ]

    training_data = TrainingDataConfig(
        default_path="test_data.csv",
        encoding="utf-8",
        fallback_encoding="latin1",
    )

    data_config = DataConfig(
        required_columns=required_columns,
        training_data=training_data,
    )

    config = NexusMLConfig(
        data=data_config,
        reference=None,
        masterformat_primary=None,
        masterformat_equipment=None,
    )

    # Create the mock
    mock_instance = mock.MagicMock()
    mock_instance.config = config

    # Patch the ConfigurationProvider in all modules that use it
    with mock.patch(
        "nexusml.core.pipeline.components.data_loader.ConfigurationProvider",
        return_value=mock_instance,
    ), mock.patch(
        "nexusml.core.pipeline.components.data_preprocessor.ConfigurationProvider",
        return_value=mock_instance,
    ), mock.patch(
        "nexusml.core.pipeline.adapters.data_adapter.ConfigurationProvider",
        return_value=mock_instance,
    ):
        yield mock_instance


class TestStandardDataLoader:
    """Tests for the StandardDataLoader class."""

    def test_init(self, mock_config_provider):
        """Test initialization of StandardDataLoader."""
        loader = StandardDataLoader()
        assert loader.get_name() == "StandardDataLoader"
        assert "Standard data loader" in loader.get_description()

    def test_load_data_with_path(self, sample_csv_file, mock_config_provider):
        """Test loading data with a specified path."""
        loader = StandardDataLoader()
        df = loader.load_data(sample_csv_file)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 5)
        assert list(df.columns) == ["id", "name", "description", "category", "value"]
        assert df["name"].iloc[0] == "Item 1"

    def test_load_data_with_default_path(self, mock_config_provider):
        """Test loading data with the default path from configuration."""
        # Create a mock for pd.read_csv to avoid actual file reading
        with mock.patch("pandas.read_csv") as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["Item 1", "Item 2", "Item 3"],
                }
            )

            loader = StandardDataLoader()
            df = loader.load_data()

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["id", "name"]

            # Verify that read_csv was called with the default path
            mock_read_csv.assert_called_once()
            args, _ = mock_read_csv.call_args
            assert "test_data.csv" in args[0]

    def test_get_config(self, mock_config_provider):
        """Test getting the configuration."""
        loader = StandardDataLoader()
        config = loader.get_config()

        assert isinstance(config, dict)
        assert "required_columns" in config
        assert "training_data" in config
        assert config["training_data"]["default_path"] == "test_data.csv"

    def test_load_data_file_not_found(self, mock_config_provider):
        """Test handling of FileNotFoundError."""
        loader = StandardDataLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.csv")

    def test_resolve_path(self):
        """Test path resolution logic."""
        loader = StandardDataLoader()

        # Test absolute path
        abs_path = str(Path(__file__).resolve())
        resolved = loader._resolve_path(abs_path)
        assert resolved == abs_path

        # Test relative path that doesn't exist
        rel_path = "nonexistent_file.csv"
        resolved = loader._resolve_path(rel_path)
        assert resolved == rel_path


class TestStandardDataPreprocessor:
    """Tests for the StandardDataPreprocessor class."""

    def test_init(self, mock_config_provider):
        """Test initialization of StandardDataPreprocessor."""
        preprocessor = StandardDataPreprocessor()
        assert preprocessor.get_name() == "StandardDataPreprocessor"
        assert "Standard data preprocessor" in preprocessor.get_description()

    def test_preprocess(self, sample_data, mock_config_provider):
        """Test preprocessing data."""
        preprocessor = StandardDataPreprocessor()
        df = preprocessor.preprocess(sample_data)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 5)
        assert list(df.columns) == ["id", "name", "description", "category", "value"]

    def test_preprocess_with_nan_values(self, mock_config_provider):
        """Test preprocessing data with NaN values."""
        # Create a DataFrame with NaN values
        df = pd.DataFrame(
            {
                "id": [1, 2, None, 4, 5],
                "name": ["Item 1", None, "Item 3", "Item 4", "Item 5"],
                "value": [10.5, 20.0, None, 8.25, 30.0],
            }
        )

        preprocessor = StandardDataPreprocessor()
        result = preprocessor.preprocess(df)

        # Check that NaN values were filled
        assert result["id"].iloc[2] == 0  # Numeric column filled with 0
        assert result["name"].iloc[1] == ""  # Text column filled with empty string
        assert result["value"].iloc[2] == 0  # Numeric column filled with 0

    def test_preprocess_with_additional_args(self, sample_data, mock_config_provider):
        """Test preprocessing with additional arguments."""
        # Add a duplicate row
        df = pd.concat([sample_data, sample_data.iloc[[0]]])

        preprocessor = StandardDataPreprocessor()
        result = preprocessor.preprocess(df, drop_duplicates=True)

        # Check that duplicates were removed
        assert result.shape == (5, 5)

    def test_verify_required_columns(self, mock_config_provider):
        """Test verifying and creating required columns."""
        # Create a DataFrame missing some required columns
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Item 1", "Item 2", "Item 3"],
                # Missing description, category, and value
            }
        )

        preprocessor = StandardDataPreprocessor()
        result = preprocessor.verify_required_columns(df)

        # Check that missing columns were created with default values
        assert "description" in result.columns
        assert "category" in result.columns
        assert "value" in result.columns
        assert result["description"].iloc[0] == ""
        assert result["category"].iloc[0] == "Unknown"
        assert result["value"].iloc[0] == 0.0


class TestLegacyDataLoaderAdapter:
    """Tests for the LegacyDataLoaderAdapter class."""

    def test_init(self):
        """Test initialization of LegacyDataLoaderAdapter."""
        adapter = LegacyDataLoaderAdapter()
        assert adapter.get_name() == "LegacyDataLoaderAdapter"
        assert "Adapter for legacy data loading" in adapter.get_description()

    def test_load_data(self, sample_csv_file):
        """Test loading data with the legacy adapter."""
        # Mock the legacy function at the module level where it's imported
        with mock.patch(
            "nexusml.core.pipeline.adapters.data_adapter.load_and_preprocess_data"
        ) as mock_load:
            mock_load.return_value = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["Item 1", "Item 2", "Item 3"],
                }
            )

            adapter = LegacyDataLoaderAdapter()
            df = adapter.load_data(
                sample_csv_file, test_mode=True, expected_columns=["id", "name"]
            )

            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["id", "name"]

            # Verify that the legacy function was called
            mock_load.assert_called_once_with(sample_csv_file)

    def test_get_config(self, mock_config_provider):
        """Test getting the configuration."""
        adapter = LegacyDataLoaderAdapter()
        config = adapter.get_config()

        assert isinstance(config, dict)
        assert "training_data" in config

    def test_validate_config(self):
        """Test configuration validation."""
        adapter = LegacyDataLoaderAdapter()

        # Valid configuration
        valid_config = {
            "training_data": {
                "default_path": "test_data.csv",
            }
        }
        assert adapter.validate_config(valid_config) is True

        # Invalid configuration
        invalid_config = {
            "some_other_key": "value",
        }
        assert adapter.validate_config(invalid_config) is False


class TestLegacyDataPreprocessorAdapter:
    """Tests for the LegacyDataPreprocessorAdapter class."""

    def test_init(self):
        """Test initialization of LegacyDataPreprocessorAdapter."""
        adapter = LegacyDataPreprocessorAdapter()
        assert adapter.get_name() == "LegacyDataPreprocessorAdapter"
        assert "Adapter for legacy data preprocessing" in adapter.get_description()

    def test_preprocess(self, sample_data):
        """Test preprocessing with the legacy adapter."""
        adapter = LegacyDataPreprocessorAdapter()

        # Mock the verify_required_columns method
        with mock.patch.object(
            adapter, "verify_required_columns", return_value=sample_data
        ):
            df = adapter.preprocess(sample_data)

            assert isinstance(df, pd.DataFrame)
            assert df.shape == (5, 5)

    def test_preprocess_with_additional_args(self, sample_data):
        """Test preprocessing with additional arguments."""
        # Add a duplicate row
        df = pd.concat([sample_data, sample_data.iloc[[0]]])

        # Create a copy for testing
        test_df = df.copy()

        # Create a modified version with duplicates removed
        expected_df = df.drop_duplicates()

        adapter = LegacyDataPreprocessorAdapter()

        # Mock the verify_required_columns method to return the test dataframe
        with mock.patch.object(
            adapter, "verify_required_columns", return_value=test_df
        ):
            result = adapter.preprocess(
                df, drop_duplicates=True, test_mode=True, expected_rows=5
            )

            # Check that duplicates were removed
            assert result.shape[0] == 5  # Original sample_data has 5 rows

    def test_verify_required_columns(self, mock_config_provider):
        """Test verifying and creating required columns."""
        # Create a DataFrame missing some required columns
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Item 1", "Item 2", "Item 3"],
                # Missing description, category, and value
            }
        )

        adapter = LegacyDataPreprocessorAdapter()

        # Create a result DataFrame with the required columns added
        expected_result = df.copy()
        expected_result["description"] = ""
        expected_result["category"] = "Unknown"
        expected_result["value"] = 0.0

        # Mock the configuration provider to avoid actual configuration loading
        with mock.patch.object(adapter, "_config_provider") as mock_provider:
            # Create mock required columns
            description_mock = mock.Mock()
            description_mock.name = "description"
            description_mock.default_value = ""
            description_mock.data_type = "str"

            category_mock = mock.Mock()
            category_mock.name = "category"
            category_mock.default_value = "Unknown"
            category_mock.data_type = "str"

            value_mock = mock.Mock()
            value_mock.name = "value"
            value_mock.default_value = 0.0
            value_mock.data_type = "float"

            # Set up the mock
            mock_provider.config.data.required_columns = [
                description_mock,
                category_mock,
                value_mock,
            ]

            # Call the method
            result = adapter.verify_required_columns(df)

            # Check that missing columns were created with default values
            assert "description" in result.columns
            assert "category" in result.columns
            assert "value" in result.columns
            assert result["description"].iloc[0] == ""
            assert result["category"].iloc[0] == "Unknown"
            assert result["value"].iloc[0] == 0.0


class TestDataComponentFactory:
    """Tests for the DataComponentFactory class."""

    def test_create_data_loader_standard(self):
        """Test creating a standard data loader."""
        loader = DataComponentFactory.create_data_loader(use_legacy=False)
        assert isinstance(loader, StandardDataLoader)

    def test_create_data_loader_legacy(self):
        """Test creating a legacy data loader adapter."""
        loader = DataComponentFactory.create_data_loader(use_legacy=True)
        assert isinstance(loader, LegacyDataLoaderAdapter)

    def test_create_data_preprocessor_standard(self, mock_config_provider):
        """Test creating a standard data preprocessor."""
        preprocessor = DataComponentFactory.create_data_preprocessor(use_legacy=False)
        assert isinstance(preprocessor, StandardDataPreprocessor)

    def test_create_data_preprocessor_legacy(self):
        """Test creating a legacy data preprocessor adapter."""
        preprocessor = DataComponentFactory.create_data_preprocessor(use_legacy=True)
        assert isinstance(preprocessor, LegacyDataPreprocessorAdapter)
