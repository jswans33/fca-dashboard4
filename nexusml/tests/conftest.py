"""
Pytest configuration for NexusML tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the parent directory to sys.path to allow importing nexusml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import DI container components
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.eav_manager import EAVManager
from nexusml.core.feature_engineering import GenericFeatureEngineer

# Initialize the container at module import time
_container_initialized = False


def _initialize_container():
    """Initialize the DI container with all required dependencies."""
    global _container_initialized

    if _container_initialized:
        return

    # Get the container provider
    provider = ContainerProvider()

    # Reset the container to ensure a clean state
    provider.reset()

    # Register the EAVManager
    provider.register_implementation(EAVManager, EAVManager, singleton=True)

    # Create a default config path for GenericFeatureEngineer
    config_path = str(
        Path(__file__).resolve().parent / "fixtures" / "feature_config.yml"
    )

    # Create the fixtures directory if it doesn't exist
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    # Create a minimal feature config file if it doesn't exist
    if not Path(config_path).exists():
        with open(config_path, "w") as f:
            f.write(
                """
# Minimal feature configuration for testing
column_mappings:
  - source: "Asset Category"
    target: "Equipment_Category"
hierarchies:
  - new_col: "Equipment_Type"
    parents: ["Asset Category", "Equip Name ID"]
    separator: "-"
  - new_col: "System_Subtype"
    parents: ["Precon System", "Operations System"]
    separator: "-"
            """
            )

    # Register the GenericFeatureEngineer
    provider.register_implementation(
        GenericFeatureEngineer, GenericFeatureEngineer, singleton=True
    )

    # Create and register an instance with the config path
    eav_manager = provider.container.resolve(EAVManager)
    feature_engineer = GenericFeatureEngineer(
        config_path=config_path, eav_manager=eav_manager
    )
    provider.register_instance(GenericFeatureEngineer, feature_engineer)

    _container_initialized = True


# Initialize the container at module import time
_initialize_container()


@pytest.fixture(scope="function", autouse=True)
def setup_di_container():
    """
    Fixture that ensures the DI container is properly set up for each test.
    This fixture runs automatically before each test function.
    """
    # Initialize the container if needed
    _initialize_container()

    # Get the container provider
    provider = ContainerProvider()

    yield provider.container


@pytest.fixture
def sample_data_path():
    """
    Fixture that provides the path to sample data for testing.

    Returns:
        str: Path to sample data file
    """
    return str(
        Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv"
    )


@pytest.fixture
def sample_description():
    """
    Fixture that provides a sample equipment description for testing.

    Returns:
        str: Sample equipment description
    """
    return "Heat Exchanger for Chilled Water system with Plate and Frame design"


@pytest.fixture
def sample_service_life():
    """
    Fixture that provides a sample service life value for testing.

    Returns:
        float: Sample service life value
    """
    return 20.0


@pytest.fixture
def mock_feature_engineer():
    """
    Fixture that provides a mock GenericFeatureEngineer for testing.

    This fixture creates a GenericFeatureEngineer with a minimal configuration
    that can be used in tests without relying on the DI container.

    Returns:
        GenericFeatureEngineer: A mock feature engineer instance
    """
    # Create a minimal config path
    config_path = str(
        Path(__file__).resolve().parent / "fixtures" / "feature_config.yml"
    )

    # Create the fixtures directory if it doesn't exist
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    # Create a minimal feature config file if it doesn't exist
    if not Path(config_path).exists():
        with open(config_path, "w") as f:
            f.write(
                """
# Minimal feature configuration for testing
column_mappings:
  - source: "Asset Category"
    target: "Equipment_Category"
hierarchies:
  - new_col: "Equipment_Type"
    parents: ["Asset Category", "Equip Name ID"]
    separator: "-"
  - new_col: "System_Subtype"
    parents: ["Precon System", "Operations System"]
    separator: "-"
            """
            )

    # Create an EAVManager instance
    eav_manager = EAVManager()

    # Create and return a GenericFeatureEngineer instance
    return GenericFeatureEngineer(config_path=config_path, eav_manager=eav_manager)
