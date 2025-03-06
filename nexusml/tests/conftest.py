"""
Pytest configuration for NexusML tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the parent directory to sys.path to allow importing nexusml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


@pytest.fixture
def sample_data_path():
    """
    Fixture that provides the path to sample data for testing.
    
    Returns:
        str: Path to sample data file
    """
    return str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")


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