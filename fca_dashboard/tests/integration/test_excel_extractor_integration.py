"""
Integration tests for the Excel extractor module.

This module contains integration tests for the Excel extractor functionality,
testing its interaction with other components of the system.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.excel_extractor import extract_excel_to_dataframe
from fca_dashboard.utils.path_util import get_root_dir


@pytest.fixture
def sample_excel_file():
    """Fixture to create a sample Excel file for testing."""
    # Create test data
    data = {
        "ID": [1, 2, 3],
        "Name": ["Product A", "Product B", "Product C"],
        "Price": [10.99, 20.50, 15.75],
        "InStock": [True, False, True]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save the DataFrame to the Excel file
    df.to_excel(tmp_path, index=False)
    
    yield tmp_path
    
    # Clean up after the test
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


def test_extract_and_upload_integration(sample_excel_file):
    """Test the integration of Excel extraction with file upload."""
    # Get the uploads directory from settings
    uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
    
    # Ensure the uploads directory exists
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Extract data and upload the file
    df = extract_excel_to_dataframe(sample_excel_file, upload=True)
    
    # Verify the DataFrame contains the expected data
    assert isinstance(df, pd.DataFrame)
    assert "ID" in df.columns
    assert "Name" in df.columns
    assert "Price" in df.columns
    assert "InStock" in df.columns
    assert len(df) == 3
    
    # Verify the file was uploaded to the uploads directory
    uploaded_files = list(uploads_dir.glob("*.xlsx"))
    assert len(uploaded_files) >= 1
    
    # Clean up uploaded files after the test
    for file in uploaded_files:
        if file.name == Path(sample_excel_file).name or file.stem.startswith(Path(sample_excel_file).stem):
            os.unlink(file)


def test_extract_with_settings_integration():
    """Test extraction using settings from the configuration."""
    # Create test data
    data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"]
    }
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Get the extracts directory from settings
    extracts_dir = get_root_dir() / settings.get("file_paths.extracts_dir", "extracts")
    
    # Ensure the extracts directory exists
    os.makedirs(extracts_dir, exist_ok=True)
    
    # Create an Excel file in the extracts directory
    test_file = extracts_dir / "test_integration.xlsx"
    df.to_excel(test_file, index=False)
    
    try:
        # Extract data from the file
        result_df = extract_excel_to_dataframe(test_file)
        
        # Verify the DataFrame contains the expected data
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == list(data.keys())
        assert len(result_df) == len(data["Column1"])
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            os.unlink(test_file)