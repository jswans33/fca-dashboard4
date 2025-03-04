"""
Integration tests for the Excel extractor with dependency injection.

This module contains integration tests for the Excel extractor with dependency injection,
testing its interaction with other components of the system.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.base_extractor import extractor_factory
from fca_dashboard.extractors.excel_extractor import ExcelExtractor
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
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # On Windows, files might still be in use by another process
            print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


@pytest.fixture
def sample_csv_file():
    """Fixture to create a sample CSV file for testing."""
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
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save the DataFrame to the CSV file
    df.to_csv(tmp_path, index=False)
    
    yield tmp_path
    
    # Clean up after the test
    if os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except PermissionError:
            # On Windows, files might still be in use by another process
            print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


def test_extractor_factory_integration(sample_excel_file, sample_csv_file):
    """Test the integration of the extractor factory with the Excel extractor."""
    # Test with Excel file
    excel_extractor = extractor_factory.get_extractor(sample_excel_file)
    assert excel_extractor is not None
    assert isinstance(excel_extractor, ExcelExtractor)
    
    # Test with CSV file
    csv_extractor = extractor_factory.get_extractor(sample_csv_file)
    assert csv_extractor is not None
    assert isinstance(csv_extractor, ExcelExtractor)
    
    # Test with a non-existent file type
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"This is not an Excel file")
        txt_file = tmp.name
    
    try:
        txt_extractor = extractor_factory.get_extractor(txt_file)
        assert txt_extractor is None
    finally:
        if os.path.exists(txt_file):
            try:
                os.unlink(txt_file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {txt_file} - it may be in use by another process")


def test_excel_extractor_with_upload_integration(sample_excel_file):
    """Test the integration of the Excel extractor with file upload."""
    # Get the uploads directory from settings
    uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
    
    # Ensure the uploads directory exists
    os.makedirs(uploads_dir, exist_ok=True)
    
    # Create an extractor
    extractor = ExcelExtractor()
    
    # Extract data and upload the file
    df = extractor.extract(sample_excel_file, upload=True)
    
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
            try:
                os.unlink(file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {file} - it may be in use by another process")


def test_excel_extractor_with_settings_integration():
    """Test the Excel extractor using settings from the configuration."""
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
    test_file = extracts_dir / "test_di_integration.xlsx"
    df.to_excel(test_file, index=False)
    
    try:
        # Create an extractor
        extractor = ExcelExtractor()
        
        # Extract data from the file
        result_df = extractor.extract(test_file)
        
        # Verify the DataFrame contains the expected data
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == list(data.keys())
        assert len(result_df) == len(data["Column1"])
    finally:
        # Clean up the test file
        if os.path.exists(test_file):
            try:
                os.unlink(test_file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {test_file} - it may be in use by another process")


def test_extract_and_save_integration(sample_excel_file):
    """Test the integration of extract_and_save with the file system."""
    # Create an extractor
    extractor = ExcelExtractor()
    
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_out:
        output_file = tmp_out.name
    
    try:
        # Extract and save data
        df = extractor.extract_and_save(sample_excel_file, output_file, output_format="csv")
        
        # Verify the output file was created
        assert os.path.exists(output_file)
        
        # Verify the saved data
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 3
        assert "ID" in saved_df.columns
        assert "Name" in saved_df.columns
        assert "Price" in saved_df.columns
        assert "InStock" in saved_df.columns
        
        # Verify the returned DataFrame
        assert len(df) == 3
        assert "ID" in df.columns
        assert "Name" in df.columns
        assert "Price" in df.columns
        assert "InStock" in df.columns
    finally:
        # Clean up the output file
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {output_file} - it may be in use by another process")


def test_factory_extract_integration(sample_excel_file):
    """Test the integration of the factory's extract method."""
    # Extract data using the factory
    df = extractor_factory.extract(sample_excel_file)
    
    # Verify the DataFrame contains the expected data
    assert isinstance(df, pd.DataFrame)
    assert "ID" in df.columns
    assert "Name" in df.columns
    assert "Price" in df.columns
    assert "InStock" in df.columns
    assert len(df) == 3


def test_factory_extract_and_save_integration(sample_excel_file):
    """Test the integration of the factory's extract_and_save method."""
    # Create a temporary output file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_out:
        output_file = tmp_out.name
    
    try:
        # Extract and save data using the factory
        df = extractor_factory.extract_and_save(sample_excel_file, output_file, output_format="csv")
        
        # Verify the output file was created
        assert os.path.exists(output_file)
        
        # Verify the saved data
        saved_df = pd.read_csv(output_file)
        assert len(saved_df) == 3
        assert "ID" in saved_df.columns
        assert "Name" in saved_df.columns
        assert "Price" in saved_df.columns
        assert "InStock" in saved_df.columns
        
        # Verify the returned DataFrame
        assert len(df) == 3
        assert "ID" in df.columns
        assert "Name" in df.columns
        assert "Price" in df.columns
        assert "InStock" in df.columns
    finally:
        # Clean up the output file
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {output_file} - it may be in use by another process")