"""
Unit tests for the Excel extractor module.

This module contains tests for the Excel extractor functionality in the
fca_dashboard.extractors.excel_extractor module.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fca_dashboard.config.settings import settings
from fca_dashboard.extractors.excel_extractor import (
    ExcelExtractionError,
    extract_excel_to_dataframe,
)
from fca_dashboard.utils.path_util import get_root_dir


def create_test_excel_file(data, filename="test_data.xlsx"):
    """Helper function to create a test Excel file."""
    # Create a DataFrame from the test data
    df = pd.DataFrame(data)
    
    # Create a temporary file directly instead of a directory
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save the DataFrame to the Excel file
    df.to_excel(tmp_path, index=False)
    
    # Return the path to the created file
    return tmp_path


def test_extract_excel_to_dataframe_success():
    """Test successful extraction of data from an Excel file."""
    # Test data
    test_data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"],
        "Column3": [True, False, True]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Call the function to extract data
        result_df = extract_excel_to_dataframe(excel_file)
        
        # Verify the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Verify the DataFrame has the expected columns and data
        assert list(result_df.columns) == list(test_data.keys())
        assert len(result_df) == len(test_data["Column1"])
        assert result_df["Column1"].tolist() == test_data["Column1"]
        assert result_df["Column2"].tolist() == test_data["Column2"]
        assert result_df["Column3"].tolist() == test_data["Column3"]
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


def test_extract_excel_to_dataframe_file_not_found():
    """Test extraction with a non-existent file."""
    # Use a file path that doesn't exist
    non_existent_file = "nonexistent_file.xlsx"
    
    # Verify that the function raises a FileNotFoundError
    with pytest.raises(FileNotFoundError):
        extract_excel_to_dataframe(non_existent_file)


def test_extract_excel_to_dataframe_invalid_format():
    """Test extraction with an invalid file format."""
    # Create a text file instead of an Excel file
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"This is not an Excel file")
        tmp_path = tmp.name
    
    try:
        # Verify that the function raises an ExcelExtractionError
        with pytest.raises(ExcelExtractionError):
            extract_excel_to_dataframe(tmp_path)
    finally:
        # Clean up the test file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@patch("fca_dashboard.extractors.excel_extractor.upload_file")
def test_extract_excel_to_dataframe_with_upload(mock_upload_file):
    """Test extraction with file upload."""
    # Configure the mock to return True
    mock_upload_file.return_value = True
    
    # Test data
    test_data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Call the function to extract data with upload=True
        result_df = extract_excel_to_dataframe(excel_file, upload=True)
        
        # Verify the result is a DataFrame with the expected data
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == list(test_data.keys())
        
        # Verify that upload_file was called with the correct arguments
        mock_upload_file.assert_called_once()
        args, kwargs = mock_upload_file.call_args
        assert str(args[0]) == str(excel_file)  # source file
        # The destination should be the uploads directory from settings
        uploads_dir = get_root_dir() / settings.get("file_paths.uploads_dir", "uploads")
        assert str(args[1]) == str(uploads_dir)
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


@patch("fca_dashboard.extractors.excel_extractor.upload_file")
def test_extract_excel_to_dataframe_with_sheet_name(mock_upload_file):
    """Test extraction with a specific sheet name."""
    # Configure the mock to return True
    mock_upload_file.return_value = True
    
    # Create a multi-sheet Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Create a writer to save multiple sheets
    with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
        # Sheet 1
        pd.DataFrame({
            "Sheet1Col1": [1, 2, 3],
            "Sheet1Col2": ["A", "B", "C"]
        }).to_excel(writer, sheet_name="Sheet1", index=False)
        
        # Sheet 2
        pd.DataFrame({
            "Sheet2Col1": [4, 5, 6],
            "Sheet2Col2": ["D", "E", "F"]
        }).to_excel(writer, sheet_name="Sheet2", index=False)
    
    try:
        # Extract data from Sheet2
        result_df = extract_excel_to_dataframe(tmp_path, sheet_name="Sheet2")
        
        # Verify the result contains data from Sheet2
        assert isinstance(result_df, pd.DataFrame)
        assert "Sheet2Col1" in result_df.columns
        assert "Sheet2Col2" in result_df.columns
        assert result_df["Sheet2Col1"].tolist() == [4, 5, 6]
        assert result_df["Sheet2Col2"].tolist() == ["D", "E", "F"]
    finally:
        # Clean up the test file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@patch("fca_dashboard.extractors.excel_extractor.get_logger")
def test_extract_excel_to_dataframe_with_logging(mock_get_logger):
    """Test that extraction operations are properly logged."""
    # Setup mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Test data
    test_data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Extract data
        extract_excel_to_dataframe(excel_file)
        
        # Verify logging calls
        assert mock_logger.info.call_count >= 1
        # Check that the log message contains the filename
        assert any(os.path.basename(excel_file) in str(args) for args, _ in mock_logger.info.call_args_list)
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


def test_extract_excel_to_dataframe_with_custom_columns():
    """Test extraction with custom column selection."""
    # Test data with multiple columns
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35],
        "City": ["New York", "Los Angeles", "Chicago"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Extract only specific columns
        columns_to_extract = ["ID", "Name"]
        result_df = extract_excel_to_dataframe(excel_file, columns=columns_to_extract)
        
        # Verify only the specified columns were extracted
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == columns_to_extract
        assert len(result_df.columns) == 2
        assert "Age" not in result_df.columns
        assert "City" not in result_df.columns
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


def test_extract_excel_to_dataframe_missing_columns():
    """Test extraction with non-existent columns."""
    # Test data
    test_data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Try to extract non-existent columns
        non_existent_columns = ["Column1", "NonExistentColumn"]
        
        # Verify that the function raises an ExcelExtractionError
        with pytest.raises(ExcelExtractionError) as excinfo:
            extract_excel_to_dataframe(excel_file, columns=non_existent_columns)
        
        # Verify the error message
        assert "Columns not found in Excel file" in str(excinfo.value)
        assert "NonExistentColumn" in str(excinfo.value)
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


def test_extract_excel_to_dataframe_pandas_error():
    """Test handling of pandas-specific errors."""
    # Create a corrupted Excel file (just a text file with .xlsx extension)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(b"This is not a valid Excel file format")
        corrupted_file = tmp.name
    
    try:
        # Try to extract data from the corrupted file
        with pytest.raises(ExcelExtractionError) as excinfo:
            extract_excel_to_dataframe(corrupted_file)
        
        # Verify the error message
        assert "Error parsing Excel file" in str(excinfo.value) or "Error extracting data from Excel file" in str(excinfo.value)
    finally:
        # Clean up the test file
        if os.path.exists(corrupted_file):
            os.unlink(corrupted_file)


def test_extract_excel_to_dataframe_empty_file():
    """Test extraction from an empty Excel file."""
    # Create an empty DataFrame
    df = pd.DataFrame()
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        empty_file = tmp.name
    
    # Save the empty DataFrame to an Excel file
    df.to_excel(empty_file, index=False)
    
    try:
        # Extract data from the empty file
        result_df = extract_excel_to_dataframe(empty_file)
        
        # Verify that the result is an empty DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0  # DataFrame should have 0 rows
    finally:
        # Clean up the test file
        if os.path.exists(empty_file):
            os.unlink(empty_file)


@patch("pandas.read_excel")
def test_extract_excel_to_dataframe_generic_error(mock_read_excel):
    """Test handling of generic errors during extraction."""
    # Configure the mock to raise a generic exception
    mock_read_excel.side_effect = RuntimeError("Simulated generic error")
    
    # Create a test file path
    test_file = "test_file.xlsx"
    
    # Mock the file existence check
    with patch("pathlib.Path.is_file", return_value=True):
        # Try to extract data
        with pytest.raises(ExcelExtractionError) as excinfo:
            extract_excel_to_dataframe(test_file)
        
        # Verify the error message
        assert "Error extracting data from Excel file" in str(excinfo.value)
        assert "Simulated generic error" in str(excinfo.value)


@patch("fca_dashboard.extractors.excel_extractor.upload_file")
def test_extract_excel_to_dataframe_with_custom_target_filename(mock_upload_file):
    """Test extraction with a custom target filename for upload."""
    # Configure the mock to return True
    mock_upload_file.return_value = True
    
    # Test data
    test_data = {
        "Column1": [1, 2, 3],
        "Column2": ["A", "B", "C"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    # Custom target filename
    custom_filename = "custom_target.xlsx"
    
    try:
        # Call the function with a custom target filename
        result_df = extract_excel_to_dataframe(excel_file, upload=True, target_filename=custom_filename)
        
        # Verify the result is a DataFrame with the expected data
        assert isinstance(result_df, pd.DataFrame)
        
        # Verify that upload_file was called with the correct arguments
        mock_upload_file.assert_called_once()
        args, kwargs = mock_upload_file.call_args
        assert str(args[0]) == str(excel_file)  # source file
        # The third argument should be the custom filename
        assert args[2] == custom_filename
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)


@patch("pandas.read_excel")
def test_extract_excel_to_dataframe_invalid_sheet_name(mock_read_excel):
    """Test extraction with an invalid sheet name or index."""
    # Configure the mock to raise a ValueError for invalid sheet
    mock_read_excel.side_effect = ValueError("No sheet named 'NonExistentSheet'")
    
    # Create a test file path
    test_file = "test_file.xlsx"
    
    # Mock the file existence check
    with patch("pathlib.Path.is_file", return_value=True):
        # Try to extract data with an invalid sheet name
        with pytest.raises(ExcelExtractionError) as excinfo:
            extract_excel_to_dataframe(test_file, sheet_name="NonExistentSheet")
        
        # Verify the error message
        assert "Error extracting data from Excel file" in str(excinfo.value)
        assert "No sheet named 'NonExistentSheet'" in str(excinfo.value)


@patch("fca_dashboard.extractors.excel_extractor.get_logger")
def test_extract_excel_to_dataframe_row_count_logging(mock_get_logger):
    """Test that the correct row count is logged."""
    # Setup mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Test data with a specific number of rows
    row_count = 5
    test_data = {
        "Column1": list(range(row_count)),
        "Column2": ["A", "B", "C", "D", "E"]
    }
    
    # Create a test Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Extract data
        extract_excel_to_dataframe(excel_file)
        
        # Verify logging calls
        success_log_calls = [
            call_args for call_args, _ in mock_logger.info.call_args_list
            if "Successfully extracted" in str(call_args)
        ]
        
        # There should be at least one success log message
        assert len(success_log_calls) >= 1
        
        # The success log message should contain the correct row count
        success_log_message = str(success_log_calls[0])
        assert f"Successfully extracted {row_count} rows" in success_log_message
    finally:
        # Clean up the test file
        if os.path.exists(excel_file):
            os.unlink(excel_file)