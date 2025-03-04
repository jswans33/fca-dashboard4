"""
Unit tests for the Excel extractor with dependency injection.

This module contains tests for the ExcelExtractor class in the
fca_dashboard.extractors.excel_extractor module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fca_dashboard.extractors.base_extractor import ExtractionError, extractor_factory
from fca_dashboard.extractors.excel_extractor import ExcelExtractor, extract_excel_to_dataframe


def create_test_excel_file(data, filename="test_data.xlsx"):
    """Helper function to create a test Excel file."""
    # Create a DataFrame from the test data
    df = pd.DataFrame(data)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save the DataFrame to the Excel file
    df.to_excel(tmp_path, index=False)
    
    # Return the path to the created file
    return tmp_path


def create_test_csv_file(data, filename="test_data.csv"):
    """Helper function to create a test CSV file."""
    # Create a DataFrame from the test data
    df = pd.DataFrame(data)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save the DataFrame to the CSV file
    df.to_csv(tmp_path, index=False)
    
    # Return the path to the created file
    return tmp_path


class TestExcelExtractor:
    """Tests for the ExcelExtractor class."""
    
    def test_can_extract(self):
        """Test the can_extract method."""
        # Create test files
        xlsx_data = {"Column1": [1, 2, 3]}
        xlsx_file = create_test_excel_file(xlsx_data)
        
        csv_data = {"Column1": [1, 2, 3]}
        csv_file = create_test_csv_file(csv_data)
        
        # Create a text file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is not an Excel file")
            txt_file = tmp.name
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Test XLSX file
            assert extractor.can_extract(xlsx_file) is True
            
            # Test CSV file
            assert extractor.can_extract(csv_file) is True
            
            # Test non-Excel file
            assert extractor.can_extract(txt_file) is False
            
            # Test non-existent file
            with pytest.raises(FileNotFoundError):
                extractor.can_extract("nonexistent_file.xlsx")
        finally:
            # Clean up test files
            for file in [xlsx_file, csv_file, txt_file]:
                if os.path.exists(file):
                    try:
                        os.unlink(file)
                    except PermissionError:
                        # On Windows, files might still be in use by another process
                        print(f"Warning: Could not delete file {file} - it may be in use by another process")
    
    def test_extract_xlsx(self):
        """Test extracting data from an XLSX file."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Extract data
            df = extractor.extract(excel_file)
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
            assert df["ID"].tolist() == [1, 2, 3]
            assert df["Name"].tolist() == ["Alice", "Bob", "Charlie"]
            assert df["Age"].tolist() == [25, 30, 35]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_extract_csv(self):
        """Test extracting data from a CSV file."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create CSV file
        csv_file = create_test_csv_file(test_data)
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Extract data
            df = extractor.extract(csv_file)
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
            assert df["ID"].tolist() == [1, 2, 3]
            assert df["Name"].tolist() == ["Alice", "Bob", "Charlie"]
            assert df["Age"].tolist() == [25, 30, 35]
        finally:
            # Clean up test file
            if os.path.exists(csv_file):
                try:
                    os.unlink(csv_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {csv_file} - it may be in use by another process")
    
    def test_extract_with_columns(self):
        """Test extracting specific columns."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Extract specific columns
            df = extractor.extract(excel_file, columns=["ID", "Name"])
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name"]
            assert "Age" not in df.columns
            assert df["ID"].tolist() == [1, 2, 3]
            assert df["Name"].tolist() == ["Alice", "Bob", "Charlie"]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_extract_with_missing_columns(self):
        """Test extracting with missing columns."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Try to extract non-existent columns
            with pytest.raises(ExtractionError) as excinfo:
                extractor.extract(excel_file, columns=["ID", "NonExistentColumn"])
            
            # Verify the error message
            assert "Columns not found in Excel file" in str(excinfo.value)
            assert "NonExistentColumn" in str(excinfo.value)
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_extract_with_upload(self):
        """Test extracting with file upload."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create a mock upload service
            mock_upload_service = MagicMock()
            
            # Create an extractor with the mock upload service
            extractor = ExcelExtractor(upload_service=mock_upload_service)
            
            # Extract data with upload
            df = extractor.extract(excel_file, upload=True)
            
            # Verify the upload service was called
            mock_upload_service.assert_called_once()
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_extract_with_target_filename(self):
        """Test extracting with a target filename for upload."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create a mock upload service
            mock_upload_service = MagicMock()
            
            # Create an extractor with the mock upload service
            extractor = ExcelExtractor(upload_service=mock_upload_service)
            
            # Extract data with upload and target filename
            target_filename = "custom_name.xlsx"
            df = extractor.extract(excel_file, upload=True, target_filename=target_filename)
            
            # Verify the upload service was called with the target filename
            mock_upload_service.assert_called_once()
            args, kwargs = mock_upload_service.call_args
            assert args[2] == target_filename  # Third argument is target_filename
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_extract_file_not_found(self):
        """Test extracting with a non-existent file."""
        # Create an extractor
        extractor = ExcelExtractor()
        
        # Try to extract from a non-existent file
        with pytest.raises(FileNotFoundError) as excinfo:
            extractor.extract("nonexistent_file.xlsx")
        
        # Verify the error message
        assert "Source file not found" in str(excinfo.value)
    
    def test_extract_with_pandas_error(self):
        """Test extracting with a pandas error."""
        # Create a corrupted Excel file (just a text file with .xlsx extension)
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            tmp.write(b"This is not a valid Excel file format")
            corrupted_file = tmp.name
        
        try:
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Try to extract from the corrupted file
            with pytest.raises(ExtractionError) as excinfo:
                extractor.extract(corrupted_file)
            
            # Verify the error message
            assert "Error extracting data from Excel file" in str(excinfo.value)
        finally:
            # Clean up test file
            if os.path.exists(corrupted_file):
                try:
                    os.unlink(corrupted_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {corrupted_file} - it may be in use by another process")
    
    def test_extract_and_save(self):
        """Test extracting and saving data."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Create a temporary output file
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_out:
                output_file = tmp_out.name
            
            # Create an extractor
            extractor = ExcelExtractor()
            
            # Extract and save data
            df = extractor.extract_and_save(excel_file, output_file, output_format="csv")
            
            # Verify the output file was created
            assert os.path.exists(output_file)
            
            # Verify the saved data
            saved_df = pd.read_csv(output_file)
            assert len(saved_df) == 3
            assert list(saved_df.columns) == ["ID", "Name", "Age"]
            
            # Verify the returned DataFrame
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
        finally:
            # Clean up test files
            for file in [excel_file, output_file]:
                if os.path.exists(file):
                    try:
                        os.unlink(file)
                    except PermissionError:
                        # On Windows, files might still be in use by another process
                        print(f"Warning: Could not delete file {file} - it may be in use by another process")
    
    def test_factory_registration(self):
        """Test that the Excel extractor is registered with the factory."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Get an extractor from the factory
            extractor = extractor_factory.get_extractor(excel_file)
            
            # Verify that an extractor was found
            assert extractor is not None
            assert isinstance(extractor, ExcelExtractor)
            
            # Extract data using the factory
            df = extractor_factory.extract(excel_file)
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")
    
    def test_backward_compatibility(self):
        """Test backward compatibility with the extract_excel_to_dataframe function."""
        # Create test data
        test_data = {
            "ID": [1, 2, 3],
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35]
        }
        
        # Create Excel file
        excel_file = create_test_excel_file(test_data)
        
        try:
            # Extract data using the function
            df = extract_excel_to_dataframe(excel_file)
            
            # Verify the extracted data
            assert len(df) == 3
            assert list(df.columns) == ["ID", "Name", "Age"]
            assert df["ID"].tolist() == [1, 2, 3]
            assert df["Name"].tolist() == ["Alice", "Bob", "Charlie"]
            assert df["Age"].tolist() == [25, 30, 35]
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                try:
                    os.unlink(excel_file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")