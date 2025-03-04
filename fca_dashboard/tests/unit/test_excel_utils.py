"""
Unit tests for the Excel utility module.

This module contains tests for the Excel utility functions in the
fca_dashboard.utils.excel package.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.excel import (
    ExcelUtilError,
    convert_excel_to_csv,
    get_column_names,
    get_excel_file_type,
    get_sheet_names,
    is_excel_file,
    is_valid_excel_file,
    merge_excel_files,
    validate_columns_exist,
)


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


def test_get_excel_file_type():
    """Test getting the file type of Excel files."""
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
        # Test XLSX file
        assert get_excel_file_type(xlsx_file) == "xlsx"
        
        # Test CSV file
        assert get_excel_file_type(csv_file) == "csv"
        
        # Test non-Excel file
        assert get_excel_file_type(txt_file) is None
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            get_excel_file_type("nonexistent_file.xlsx")
    finally:
        # Clean up test files
        for file in [xlsx_file, csv_file, txt_file]:
            if os.path.exists(file):
                os.unlink(file)


def test_is_excel_file():
    """Test checking if a file is an Excel file."""
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
        # Test XLSX file
        assert is_excel_file(xlsx_file) is True
        
        # Test CSV file (should be considered an Excel file)
        assert is_excel_file(csv_file) is True
        
        # Test non-Excel file
        assert is_excel_file(txt_file) is False
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            is_excel_file("nonexistent_file.xlsx")
    finally:
        # Clean up test files
        for file in [xlsx_file, csv_file, txt_file]:
            if os.path.exists(file):
                os.unlink(file)


def test_is_valid_excel_file():
    """Test validating Excel files."""
    # Create a valid Excel file
    valid_data = {"Column1": [1, 2, 3]}
    valid_file = create_test_excel_file(valid_data)
    
    # Create a corrupted Excel file (just a text file with .xlsx extension)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(b"This is not a valid Excel file format")
        corrupted_file = tmp.name
    
    try:
        # Test valid Excel file
        assert is_valid_excel_file(valid_file) is True
        
        # Test corrupted Excel file
        assert is_valid_excel_file(corrupted_file) is False
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            is_valid_excel_file("nonexistent_file.xlsx")
    finally:
        # Clean up test files
        for file in [valid_file, corrupted_file]:
            if os.path.exists(file):
                os.unlink(file)


def test_get_sheet_names():
    """Test getting sheet names from an Excel file."""
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
    
    # Create a CSV file (has no sheets)
    csv_data = {"Column1": [1, 2, 3]}
    csv_file = create_test_csv_file(csv_data)
    
    try:
        # Test multi-sheet Excel file
        sheet_names = get_sheet_names(tmp_path)
        assert isinstance(sheet_names, list)
        assert len(sheet_names) == 2
        assert "Sheet1" in sheet_names
        assert "Sheet2" in sheet_names
        
        # Test CSV file (should return a list with a single None element)
        csv_sheet_names = get_sheet_names(csv_file)
        assert isinstance(csv_sheet_names, list)
        assert len(csv_sheet_names) == 1
        assert csv_sheet_names[0] == 0  # Default sheet name for CSV is 0
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            get_sheet_names("nonexistent_file.xlsx")
    finally:
        # Clean up test files
        for file in [tmp_path, csv_file]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")


def test_get_column_names():
    """Test getting column names from an Excel file."""
    # Create test data with specific columns
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    }
    
    # Create Excel file
    excel_file = create_test_excel_file(test_data)
    
    # Create a multi-sheet Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        multi_sheet_file = tmp.name
    
    # Create a writer to save multiple sheets with different columns
    with pd.ExcelWriter(multi_sheet_file, engine="openpyxl") as writer:
        # Sheet 1
        pd.DataFrame({
            "Sheet1Col1": [1, 2, 3],
            "Sheet1Col2": ["A", "B", "C"]
        }).to_excel(writer, sheet_name="Sheet1", index=False)
        
        # Sheet 2
        pd.DataFrame({
            "Sheet2Col1": [4, 5, 6],
            "Sheet2Col2": ["D", "E", "F"],
            "Sheet2Col3": [True, False, True]
        }).to_excel(writer, sheet_name="Sheet2", index=False)
    
    try:
        # Test single-sheet Excel file
        columns = get_column_names(excel_file)
        assert isinstance(columns, list)
        assert len(columns) == 3
        assert "ID" in columns
        assert "Name" in columns
        assert "Age" in columns
        
        # Test multi-sheet Excel file with specific sheet
        sheet1_columns = get_column_names(multi_sheet_file, sheet_name="Sheet1")
        assert isinstance(sheet1_columns, list)
        assert len(sheet1_columns) == 2
        assert "Sheet1Col1" in sheet1_columns
        assert "Sheet1Col2" in sheet1_columns
        
        sheet2_columns = get_column_names(multi_sheet_file, sheet_name="Sheet2")
        assert isinstance(sheet2_columns, list)
        assert len(sheet2_columns) == 3
        assert "Sheet2Col1" in sheet2_columns
        assert "Sheet2Col2" in sheet2_columns
        assert "Sheet2Col3" in sheet2_columns
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            get_column_names("nonexistent_file.xlsx")
        
        # Test non-existent sheet
        with pytest.raises(ExcelUtilError):
            get_column_names(multi_sheet_file, sheet_name="NonExistentSheet")
    finally:
        # Clean up test files
        for file in [excel_file, multi_sheet_file]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")


def test_validate_columns_exist():
    """Test validating that columns exist in an Excel file."""
    # Create test data with specific columns
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    }
    
    # Create Excel file
    excel_file = create_test_excel_file(test_data)
    
    try:
        # Test with all columns existing
        assert validate_columns_exist(excel_file, ["ID", "Name", "Age"]) is True
        
        # Test with some columns existing
        assert validate_columns_exist(excel_file, ["ID", "Name"]) is True
        
        # Test with non-existent columns
        assert validate_columns_exist(excel_file, ["ID", "NonExistentColumn"]) is False
        
        # Test with empty columns list
        assert validate_columns_exist(excel_file, []) is True
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            validate_columns_exist("nonexistent_file.xlsx", ["ID"])
    finally:
        # Clean up test file
        if os.path.exists(excel_file):
            try:
                os.unlink(excel_file)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {excel_file} - it may be in use by another process")


def test_convert_excel_to_csv():
    """Test converting Excel file to CSV."""
    # Create test data
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    }
    
    # Create Excel file
    excel_file = create_test_excel_file(test_data)
    
    # Create a multi-sheet Excel file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        multi_sheet_file = tmp.name
    
    # Create a writer to save multiple sheets
    with pd.ExcelWriter(multi_sheet_file, engine="openpyxl") as writer:
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
        # Create temporary output files
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_out:
            csv_output = tmp_out.name
        
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_out2:
            csv_output2 = tmp_out2.name
        
        # Test converting single-sheet Excel file to CSV
        convert_excel_to_csv(excel_file, csv_output)
        
        # Verify the CSV file was created and contains the correct data
        assert os.path.exists(csv_output)
        csv_df = pd.read_csv(csv_output)
        assert len(csv_df) == 3
        assert list(csv_df.columns) == ["ID", "Name", "Age"]
        
        # Test converting multi-sheet Excel file to CSV with specific sheet
        convert_excel_to_csv(multi_sheet_file, csv_output2, sheet_name="Sheet2")
        
        # Verify the CSV file was created and contains the correct data
        assert os.path.exists(csv_output2)
        csv_df2 = pd.read_csv(csv_output2)
        assert len(csv_df2) == 3
        assert list(csv_df2.columns) == ["Sheet2Col1", "Sheet2Col2"]
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            convert_excel_to_csv("nonexistent_file.xlsx", "output.csv")
        
        # Test non-existent sheet
        with pytest.raises(ExcelUtilError):
            convert_excel_to_csv(multi_sheet_file, "output.csv", sheet_name="NonExistentSheet")
    finally:
        # Clean up test files
        for file in [excel_file, multi_sheet_file, csv_output, csv_output2]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")


def test_merge_excel_files():
    """Test merging multiple Excel files."""
    # Create test data for first file
    data1 = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"]
    }
    
    # Create test data for second file
    data2 = {
        "ID": [4, 5, 6],
        "Name": ["David", "Eve", "Frank"]
    }
    
    # Create Excel files
    file1 = create_test_excel_file(data1)
    file2 = create_test_excel_file(data2)
    
    # Create a file with different columns
    data3 = {
        "Different": [7, 8, 9],
        "Columns": ["G", "H", "I"]
    }
    file3 = create_test_excel_file(data3)
    
    try:
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_out:
            output_file = tmp_out.name
        
        # Test merging files with same columns
        merge_excel_files([file1, file2], output_file)
        
        # Verify the merged file was created and contains the correct data
        assert os.path.exists(output_file)
        merged_df = pd.read_excel(output_file)
        assert len(merged_df) == 6  # 3 rows from each file
        assert list(merged_df.columns) == ["ID", "Name"]
        
        # Test merging files with different columns
        with pytest.raises(ExcelUtilError):
            merge_excel_files([file1, file3], "output2.xlsx")
        
        # Test with empty file list
        with pytest.raises(ValueError):
            merge_excel_files([], "output3.xlsx")
        
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            merge_excel_files([file1, "nonexistent_file.xlsx"], "output4.xlsx")
    finally:
        # Clean up test files
        for file in [file1, file2, file3, output_file]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")