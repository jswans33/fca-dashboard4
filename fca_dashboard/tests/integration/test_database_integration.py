"""
Integration tests for the database utilities.

This module contains integration tests for the database utilities,
particularly focusing on their integration with Excel utilities.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database import (
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.excel import (
    convert_excel_to_csv,
    save_excel_to_database,  # Bridge function
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


def test_excel_to_database_integration():
    """Test the integration between Excel utilities and database utilities."""
    # Create test data
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    }
    
    # Create Excel file
    excel_file = create_test_excel_file(test_data)
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{db_path}"
        
        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file)
        
        # Save the DataFrame to the database using the bridge function
        save_excel_to_database(
            df=df,
            table_name="test_table",
            connection_string=connection_string,
            if_exists="replace"
        )
        
        # Verify the data was saved correctly
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test_table"))
            rows = result.fetchall()
            
            # Check the number of rows
            assert len(rows) == 3
            
            # Check the values in the first row
            assert rows[0][0] == 1
            assert rows[0][1] == "Alice"
            assert rows[0][2] == 25
        
        # Get the schema of the table
        schema = get_table_schema(connection_string, "test_table")
        
        # Check that the schema contains the column names
        assert "ID" in schema
        assert "Name" in schema
        assert "Age" in schema
    finally:
        # Clean up test files
        for file in [excel_file, db_path]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")


def test_excel_csv_database_pipeline():
    """Test a complete pipeline from Excel to CSV to database."""
    # Create test data
    test_data = {
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    }
    
    # Create Excel file
    excel_file = create_test_excel_file(test_data)
    
    # Create temporary output files
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        csv_path = tmp_csv.name
    
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name
    
    try:
        # Step 1: Convert Excel to CSV
        convert_excel_to_csv(excel_file, csv_path)
        
        # Verify the CSV file was created and contains the correct data
        assert os.path.exists(csv_path)
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == 3
        assert list(csv_df.columns) == ["ID", "Name", "Age"]
        
        # Step 2: Save CSV data to database
        connection_string = f"sqlite:///{db_path}"
        save_dataframe_to_database(
            df=csv_df,
            table_name="test_table",
            connection_string=connection_string,
            if_exists="replace"
        )
        
        # Verify the data was saved correctly
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test_table"))
            rows = result.fetchall()
            
            # Check the number of rows
            assert len(rows) == 3
            
            # Check the values in the first row
            assert rows[0][0] == 1
            assert rows[0][1] == "Alice"
            assert rows[0][2] == 25
        
        # Step 3: Get the schema of the table
        schema = get_table_schema(connection_string, "test_table")
        
        # Check that the schema contains the column names
        assert "ID" in schema
        assert "Name" in schema
        assert "Age" in schema
    finally:
        # Clean up test files
        for file in [excel_file, csv_path, db_path]:
            if os.path.exists(file):
                try:
                    os.unlink(file)
                except PermissionError:
                    # On Windows, files might still be in use by another process
                    print(f"Warning: Could not delete file {file} - it may be in use by another process")