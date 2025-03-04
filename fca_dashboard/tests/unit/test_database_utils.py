"""
Unit tests for the database utility module.

This module contains tests for the database utility functions in the
fca_dashboard.utils.database package.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fca_dashboard.utils.database import (
    DatabaseError,
    get_table_schema,
    save_dataframe_to_database,
)
from fca_dashboard.utils.database.sqlite_utils import (
    get_sqlite_table_schema,
    save_dataframe_to_sqlite,
)


def test_save_dataframe_to_database_sqlite():
    """Test saving a DataFrame to a SQLite database."""
    # Create a test DataFrame
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    })
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{tmp_path}"
        
        # Save the DataFrame to the database
        save_dataframe_to_database(
            df=df,
            table_name="test_table",
            connection_string=connection_string,
            if_exists="replace"
        )
        
        # Verify the data was saved correctly
        engine = create_engine(connection_string)
        rows = []
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test_table"))
            rows = result.fetchall()
        
        # Dispose of the engine to close all connections
        engine.dispose()
            
        # Check the number of rows
        assert len(rows) == 3
            
        # Check the values in the first row
        assert rows[0][0] == 1
        assert rows[0][1] == "Alice"
        assert rows[0][2] == 25
    finally:
        # Clean up the temporary database file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


def test_get_table_schema_sqlite():
    """Test getting the schema of a SQLite table."""
    # Create a test DataFrame
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    })
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{tmp_path}"
        
        # Save the DataFrame to the database
        engine = create_engine(connection_string)
        df.to_sql("test_table", engine, index=False)
        
        # Get the schema of the table
        schema = get_table_schema(connection_string, "test_table")
        
        # Dispose of the engine to close all connections
        engine.dispose()
        
        # Check that the schema contains the column names
        assert "ID" in schema
        assert "Name" in schema
        assert "Age" in schema
        
        # Check that the schema is a CREATE TABLE statement
        assert schema.startswith("CREATE TABLE test_table")
    finally:
        # Clean up the temporary database file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


def test_save_dataframe_to_sqlite():
    """Test saving a DataFrame to a SQLite database using the SQLite-specific function."""
    # Create a test DataFrame
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    })
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{tmp_path}"
        
        # Save the DataFrame to the database
        save_dataframe_to_sqlite(
            df=df,
            table_name="test_table",
            connection_string=connection_string,
            if_exists="replace"
        )
        
        # Verify the data was saved correctly
        engine = create_engine(connection_string)
        rows = []
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM test_table"))
            rows = result.fetchall()
        
        # Dispose of the engine to close all connections
        engine.dispose()
            
        # Check the number of rows
        assert len(rows) == 3
            
        # Check the values in the first row
        assert rows[0][0] == 1
        assert rows[0][1] == "Alice"
        assert rows[0][2] == 25
    finally:
        # Clean up the temporary database file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


def test_get_sqlite_table_schema():
    """Test getting the schema of a SQLite table using the SQLite-specific function."""
    # Create a test DataFrame
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    })
    
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{tmp_path}"
        
        # Save the DataFrame to the database
        engine = create_engine(connection_string)
        df.to_sql("test_table", engine, index=False)
        
        # Get the schema of the table
        schema = get_sqlite_table_schema(connection_string, "test_table")
        
        # Dispose of the engine to close all connections
        engine.dispose()
        
        # Check that the schema contains the column names
        assert "ID" in schema
        assert "Name" in schema
        assert "Age" in schema
        
        # Check that the schema is a CREATE TABLE statement
        assert schema.startswith("CREATE TABLE test_table")
    finally:
        # Clean up the temporary database file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")


def test_database_error():
    """Test that DatabaseError is raised for invalid operations."""
    # Create a test DataFrame
    df = pd.DataFrame({
        "ID": [1, 2, 3],
        "Name": ["Alice", "Bob", "Charlie"],
        "Age": [25, 30, 35]
    })
    
    # Test with an invalid connection string
    with pytest.raises(DatabaseError):
        save_dataframe_to_database(
            df=df,
            table_name="test_table",
            connection_string="invalid://connection/string",
            if_exists="replace"
        )
    
    # Test with an invalid table name
    # Create a temporary SQLite database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create a SQLite connection string
        connection_string = f"sqlite:///{tmp_path}"
        
        # Create an empty database
        engine = create_engine(connection_string)
        engine.dispose()
        
        # Try to get the schema of a non-existent table
        with pytest.raises(Exception):
            get_table_schema(connection_string, "non_existent_table")
    finally:
        # Clean up the temporary database file
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except PermissionError:
                # On Windows, files might still be in use by another process
                print(f"Warning: Could not delete file {tmp_path} - it may be in use by another process")