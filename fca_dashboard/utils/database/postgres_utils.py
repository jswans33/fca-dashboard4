"""
PostgreSQL utilities for the FCA Dashboard application.

This module provides utilities for working with PostgreSQL databases.
"""

import pandas as pd
from sqlalchemy import create_engine, text

from fca_dashboard.utils.logging_config import get_logger


def save_dataframe_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    connection_string: str,
    schema: str = None,
    if_exists: str = "replace",
    index: bool = False,
    **kwargs
) -> None:
    """
    Save a DataFrame to a PostgreSQL table.
    
    Args:
        df: The DataFrame to save.
        table_name: The name of the table to save to.
        connection_string: The PostgreSQL connection string.
        schema: The database schema.
        if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        index: Whether to include the index in the table.
        **kwargs: Additional arguments to pass to pandas.DataFrame.to_sql().
    """
    logger = get_logger("postgres_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Save the DataFrame to the database
    df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=index,
        **kwargs
    )
    
    logger.info(f"Successfully saved {len(df)} rows to PostgreSQL table {table_name}")


def get_postgres_table_schema(
    connection_string: str,
    table_name: str,
    schema: str = "public"
) -> str:
    """
    Get the schema of a PostgreSQL table.
    
    Args:
        connection_string: The PostgreSQL connection string.
        table_name: The name of the table to get the schema for.
        schema: The database schema (default: 'public').
        
    Returns:
        A string containing the schema of the table.
    """
    logger = get_logger("postgres_utils")
    
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)
    
    # Get the schema of the table
    with engine.connect() as conn:
        query = f"""
            SELECT column_name, data_type, character_maximum_length
            FROM information_schema.columns
            WHERE table_name = '{table_name}'
            AND table_schema = '{schema}'
            ORDER BY ordinal_position
        """
        result = conn.execute(text(query))
        columns = []
        for row in result:
            column_type = row[1]
            if row[2] is not None:
                column_type = f"{column_type}({row[2]})"
            columns.append(f"{row[0]} {column_type}")
        
        schema_str = f"CREATE TABLE {schema}.{table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    
    logger.info(f"Successfully retrieved schema for PostgreSQL table {table_name}")
    return schema_str