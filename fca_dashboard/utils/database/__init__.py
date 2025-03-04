"""
Database utilities for the FCA Dashboard application.

This module provides utilities for working with databases, including
connection management, schema operations, and data import/export.
"""

from fca_dashboard.utils.database.base import (
    DatabaseError,
    get_table_schema,
    save_dataframe_to_database,
)

__all__ = [
    "DatabaseError",
    "save_dataframe_to_database",
    "get_table_schema",
]