"""
Base module for Excel utilities.

This module provides base classes and common utilities for Excel operations.
"""

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger


class ExcelUtilError(FCADashboardError):
    """Exception raised for errors in Excel utility functions."""
    pass