"""
Reference Data Manager

This module provides a unified interface for managing reference data from multiple sources.
It's a wrapper around the more modular implementation in the reference package.
"""

# Re-export the ReferenceManager from the package
from nexusml.core.reference.manager import ReferenceManager


# For backward compatibility
def get_reference_manager(config_path=None):
    """
    Get an instance of the ReferenceManager.

    Args:
        config_path: Optional path to the configuration file

    Returns:
        ReferenceManager instance
    """
    return ReferenceManager(config_path)
