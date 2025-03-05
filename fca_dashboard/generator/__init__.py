"""
Generator package for the FCA Dashboard application.

This package provides utilities for generating training data for the classifier.
"""

# Import submodules
from fca_dashboard.generator.omniclass import extract_omniclass_data

# For backward compatibility
__all__ = [
    'extract_omniclass_data',
]