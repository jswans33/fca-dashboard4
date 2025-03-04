"""
Pytest configuration file.

This file contains shared fixtures and configuration for pytest.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
# This ensures that the tests can import modules from the project
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
