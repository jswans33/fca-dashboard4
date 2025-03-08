"""
Notebook Setup Script for NexusML

This script sets up the environment for Jupyter notebooks,
ensuring that the nexusml package can be imported correctly
and providing utility functions for working with paths.

Usage:
    # At the beginning of your notebook, add:
    %run setup_notebook.py
    
    # Then you can use the setup_notebook_environment function:
    paths = setup_notebook_environment()
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Import the init_notebook_environment function from init_notebook.py
from init_notebook import init_notebook_environment

def setup_notebook_environment() -> Dict[str, str]:
    """
    Set up the notebook environment by adding the necessary paths
    to the Python path to ensure nexusml can be imported.
    
    Returns:
        Dictionary of useful paths for notebooks
    """
    # Call the init_notebook_environment function
    paths = init_notebook_environment()
    
    # Add additional setup if needed
    print("Notebook environment setup complete.")
    
    return dict(paths)  # Convert to dict to ensure correct return type
# When run directly (via %run), set up the environment
if __name__ == "__main__":
    paths = setup_notebook_environment()
    
    # Print the paths
    print("\nAvailable paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")