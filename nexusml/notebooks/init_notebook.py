"""
Notebook Initialization Script for NexusML

This script initializes the environment for Jupyter notebooks,
ensuring that the nexusml package can be imported correctly.

Usage:
    # At the beginning of your notebook, add:
    %run init_notebook.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional


def init_notebook_environment() -> Dict[str, str]:
    """
    Initialize the notebook environment by adding the necessary paths
    to the Python path to ensure nexusml can be imported.
    
    Returns:
        Dictionary of useful paths for notebooks
    """
    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    
    # Go up to the project root (parent of nexusml)
    project_root = script_dir.parent.parent
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")
    
    # Add nexusml parent directory to Python path if not already there
    nexusml_parent = project_root
    if str(nexusml_parent) not in sys.path:
        sys.path.insert(0, str(nexusml_parent))
        print(f"Added {nexusml_parent} to Python path")
    
    # Print confirmation
    print(f"Notebook environment initialized. You can now import nexusml.")
    print(f"Project root: {project_root}")
    
    # Create and return useful paths
    paths: Dict[str, str] = {
        "project_root": str(project_root),
        "nexusml_root": str(script_dir.parent),
        "notebooks_dir": str(script_dir),
        "data_dir": str(project_root / "data"),
        "examples_dir": str(project_root / "examples"),
        "outputs_dir": str(project_root / "outputs"),
    }
    return paths


# When run directly (via %run), initialize the environment
paths = init_notebook_environment()

# Try to import nexusml to verify it works
try:
    import nexusml
    print(f"Successfully imported nexusml from {nexusml.__file__}")
except ImportError as e:
    print(f"Warning: Failed to import nexusml: {e}")
    print("You may need to install the nexusml package or check your Python path.")