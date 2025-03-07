"""
Data Selection Utility for NexusML

This module provides utilities for finding and loading data files from different locations.
It can be imported and used directly in notebooks or scripts.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    # Assuming this module is in nexusml/utils
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the project root
    return os.path.dirname(os.path.dirname(module_dir))


def find_data_files(
    locations: Optional[List[str]] = None, extensions: List[str] = [".xlsx", ".csv"]
) -> Dict[str, str]:
    """
    Find all data files with specified extensions in the given locations.

    Args:
        locations: List of directory paths to search. If None, uses default locations.
        extensions: List of file extensions to include

    Returns:
        Dictionary mapping file names to their full paths
    """
    if locations is None:
        # Default locations to search
        project_root = get_project_root()
        project_root_parent = os.path.dirname(project_root)
        locations = [
            os.path.join(project_root_parent, "examples"),
            os.path.join(project_root_parent, "uploads"),
            os.path.join(project_root, "data"),
        ]

    data_files = {}
    for location in locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(location, file)
                    data_files[file] = file_path
    return data_files


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file based on its extension.

    Args:
        file_path: Path to the data file

    Returns:
        Pandas DataFrame containing the loaded data
    """
    if file_path.endswith(".xlsx"):
        return pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def list_available_data() -> Dict[str, str]:
    """
    List all available data files in the default locations.

    Returns:
        Dictionary mapping file names to their full paths
    """
    data_files = find_data_files()

    print("Available data files:")
    for i, (file_name, file_path) in enumerate(data_files.items()):
        print(f"{i+1}. {file_name} ({file_path})")

    return data_files


def select_and_load_data(file_name: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """
    Select and load a data file.

    Args:
        file_name: Name of the file to load. If None, uses the first available file.

    Returns:
        Tuple of (loaded DataFrame, file path)
    """
    data_files = find_data_files()

    if not data_files:
        raise FileNotFoundError("No data files found in the default locations.")

    if file_name is None:
        # Use the first file
        file_name = list(data_files.keys())[0]
    elif file_name not in data_files:
        raise FileNotFoundError(f"File not found: {file_name}")

    data_path = data_files[file_name]
    print(f"Selected file: {file_name}")
    print(f"Full path: {data_path}")

    # Load the data
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")

    return data, data_path


# Example usage in a notebook:
"""
from nexusml.utils.data_selection import list_available_data, select_and_load_data

# List all available data files
list_available_data()

# Load a specific file
data, data_path = select_and_load_data("sample_data.xlsx")

# Or let it choose the first available file
data, data_path = select_and_load_data()
"""
