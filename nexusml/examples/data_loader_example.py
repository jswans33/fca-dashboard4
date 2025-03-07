"""
NexusML Data Loader Example

This script demonstrates how to load data from different locations in the project.
It provides a flexible way to select which data file to load.
"""

import os
import sys
from typing import Dict, List, Optional

import pandas as pd


def get_project_root() -> str:
    """Get the absolute path to the project root directory."""
    # Assuming this script is in nexusml/examples
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root
    return os.path.dirname(script_dir)


def find_data_files(
    locations: List[str], extensions: List[str] = [".xlsx", ".csv"]
) -> Dict[str, str]:
    """
    Find all data files with specified extensions in the given locations.

    Args:
        locations: List of directory paths to search
        extensions: List of file extensions to include

    Returns:
        Dictionary mapping file names to their full paths
    """
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


def main():
    # Get project root
    project_root = get_project_root()
    project_root_parent = os.path.dirname(project_root)

    # Define locations to search for data files
    data_locations = [
        os.path.join(project_root_parent, "examples"),
        os.path.join(project_root_parent, "uploads"),
    ]

    # Find all data files
    data_files = find_data_files(data_locations)

    # Print available files
    print("Available data files:")
    for i, (file_name, file_path) in enumerate(data_files.items()):
        print(f"{i+1}. {file_name} ({file_path})")

    # Let user select a file
    if data_files:
        # In a script, you could prompt the user
        # selection = input("\nEnter the number of the file to load: ")
        # selected_index = int(selection) - 1

        # For demonstration, just use the first file
        selected_index = 0
        selected_file = list(data_files.keys())[selected_index]
        data_path = data_files[selected_file]

        print(f"\nSelected file: {selected_file}")
        print(f"Full path: {data_path}")

        # Load the data
        data = load_data(data_path)

        # Display information about the data
        print(f"\nData shape: {data.shape}")
        print("\nFirst 5 rows:")
        print(data.head())
    else:
        print("No data files found in the specified locations.")


if __name__ == "__main__":
    main()
