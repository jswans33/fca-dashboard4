"""
Enhanced Data Loader Example

This script demonstrates how to use the enhanced StandardDataLoader
to discover and load data files from various locations.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the enhanced data loader
from nexusml.src.pipeline.components.data_loader import StandardDataLoader


def main():
    """Demonstrate the enhanced data loader capabilities."""
    print("Enhanced Data Loader Example")
    print("============================\n")

    # Create a data loader instance
    data_loader = StandardDataLoader()
    print(f"Created {data_loader.get_name()}: {data_loader.get_description()}\n")

    # Discover available data files
    print("Discovering available data files...")
    available_files = data_loader.discover_data_files()

    if not available_files:
        print("No data files found!")
        return

    print(f"Found {len(available_files)} data files:")
    for i, (file_name, file_path) in enumerate(available_files.items(), 1):
        print(f"  {i}. {file_name}: {file_path}")

    print("\nLoading the first available file...")
    first_file = list(available_files.keys())[0]
    first_file_path = available_files[first_file]

    # Load the data using the data loader
    # This will automatically handle different file formats (CSV, Excel, etc.)
    data = data_loader.load_data(first_file_path)

    # Display information about the loaded data
    print(f"\nLoaded data from: {first_file}")
    print(f"Data shape: {data.shape}")
    print(f"Columns: {', '.join(data.columns)}")
    print(f"First few rows:")
    print(data.head())

    # Demonstrate automatic discovery and loading
    print("\nDemonstrating automatic discovery and loading...")
    auto_data = data_loader.load_data(discover_files=True)
    print(f"Automatically loaded data with shape: {auto_data.shape}")


if __name__ == "__main__":
    main()
