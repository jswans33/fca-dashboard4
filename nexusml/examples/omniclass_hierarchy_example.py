"""
OmniClass Hierarchy Visualization Example

This example demonstrates how to use the OmniClass hierarchy visualization tools
to display OmniClass data in a hierarchical tree structure.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# Add path to allow importing from nexusml package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from nexusml.ingest.data import __file__ as data_file
from nexusml.ingest.generator.omniclass_hierarchy import (
    build_tree,
    print_tree_markdown,
    print_tree_terminal,
)
from nexusml.utils import clean_omniclass_csv, get_logger, read_csv_safe

# Path to the data directory
DATA_DIR = os.path.dirname(data_file)
logger = get_logger(__name__)


def main():
    """
    Main function to demonstrate OmniClass hierarchy visualization.
    """
    # Default output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the OmniClass data
    omniclass_file = os.path.join(DATA_DIR, "omniclass.csv")
    logger.info(f"Loading OmniClass data from: {omniclass_file}")
    print(f"Loading OmniClass data from: {omniclass_file}")

    try:
        # Try to read the CSV file safely
        try:
            logger.info("Attempting to read CSV file")
            df = read_csv_safe(omniclass_file)
            logger.info(f"Successfully loaded {len(df)} rows")
        except Exception as e:
            logger.warning(f"Error reading CSV file: {e}")
            logger.info("Cleaning the CSV file...")
            print("CSV file has issues. Cleaning it...")

            # Clean the CSV file
            cleaned_file = clean_omniclass_csv(omniclass_file)
            logger.info(f"Using cleaned file: {cleaned_file}")
            print(f"Using cleaned file: {cleaned_file}")

            # Read the cleaned file
            df = read_csv_safe(cleaned_file)
            logger.info(f"Successfully loaded {len(df)} rows from cleaned file")

        # Display available columns
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")

        # Set column names
        code_col = "OmniClass_Code"
        title_col = "OmniClass_Title"
        desc_col = "Description"

        # Filter data (optional)
        # For example, filter to only show Table 23 (Products) entries
        filter_value = "23-"
        logger.info(f"Filtering by: {filter_value}")
        print(f"\nFiltering by: {filter_value}")
        filtered_df = df[df[code_col].str.contains(filter_value, na=False)]
        logger.info(f"Filtered to {len(filtered_df)} rows")

        # Further filter to limit the number of entries for the example
        # For example, only show entries related to HVAC
        hvac_filter = "HVAC|mechanical|boiler|pump|chiller"
        logger.info(f"Further filtering by: {hvac_filter}")
        print(f"Further filtering by: {hvac_filter}")
        hvac_df = filtered_df[
            (filtered_df[title_col].str.contains(hvac_filter, case=False, na=False))
            | (filtered_df[desc_col].str.contains(hvac_filter, case=False, na=False))
        ]
        logger.info(f"Final dataset has {len(hvac_df)} rows")

        # Build the tree
        logger.info("Building hierarchy tree...")
        print("\nBuilding hierarchy tree...")
        tree = build_tree(hvac_df, code_col, title_col, desc_col)

        # Display the tree in terminal format
        logger.info("Generating terminal output...")
        print("\nOmniClass Hierarchy Tree (Terminal Format):")
        print_tree_terminal(tree)

        # Display the tree in markdown format
        logger.info("Generating markdown output...")
        print("\nOmniClass Hierarchy Tree (Markdown Format):")
        markdown_lines = print_tree_markdown(tree)
        print("\n".join(markdown_lines))

        # Save to file
        output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.md")
        with open(output_file, "w") as f:
            f.write("\n".join(markdown_lines))
        logger.info(f"Saved markdown output to {output_file}")
        print(f"\nSaved to {output_file}")

        # Save terminal output to file as well
        terminal_output_file = os.path.join(output_dir, "omniclass_hvac_hierarchy.txt")
        with open(terminal_output_file, "w", encoding="utf-8") as f:
            # Redirect stdout to file temporarily
            import contextlib

            with contextlib.redirect_stdout(f):
                print("OmniClass Hierarchy Tree (Terminal Format):")
                print_tree_terminal(tree)
        logger.info(f"Saved terminal output to {terminal_output_file}")
        print(f"Saved terminal output to {terminal_output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
