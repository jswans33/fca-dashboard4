"""
OmniClass Hierarchy Visualization Tool

This module provides functionality to visualize OmniClass data in a hierarchical tree structure.
It can parse OmniClass codes in the format xx-yy yy yy-zz and display the hierarchy in terminal
or markdown format.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add path to allow importing from nexusml package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from nexusml.ingest.data import __file__ as data_file
from nexusml.utils import clean_omniclass_csv, get_logger, read_csv_safe

# Path to the data directory
DATA_DIR = os.path.dirname(data_file)
logger = get_logger(__name__)


def parse_omniclass_code(code):
    """
    Parse an OmniClass code into its hierarchical components.
    Format: xx-yy yy yy-zz where:
    - xx: OmniClass table
    - yy yy yy: hierarchy
    - zz: detail number
    """
    # Remove any whitespace and split by hyphens
    parts = re.split(r"[-\s]+", code.strip())

    # Return the parsed components
    if len(parts) >= 4:  # Full format with detail number
        return {"table": parts[0], "hierarchy": parts[1:-1], "detail": parts[-1]}
    else:  # Partial format without detail number
        return {"table": parts[0], "hierarchy": parts[1:], "detail": None}


def build_tree(df, code_column, title_column, description_column=None):
    """
    Build a hierarchical tree from OmniClass data.

    Args:
        df: DataFrame containing OmniClass data
        code_column: Name of the column containing OmniClass codes
        title_column: Name of the column containing titles
        description_column: Optional name of the column containing descriptions

    Returns:
        A nested dictionary representing the tree structure
    """
    tree = {}

    # Sort by code to ensure parent nodes are processed before children
    df_sorted = df.sort_values(by=code_column)

    for _, row in df_sorted.iterrows():
        code = row[code_column]
        title = row[title_column]
        description = row[description_column] if description_column else ""

        # Parse the code
        parsed = parse_omniclass_code(code)

        # Navigate to the correct position in the tree
        current = tree
        path = [parsed["table"]] + parsed["hierarchy"]

        for i, part in enumerate(path):
            # Create the node if it doesn't exist
            if part not in current:
                current[part] = {"title": "", "description": "", "children": {}}

            if i == len(path) - 1:  # Last part (leaf node)
                # Update the leaf node with actual data
                current[part]["title"] = title
                current[part]["description"] = description
            else:
                # Ensure children dictionary exists for intermediate nodes
                if "children" not in current[part]:
                    current[part]["children"] = {}

                # Move to the next level
                current = current[part]["children"]

    return tree


def print_tree_terminal(tree, indent=0, prefix=""):
    """
    Print the tree structure to the terminal.
    """
    for key, node in sorted(tree.items()):
        if isinstance(node, dict) and "title" in node:
            title = node["title"]
            description = node["description"]

            # Print the current node
            if description:
                print(f"{' ' * indent}{prefix}{key}: {title} - {description}")
            else:
                print(f"{' ' * indent}{prefix}{key}: {title}")

            # Print children
            if "children" in node and node["children"]:
                print_tree_terminal(node["children"], indent + 4, "└── ")
        else:
            # Handle case where node is not properly formatted
            print(f"{' ' * indent}{prefix}{key}")
            if isinstance(node, dict) and "children" in node:
                print_tree_terminal(node["children"], indent + 4, "└── ")


def print_tree_markdown(tree, indent=0, prefix=""):
    """
    Print the tree structure in Markdown format.
    """
    markdown = []

    for key, node in sorted(tree.items()):
        if isinstance(node, dict) and "title" in node:
            title = node["title"]
            description = node["description"]

            # Create the current node line
            if description:
                line = f"{'  ' * indent}{prefix}**{key}**: {title} - *{description}*"
            else:
                line = f"{'  ' * indent}{prefix}**{key}**: {title}"

            markdown.append(line)

            # Add children
            if "children" in node and node["children"]:
                child_md = print_tree_markdown(node["children"], indent + 1, "- ")
                markdown.extend(child_md)
        else:
            # Handle case where node is not properly formatted
            line = f"{'  ' * indent}{prefix}**{key}**"
            markdown.append(line)
            if isinstance(node, dict) and "children" in node:
                child_md = print_tree_markdown(node["children"], indent + 1, "- ")
                markdown.extend(child_md)

    return markdown


def main():
    # Default values
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../nexusml/output")
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the OmniClass data
    try:
        default_file = os.path.join(DATA_DIR, "omniclass.csv")
        file_path = (
            input(
                f"Enter the path to the OmniClass CSV file (default: {default_file}): "
            )
            or default_file
        )

        # Ask for output directory
        output_dir_input = input(f"Enter output directory (default: {output_dir}): ")
        if output_dir_input:
            output_dir = output_dir_input
            os.makedirs(output_dir, exist_ok=True)

        # Check if the file needs cleaning
        try:
            logger.info(f"Attempting to read CSV file: {file_path}")
            df = read_csv_safe(file_path)
            logger.info("CSV file read successfully")
        except Exception as e:
            logger.warning(f"Error reading CSV file: {e}")
            logger.info("Attempting to clean the CSV file...")

            # Ask user if they want to clean the file
            clean_option = (
                input("CSV file has issues. Clean it? (y/n, default: y): ").lower()
                or "y"
            )
            if clean_option == "y":
                cleaned_file = clean_omniclass_csv(file_path)
                logger.info(f"Using cleaned file: {cleaned_file}")
                file_path = cleaned_file
                df = read_csv_safe(file_path)
            else:
                raise ValueError("Cannot proceed without cleaning the CSV file")

        # Determine column names
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")

        code_col = (
            input(
                "\nEnter the name of the column containing OmniClass codes (default: OmniClass_Code): "
            )
            or "OmniClass_Code"
        )
        title_col = (
            input(
                "Enter the name of the column containing titles (default: OmniClass_Title): "
            )
            or "OmniClass_Title"
        )
        desc_col = input(
            "Enter the name of the column containing descriptions (optional, press Enter to skip): "
        )

        # Optional filtering
        filter_option = (
            input("\nDo you want to filter the data? (y/n, default: n): ").lower()
            or "n"
        )
        if filter_option == "y":
            filter_column = (
                input("Enter the column to filter on (default: OmniClass_Code): ")
                or "OmniClass_Code"
            )
            filter_value = input(
                "Enter the value to filter for (e.g., '23-' for Table 23): "
            )
            df = df[df[filter_column].str.contains(filter_value, na=False)]
            logger.info(f"Filtered data to {len(df)} rows matching '{filter_value}'")

        # Build the tree
        logger.info("Building hierarchy tree...")
        tree = build_tree(df, code_col, title_col, desc_col if desc_col else None)

        # Output format
        output_format = (
            input("\nOutput format (terminal/markdown, default: terminal): ").lower()
            or "terminal"
        )

        if output_format == "markdown":
            logger.info("Generating markdown output...")
            markdown_lines = print_tree_markdown(tree)
            print("\n".join(markdown_lines))

            # Option to save to file
            save_option = input("\nSave to file? (y/n, default: y): ").lower() or "y"
            if save_option == "y":
                filename = (
                    input("Enter output file name (default: omniclass_hierarchy.md): ")
                    or "omniclass_hierarchy.md"
                )
                output_file = os.path.join(output_dir, filename)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(markdown_lines))
                logger.info(f"Saved markdown output to {output_file}")
                print(f"Saved to {output_file}")
        else:
            logger.info("Generating terminal output...")
            print("\nOmniClass Hierarchy Tree:")
            print_tree_terminal(tree)

            # Save terminal output to file as well
            save_option = input("\nSave to file? (y/n, default: y): ").lower() or "y"
            if save_option == "y":
                filename = (
                    input("Enter output file name (default: omniclass_hierarchy.txt): ")
                    or "omniclass_hierarchy.txt"
                )
                output_file = os.path.join(output_dir, filename)
                with open(output_file, "w", encoding="utf-8") as f:
                    # Redirect stdout to file temporarily
                    import contextlib

                    with contextlib.redirect_stdout(f):
                        print("OmniClass Hierarchy Tree:")
                        print_tree_terminal(tree)
                logger.info(f"Saved terminal output to {output_file}")
                print(f"Saved to {output_file}")

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
