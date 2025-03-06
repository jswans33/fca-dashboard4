"""
OmniClass Tree Visualization Tool

This module provides a simplified command-line tool to visualize OmniClass data in a hierarchical tree structure.
It can parse OmniClass codes in the format xx-yy yy yy-zz and display the hierarchy in terminal or markdown format.
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
    # Remove any whitespace and split by hyphens and spaces
    parts = re.split(r"[-\s]+", code.strip())

    # Return the parsed components
    if len(parts) >= 4:  # Full format with detail number
        return {"table": parts[0], "hierarchy": parts[1:-1], "detail": parts[-1]}
    else:  # Partial format without detail number
        return {"table": parts[0], "hierarchy": parts[1:], "detail": None}


def build_tree(df, code_column, title_column, description_column=None):
    """
    Build a hierarchical tree from OmniClass data.
    """
    tree = {}

    # Sort by code to ensure parent nodes are processed before children
    df_sorted = df.sort_values(by=code_column)

    for _, row in df_sorted.iterrows():
        code = row[code_column]
        title = row[title_column]
        description = (
            row[description_column]
            if description_column and description_column in row
            else ""
        )

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
    default_file = os.path.join(DATA_DIR, "omniclass.csv")
    file_path = default_file
    code_column = "OmniClass_Code"
    title_column = "OmniClass_Title"
    description_column = "Description"
    output_format = "terminal"
    filter_value = None
    clean_csv = False
    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../nexusml/output")
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check for command line arguments
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    # Additional command line arguments
    if len(sys.argv) > 2:
        filter_value = sys.argv[2]

    if len(sys.argv) > 3:
        output_format = sys.argv[3]

    if len(sys.argv) > 4:
        if sys.argv[4].lower() == "clean":
            clean_csv = True
        else:
            output_dir = sys.argv[4]

    if len(sys.argv) > 5 and sys.argv[5].lower() == "clean":
        clean_csv = True

    # Load the OmniClass data
    try:
        logger.info(f"Loading data from {file_path}...")

        if clean_csv:
            logger.info("Cleaning CSV file before loading...")
            file_path = clean_omniclass_csv(file_path)
            logger.info(f"Using cleaned file: {file_path}")

        try:
            df = read_csv_safe(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from {file_path}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            logger.info("Attempting to clean the CSV file...")
            file_path = clean_omniclass_csv(file_path)
            logger.info(f"Using cleaned file: {file_path}")
            df = read_csv_safe(file_path)
            logger.info(f"Successfully loaded {len(df)} rows from cleaned file")

        # Apply filter if specified
        if filter_value:
            logger.info(f"Filtering by: {filter_value}")
            df = df[df[code_column].str.contains(filter_value, na=False)]
            logger.info(f"Filtered to {len(df)} rows")

        # Build the tree
        logger.info("Building hierarchy tree...")
        tree = build_tree(df, code_column, title_column, description_column)

        # Output the tree
        if output_format.lower() == "markdown":
            logger.info("Generating markdown output...")
            markdown_lines = print_tree_markdown(tree)
            print("\n".join(markdown_lines))

            # Save to file
            output_file = os.path.join(
                output_dir,
                f"omniclass_{filter_value if filter_value else 'all'}_hierarchy.md",
            )
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("\n".join(markdown_lines))
            logger.info(f"Saved markdown output to {output_file}")
            print(f"Saved to {output_file}")
        else:
            logger.info("Generating terminal output...")
            print("\nOmniClass Hierarchy Tree:")
            print_tree_terminal(tree)

            # Save terminal output to file as well
            output_file = os.path.join(
                output_dir,
                f"omniclass_{filter_value if filter_value else 'all'}_hierarchy.txt",
            )
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
