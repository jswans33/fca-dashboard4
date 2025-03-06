"""
OmniClass Example Visualization Tool

This module provides a simple example of visualizing OmniClass data in a hierarchical tree structure.
It uses a hardcoded example dataset of medical equipment (dialysis products) to demonstrate the
hierarchy visualization capabilities.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add path to allow importing from nexusml package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from nexusml.utils import get_logger

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


def build_tree(data_lines):
    """
    Build a hierarchical tree from OmniClass data lines.

    Args:
        data_lines: List of strings in format "code,title,description"

    Returns:
        A nested dictionary representing the tree structure
    """
    tree = {}

    for line in data_lines:
        # Split the line into code, title, and description
        parts = line.split(",", 2)
        if len(parts) < 2:
            continue

        code = parts[0].strip()
        title = parts[1].strip()
        description = parts[2].strip() if len(parts) > 2 else ""

        # Remove quotes from description if present
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]

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


# Example data of medical equipment (dialysis products)
example_data = """
23-25 23 13-1,Peritoneal Dialysis Products,"Products used for peritoneal dialysis, a type of renal replacement therapy."
23-25 23 13-2,Continuous Ambulatory Peritoneal Dialysis Transfer Units,"Units used for continuous ambulatory peritoneal dialysis, a type of dialysis."
23-25 23 13-3,Dialysis Boxes,Boxes used in peritoneal dialysis treatments.
23-25 23 13-4,Pheresis Units,"Units used for pheresis, a type of blood filtration therapy."
23-25 23 13-5,Peritoneal Dialysis Units,"Units used for peritoneal dialysis, a type of renal replacement therapy."
23-25 23 13-6,Hollow Peritoneal Dialysis Units,"Units used for hollow fiber peritoneal dialysis, a type of dialysis."
23-25 23 13-7,Hemodialysis Products,"Products used for hemodialysis, a type of renal replacement therapy."
23-25 23 13-8,Hemodialysis Blood Oxygen Demand Units,Units that monitor the oxygen demand of blood during hemodialysis.
23-25 23 13-9,Hemodialysis Conductivity Meters,Meters used to monitor the conductivity of dialysis fluid during hemodialysis.
23-25 23 13-10,Hemodialysis Filters,Filters used in hemodialysis to remove waste and excess water from the blood.
23-25 23 13-11,Hemodialysis Level Detectors,Devices used to monitor the level of dialysis fluid during hemodialysis.
23-25 23 13-12,Hemodialysis Pressure Pumps,Pumps used to circulate blood during hemodialysis.
23-25 23 13-13,Hemodialysis Reprocessing Units,Units used to reprocess dialysis equipment for reuse.
23-25 23 13-14,Hemodialysis Tanks,Tanks used to hold dialysis fluid during hemodialysis.
"""


def main():
    try:
        # Default output directory
        output_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../nexusml/output")
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process the example data
        logger.info("Processing example OmniClass data for medical equipment")
        data_lines = [
            line.strip() for line in example_data.strip().split("\n") if line.strip()
        ]
        logger.info(f"Loaded {len(data_lines)} example data lines")

        # Build the tree
        logger.info("Building hierarchy tree from example data")
        tree = build_tree(data_lines)

        # Ask for output directory
        output_dir_input = input(f"Enter output directory (default: {output_dir}): ")
        if output_dir_input:
            output_dir = output_dir_input
            os.makedirs(output_dir, exist_ok=True)

        # Output format
        output_format = (
            input("Output format (terminal/markdown, default: terminal): ").lower()
            or "terminal"
        )

        if output_format == "markdown":
            logger.info("Generating markdown output")
            markdown_lines = print_tree_markdown(tree)
            print("\n".join(markdown_lines))

            # Option to save to file
            save_option = input("Save to file? (y/n, default: y): ").lower() or "y"
            if save_option == "y":
                filename = (
                    input("Enter output file name (default: omniclass_example.md): ")
                    or "omniclass_example.md"
                )
                output_file = os.path.join(output_dir, filename)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(markdown_lines))
                logger.info(f"Saved markdown output to {output_file}")
                print(f"Saved to {output_file}")
        else:
            logger.info("Generating terminal output")
            print("\nOmniClass Hierarchy Tree:")
            print_tree_terminal(tree)

            # Save terminal output to file as well
            save_option = input("Save to file? (y/n, default: y): ").lower() or "y"
            if save_option == "y":
                filename = (
                    input("Enter output file name (default: omniclass_example.txt): ")
                    or "omniclass_example.txt"
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
