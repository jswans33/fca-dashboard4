#!/usr/bin/env python3
"""
Convert MCAA Glossary and Abbreviations from PDF.md table format to the recommended format.

This script reads the Glossary.pdf.md and Abbreviations.pdf.md files, extracts the terms
and definitions from the table format, and converts them to the recommended format with
bold terms followed by definitions.
"""

import argparse
import os
import re
from pathlib import Path


def extract_terms_from_table(content):
    """
    Extract terms and definitions from a markdown table.

    Args:
        content (str): The content of the markdown file

    Returns:
        list: A list of (term, definition) tuples
    """
    # Split the content into lines
    lines = content.strip().split("\n")

    # Find table rows (lines that contain '|')
    terms_and_definitions = []
    for line in lines:
        if "|" in line:
            # Extract the term and definition from the table row
            # Format: | | Term | Definition |
            parts = line.split("|")
            if len(parts) >= 3:  # At least 3 parts: '', term, definition, ''
                term = parts[1].strip()
                definition = parts[2].strip()
                if (
                    term
                    and definition
                    and term != "Terms"
                    and definition != "Definitions"
                    and term != "Abbreviations"
                    and definition != "Definition"
                    and not term.startswith("---")  # Skip table header separator rows
                    and not definition.startswith(
                        "---"
                    )  # Skip table header separator rows
                ):
                    terms_and_definitions.append((term, definition))

    # Debug output
    print(f"Found {len(terms_and_definitions)} terms and definitions")
    if terms_and_definitions:
        print(f"First few terms: {terms_and_definitions[:3]}")

    return terms_and_definitions


def convert_to_csv_format(terms_and_definitions):
    """
    Convert terms and definitions to CSV format.

    Args:
        terms_and_definitions (list): A list of (term, definition) tuples

    Returns:
        str: The content in CSV format
    """
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Write header
    writer.writerow(["Term", "Definition"])

    # Write data
    for term, definition in terms_and_definitions:
        writer.writerow([term, definition])

    return output.getvalue()


def process_file(input_path, output_path):
    """
    Process a file and convert it to the recommended format.

    Args:
        input_path (str): The path to the input file
        output_path (str): The path to the output file
    """
    print(f"Processing {input_path}...")

    # Read the input file
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract terms and definitions
    terms_and_definitions = extract_terms_from_table(content)

    # Convert to CSV format
    output_content = convert_to_csv_format(terms_and_definitions)

    # Write the output file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    print(f"Converted {len(terms_and_definitions)} terms and saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert MCAA Glossary and Abbreviations from PDF.md table format to the recommended format."
    )
    parser.add_argument(
        "--glossary",
        type=str,
        default="files/mcaa-glossary/Glossary.pdf.md",
        help="Path to the glossary PDF.md file",
    )
    parser.add_argument(
        "--abbreviations",
        type=str,
        default="files/mcaa-glossary/Abbreviations.pdf.md",
        help="Path to the abbreviations PDF.md file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="nexusml/ingest/reference/mcaa-glossary",
        help="Directory to save the converted files",
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process the glossary file
    glossary_output_path = os.path.join(args.output_dir, "Glossary.csv")
    process_file(args.glossary, glossary_output_path)

    # Process the abbreviations file
    abbreviations_output_path = os.path.join(args.output_dir, "Abbreviations.csv")
    process_file(args.abbreviations, abbreviations_output_path)

    print("Conversion complete!")


if __name__ == "__main__":
    main()
