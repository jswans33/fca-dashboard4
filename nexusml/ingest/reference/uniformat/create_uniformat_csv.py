#!/usr/bin/env python
"""
Create Uniformat CSV File

This script creates a Uniformat CSV file in the format expected by the
UniformatDataSource class, using the data from facility_services_hierarchical.csv.
It includes all rows with proper numeric Uniformat codes (e.g., D3020.10) and
also includes the MasterFormat number.
"""

import csv
import os
import re


def clean_text(text):
    """Clean text by removing references, pipe characters, and other unwanted text."""
    if not text:
        return ""

    # Remove pipe characters
    text = text.replace("|", "")

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove quotes if they wrap the entire text
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]

    # Remove references to other sections (e.g., "Swimming Pool Plumbing Systems: F1050.10.")
    if ":" in text:
        # Check if the text after the colon looks like a reference (e.g., "F1050.10.")
        parts = text.split(":", 1)
        if len(parts) == 2 and re.search(r"[A-Z][0-9]", parts[1]):
            text = parts[0].strip()

    # Remove "See Also:" prefix
    if text.startswith("See Also:"):
        text = ""

    # Remove asterisks
    text = text.replace("*", "")

    return text


def is_valid_uniformat_code(code):
    """Check if a code is a valid Uniformat code (e.g., D3020.10)."""
    # Valid codes should start with a letter followed by numbers and possibly dots
    # But should not contain any text after the numbers
    return bool(re.match(r"^[A-Z][0-9]+(\.[0-9]+)*$", code))


def create_uniformat_csv():
    """
    Create a Uniformat CSV file in the format expected by the UniformatDataSource class.
    """
    print("Creating Uniformat CSV file...")

    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(
        script_dir, "services", "facility_services_hierarchical.csv"
    )
    output_csv = os.path.join(script_dir, "uniformat_classifications.csv")

    # Read the input CSV file
    rows = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Process rows to create uniformat data
    uniformat_data = []
    for row in rows:
        # Only include rows with a valid Uniformat code
        if row["Number"] and is_valid_uniformat_code(row["Number"]):
            # Clean the title and description
            title = clean_text(row["Title"])
            description = clean_text(row["Explanation"])

            # Skip rows with empty titles
            if not title:
                continue

            # Get the MasterFormat number
            mf_number = row["MF_Number"].strip() if row["MF_Number"] else ""

            # Ensure description is never null
            if not description:
                description = ""

            uniformat_data.append(
                {
                    "UniFormat Code": row["Number"],
                    "UniFormat Title": title,
                    "MasterFormat Number": mf_number,
                    "Description": description,
                }
            )

    # Write the output CSV file
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "UniFormat Code",
            "UniFormat Title",
            "MasterFormat Number",
            "Description",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(uniformat_data)

    print(
        f"Created Uniformat CSV file with {len(uniformat_data)} entries: {output_csv}"
    )


if __name__ == "__main__":
    create_uniformat_csv()
    print("Done.")
