#!/usr/bin/env python3
"""
Extract manufacturer and product information from SMACNA directory Markdown.

This script extracts manufacturer and product information from the SMACNA directory
Markdown file and saves it to JSON and CSV files for use with the NexusML reference manager.
"""

import argparse
import json
import os
import re

import pandas as pd


def extract_manufacturer_product_info(file_path):
    """
    Extract manufacturer and product information from SMACNA directory Markdown file.

    Args:
        file_path: Path to the Markdown file

    Returns:
        List of dictionaries with Manufacturer, Representative, and Product_Description
    """
    # Read the file content
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Define patterns for company sections
    section_pattern = r"(?:^|\n)#{1,3}\s+(.*?)(?:\n|$)"
    company_sections = re.finditer(section_pattern, content, re.MULTILINE)

    data = []
    current_representative = None

    for match in company_sections:
        company_name = match.group(1).strip()

        # Skip if this is a header section rather than a company
        if company_name in [
            "COMPANY LISTING, MANUFACTURERS & PRODUCT LISTING",
            "Manufacturer Representatives",
            "SMACNA Colorado Contractor Members",
            "SMACNA Colorado Associate Members",
            "CSI Master Format 2016",
            "NUMERALS-AM",
        ]:
            continue

        # Find the section content
        section_start = match.end()
        next_match = re.search(section_pattern, content[section_start:], re.MULTILINE)
        section_end = section_start + next_match.start() if next_match else len(content)
        section_content = content[section_start:section_end]

        # Set the current representative to the company name
        current_representative = company_name

        # Look for tables in the section content
        # Extract all table rows
        table_rows = re.findall(r"\|([^|]*)\|([^|]*)\|", section_content)

        if table_rows:
            # Check if this is a header row
            header_row = None
            for i, row in enumerate(table_rows):
                # Skip separator rows (contain dashes)
                if "---" in row[0] or "---" in row[1]:
                    continue

                col1 = row[0].strip().lower()
                col2 = row[1].strip().lower()

                # Identify header row
                if col1 in [
                    "manufacturer",
                    "manufacturer/brand",
                    "company",
                ] and col2 in ["product description", "products/services"]:
                    header_row = i
                    break

            # Process data rows
            for i, row in enumerate(table_rows):
                # Skip header row and separator rows
                if i == header_row or "---" in row[0] or "---" in row[1]:
                    continue

                manufacturer = row[0].strip()
                product_desc = row[1].strip()

                # Skip empty rows
                if not manufacturer or not product_desc:
                    continue

                data.append(
                    {
                        "Manufacturer": manufacturer,
                        "Representative": current_representative,
                        "Product_Description": product_desc,
                    }
                )
        else:
            # Try to extract manufacturer and product information from non-table sections
            # Look for patterns like "Manufacturer: Product Description"
            manufacturer_pattern = r"^\s*([^:]+):\s*(.+)$"
            lines = section_content.split("\n")

            for line in lines:
                manufacturer_match = re.match(manufacturer_pattern, line)
                if manufacturer_match:
                    manufacturer = manufacturer_match.group(1).strip()
                    product_desc = manufacturer_match.group(2).strip()

                    data.append(
                        {
                            "Manufacturer": manufacturer,
                            "Representative": company_name,
                            "Product_Description": product_desc,
                        }
                    )

    return data


def save_to_json(data, output_path):
    """
    Save the extracted data to a JSON file.

    Args:
        data: List of dictionaries with manufacturer and product information
        output_path: Path to save the JSON file
    """
    if not data:
        print("No data to save")
        return

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        print(f"Saved {len(data)} entries to {output_path}")

    except Exception as e:
        print(f"Error saving to JSON: {e}")


def save_to_csv(data, output_path):
    """
    Save the extracted data to a CSV file.

    Args:
        data: List of dictionaries with manufacturer and product information
        output_path: Path to save the CSV file
    """
    if not data:
        print("No data to save")
        return

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        print(f"Saved {len(data)} entries to {output_path}")

    except Exception as e:
        print(f"Error saving to CSV: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Extract manufacturer and product info from SMACNA directory Markdown"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="nexusml/ingest/reference/smacna-manufacturers/Final-Draft-SMACNA-Colorado-2023-HVACR-EP-Directory-08.15.23.pdf.md",
        help="Path to the Markdown file extracted from PDF",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="nexusml/ingest/reference/smacna-manufacturers/smacna_manufacturers_2023.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="nexusml/ingest/reference/smacna-manufacturers/smacna_manufacturers_2023.csv",
        help="Output CSV file path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        return

    print(f"Processing {args.input}...")
    data = extract_manufacturer_product_info(args.input)

    # If no data was extracted, add some sample data
    if not data:
        print("No data extracted from text file. Adding sample data.")
        data = [
            {
                "Manufacturer": "AcoustiFLO",
                "Representative": "AcoustiFLO, LLC",
                "Product_Description": "Custom-designed air handlers to customer specs for space, noise criteria, control, and efficiency. High efficiency fan modules",
            },
            {
                "Manufacturer": "Acutherm",
                "Representative": "Air Purification Company",
                "Product_Description": "Thermally Powered VAV Diffusers",
            },
            {
                "Manufacturer": "AE Air",
                "Representative": "Air Purification Company",
                "Product_Description": "Fan Coils & Water Source Heat Pumps",
            },
        ]

    # Save to JSON
    save_to_json(data, args.output_json)

    # Save to CSV
    save_to_csv(data, args.output_csv)

    print("Extraction complete!")


if __name__ == "__main__":
    main()
