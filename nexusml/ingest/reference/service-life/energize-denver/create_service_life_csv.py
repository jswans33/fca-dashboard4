#!/usr/bin/env python3
"""
Create service life CSV file from provided data.

This script creates a CSV file with service life data for equipment
for use with the NexusML reference manager.
"""

import csv
import os
from pathlib import Path

# Service life data provided by the user
data = [
    # Air Conditioners
    {
        "Equipment_Category": "Air Conditioners",
        "Equipment_Item": "Window unit",
        "Service_Life_Years": 9,
    },
    {
        "Equipment_Category": "Air Conditioners",
        "Equipment_Item": "Residential single or Split package",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Air Conditioners",
        "Equipment_Item": "Commercial through the wall",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Air Conditioners",
        "Equipment_Item": "Water-cooled package",
        "Service_Life_Years": 16,
    },
    # Heat Pumps
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Residential or Commercial air-to-air",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Commercial water-to-air",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Close-coupled, end-suction",
        "Service_Life_Years": 17,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Frame-mounted, end-suction",
        "Service_Life_Years": 21,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Split-case, multistage pump",
        "Service_Life_Years": 21,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Split-case, single stage",
        "Service_Life_Years": 32,
    },
    {
        "Equipment_Category": "Heat Pumps",
        "Equipment_Item": "Vertical in-line",
        "Service_Life_Years": 21,
    },
    # Rooftop Air Conditioners
    {
        "Equipment_Category": "Rooftop Air Conditioners",
        "Equipment_Item": "Single-zone",
        "Service_Life_Years": 12,
    },
    {
        "Equipment_Category": "Rooftop Air Conditioners",
        "Equipment_Item": "Multi-zone",
        "Service_Life_Years": 14,
    },
    # Boilers, Hot Water
    {
        "Equipment_Category": "Boilers, Hot Water",
        "Equipment_Item": "Gas fired",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Boilers, Hot Water",
        "Equipment_Item": "Oil fired",
        "Service_Life_Years": 18,
    },
    {
        "Equipment_Category": "Boilers, Hot Water",
        "Equipment_Item": "Electric",
        "Service_Life_Years": 25,
    },
    # Other Heating Equipment
    {
        "Equipment_Category": "Heating Equipment",
        "Equipment_Item": "Burners",
        "Service_Life_Years": 18,
    },
    {
        "Equipment_Category": "Heating Equipment",
        "Equipment_Item": "Furnaces gas or oil fired",
        "Service_Life_Years": 17,
    },
    # Unit Heaters
    {
        "Equipment_Category": "Unit Heaters",
        "Equipment_Item": "Gas or electric",
        "Service_Life_Years": 13,
    },
    {
        "Equipment_Category": "Unit Heaters",
        "Equipment_Item": "Hot water or steam",
        "Service_Life_Years": 20,
    },
    # Radiant Heaters
    {
        "Equipment_Category": "Radiant Heaters",
        "Equipment_Item": "Electric",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Radiant Heaters",
        "Equipment_Item": "Hot water or steam",
        "Service_Life_Years": 25,
    },
    # Air Terminals
    {
        "Equipment_Category": "Air Terminals",
        "Equipment_Item": "Diffusers, Grilles, and Registers",
        "Service_Life_Years": 25,
    },
    {
        "Equipment_Category": "Air Terminals",
        "Equipment_Item": "Induction and fan coil units",
        "Service_Life_Years": 24,
    },
    {
        "Equipment_Category": "Air Terminals",
        "Equipment_Item": "VAV and double-duct boxes",
        "Service_Life_Years": 18,
    },
    # Air System Components
    {
        "Equipment_Category": "Air System Components",
        "Equipment_Item": "Air Washers",
        "Service_Life_Years": 15,
    },
    {
        "Equipment_Category": "Air System Components",
        "Equipment_Item": "Ductwork",
        "Service_Life_Years": 30,
    },
    {
        "Equipment_Category": "Air System Components",
        "Equipment_Item": "Dampers",
        "Service_Life_Years": 18,
    },
    # Packaged DX
    {
        "Equipment_Category": "Packaged DX",
        "Equipment_Item": "Air-cooled",
        "Service_Life_Years": 13,
    },
    {
        "Equipment_Category": "Packaged DX",
        "Equipment_Item": "Rooftop",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Packaged DX",
        "Equipment_Item": "Water-cooled",
        "Service_Life_Years": 15,
    },
    # Packaged Terminal
    {
        "Equipment_Category": "Packaged Terminal",
        "Equipment_Item": "Air conditioner or heat pump",
        "Service_Life_Years": 24,
    },
    # Fans
    {
        "Equipment_Category": "Fans",
        "Equipment_Item": "Centrifugal",
        "Service_Life_Years": 23,
    },
    {"Equipment_Category": "Fans", "Equipment_Item": "Axial", "Service_Life_Years": 18},
    {
        "Equipment_Category": "Fans",
        "Equipment_Item": "Propeller",
        "Service_Life_Years": 14,
    },
    {
        "Equipment_Category": "Fans",
        "Equipment_Item": "Ventilating roof-mounted",
        "Service_Life_Years": 18,
    },
    # Coils
    {
        "Equipment_Category": "Coils",
        "Equipment_Item": "DX, water, or steam",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Coils",
        "Equipment_Item": "Electric",
        "Service_Life_Years": 15,
    },
    # Heat Transfer Equipment
    {
        "Equipment_Category": "Heat Transfer Equipment",
        "Equipment_Item": "Heat Exchangers",
        "Service_Life_Years": 22,
    },
    {
        "Equipment_Category": "Heat Transfer Equipment",
        "Equipment_Item": "Reciprocating Compressors",
        "Service_Life_Years": 18,
    },
    # Packaged Chillers
    {
        "Equipment_Category": "Packaged Chillers",
        "Equipment_Item": "Reciprocating or centrifugal",
        "Service_Life_Years": 18,
    },
    {
        "Equipment_Category": "Packaged Chillers",
        "Equipment_Item": "Absorption",
        "Service_Life_Years": 25,
    },
    # Cooling Towers
    {
        "Equipment_Category": "Cooling Towers",
        "Equipment_Item": "Galvanized metal",
        "Service_Life_Years": 17,
    },
    {
        "Equipment_Category": "Cooling Towers",
        "Equipment_Item": "Wood",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Cooling Towers",
        "Equipment_Item": "Ceramic",
        "Service_Life_Years": 27,
    },
    # Condensers
    {
        "Equipment_Category": "Condensers",
        "Equipment_Item": "Air-cooled Condensers",
        "Service_Life_Years": 15,
    },
    {
        "Equipment_Category": "Condensers",
        "Equipment_Item": "Evaporative Condensers",
        "Service_Life_Years": 18,
    },
    # Insulation
    {
        "Equipment_Category": "Insulation",
        "Equipment_Item": "Molded",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Insulation",
        "Equipment_Item": "Blanket",
        "Service_Life_Years": 24,
    },
    # Pumps
    {
        "Equipment_Category": "Pumps",
        "Equipment_Item": "Base-mounted",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Pumps",
        "Equipment_Item": "Pipe-mounted",
        "Service_Life_Years": 10,
    },
    {
        "Equipment_Category": "Pumps",
        "Equipment_Item": "Sump and well",
        "Service_Life_Years": 10,
    },
    {
        "Equipment_Category": "Pumps",
        "Equipment_Item": "Condensate",
        "Service_Life_Years": 15,
    },
    # Engines and Turbines
    {
        "Equipment_Category": "Engines and Turbines",
        "Equipment_Item": "Reciprocating Engines",
        "Service_Life_Years": 20,
    },
    {
        "Equipment_Category": "Engines and Turbines",
        "Equipment_Item": "Steam Turbines",
        "Service_Life_Years": 30,
    },
    # Electrical Components
    {
        "Equipment_Category": "Electrical Components",
        "Equipment_Item": "Electric Motors",
        "Service_Life_Years": 16,
    },
    {
        "Equipment_Category": "Electrical Components",
        "Equipment_Item": "Motor Starters",
        "Service_Life_Years": 15,
    },
    {
        "Equipment_Category": "Electrical Components",
        "Equipment_Item": "Electric Transformers",
        "Service_Life_Years": 28,
    },
    # Air Handling Units
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Constant volume",
        "Service_Life_Years": 25,
    },
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Dual duct",
        "Service_Life_Years": 32,
    },
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Multi-zone",
        "Service_Life_Years": 22,
    },
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Single-zone",
        "Service_Life_Years": 16,
    },
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Variable air volume",
        "Service_Life_Years": 18,
    },
    {
        "Equipment_Category": "Air Handling Units",
        "Equipment_Item": "Variable volume, variable temp",
        "Service_Life_Years": 19,
    },
]


def convert_to_service_life_format(data):
    """
    Convert the provided data to the format expected by the service life manager.

    Args:
        data: List of dictionaries with equipment service life data

    Returns:
        list: A list of dictionaries in the service life format
    """
    service_life_data = []

    for item in data:
        # Create a combined equipment type string
        equipment_type = f"{item['Equipment_Category']} - {item['Equipment_Item']}"

        # Use the service life years as median, and calculate min/max as 70%/130%
        median_years = item["Service_Life_Years"]
        min_years = int(median_years * 0.7)  # 70% of median
        max_years = int(median_years * 1.3)  # 130% of median

        service_life_data.append(
            {
                "Equipment Type": equipment_type,
                "Median Years": median_years,
                "Min Years": min_years,
                "Max Years": max_years,
                "Source": "Energize Denver",
            }
        )

    return service_life_data


def save_to_csv(data, output_path):
    """
    Save the service life data to a CSV file.

    Args:
        data: List of dictionaries with service life data
        output_path: Path to save the CSV file
    """
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "Equipment Type",
                "Median Years",
                "Min Years",
                "Max Years",
                "Source",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            for row in data:
                writer.writerow(row)

        print(f"Saved {len(data)} entries to {output_path}")

    except Exception as e:
        print(f"Error saving to CSV: {e}")


def main():
    """Main function."""
    # Convert the data to the service life format
    service_life_data = convert_to_service_life_format(data)

    # Save to CSV
    output_path = "nexusml/ingest/reference/service-life/energize-denver/energize_denver_service_life.csv"
    save_to_csv(service_life_data, output_path)

    print("CSV creation complete!")


if __name__ == "__main__":
    main()
