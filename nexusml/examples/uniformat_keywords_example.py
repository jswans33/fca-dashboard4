#!/usr/bin/env python
"""
Uniformat Keywords Example

This script demonstrates how to use the Uniformat keywords functionality
to find Uniformat codes by keyword and enrich equipment data.
"""

import pandas as pd

from nexusml.core.reference.manager import ReferenceManager


def main():
    # Initialize the reference manager
    ref_manager = ReferenceManager()

    # Load all reference data
    ref_manager.load_all()

    # Example 1: Find Uniformat codes by keyword
    print("\nExample 1: Find Uniformat codes by keyword")
    print("-------------------------------------------")

    keywords = ["Air Barriers", "Boilers", "Elevators", "Pumps"]

    for keyword in keywords:
        print(f"\nSearching for keyword: {keyword}")
        results = ref_manager.find_uniformat_codes_by_keyword(keyword)

        if results:
            print(f"Found {len(results)} results:")
            for result in results:
                print(f"  - Keyword: {result['keyword']}")
                print(f"    Uniformat Code: {result['uniformat_code']}")
                print(f"    MasterFormat Code: {result['masterformat_code']}")

                # Get the description for the Uniformat code
                if result["uniformat_code"]:
                    description = ref_manager.get_uniformat_description(
                        result["uniformat_code"]
                    )
                    if description:
                        print(f"    Description: {description}")
        else:
            print("No results found.")

    # Example 2: Enrich equipment data with Uniformat and MasterFormat information
    print("\nExample 2: Enrich equipment data")
    print("--------------------------------")

    # Create a sample DataFrame with equipment data
    equipment_data = [
        {
            "equipment_id": "EQ001",
            "equipment_name": "Air Handling Unit",
            "uniformat_code": None,
            "masterformat_code": None,
        },
        {
            "equipment_id": "EQ002",
            "equipment_name": "Boiler",
            "uniformat_code": None,
            "masterformat_code": None,
        },
        {
            "equipment_id": "EQ003",
            "equipment_name": "Chiller",
            "uniformat_code": "D3030.10",
            "masterformat_code": None,
        },
        {
            "equipment_id": "EQ004",
            "equipment_name": "Pump",
            "uniformat_code": None,
            "masterformat_code": None,
        },
        {
            "equipment_id": "EQ005",
            "equipment_name": "Elevator",
            "uniformat_code": None,
            "masterformat_code": None,
        },
    ]

    df = pd.DataFrame(equipment_data)
    print("\nOriginal DataFrame:")
    print(df)

    # Enrich the DataFrame with reference information
    enriched_df = ref_manager.enrich_equipment_data(df)

    print("\nEnriched DataFrame:")
    print(
        enriched_df[
            [
                "equipment_id",
                "equipment_name",
                "uniformat_code",
                "uniformat_description",
                "masterformat_code",
                "masterformat_description",
            ]
        ]
    )

    # Show which codes were found by keyword matching
    print("\nCodes found by keyword matching:")
    for _, row in enriched_df.iterrows():
        if pd.notna(row["uniformat_code"]) and row["equipment_id"] in [
            "EQ001",
            "EQ002",
            "EQ004",
            "EQ005",
        ]:
            print(
                f"{row['equipment_name']}: {row['uniformat_code']} - {row['uniformat_description']}"
            )


if __name__ == "__main__":
    main()
