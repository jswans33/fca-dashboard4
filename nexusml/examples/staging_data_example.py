"""
Staging Data Classification Example

This example demonstrates how to use the ML model with staging data that has different column names.
It shows the complete workflow from staging data to master database field mapping.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add the parent directory to the path to import nexusml modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nexusml.core.data_mapper import map_staging_to_model_input
from nexusml.core.model import EquipmentClassifier


def create_test_staging_data():
    """Create a sample staging data CSV file for testing."""
    data = [
        {
            "Asset Name": "Centrifugal Chiller",
            "Asset Tag": "CH-01",
            "Trade": "H",
            "Equip Served by": "AHU-1",
            "Vendor": "Johnson Controls",
            "Manufacturer": "York",
            "Model Number": "YK-EP-8400",
            "Serial Number": "123456",
            "Size": "800",
            "Unit": "Tons",
            "Motor HP": "0",
            "Estimated Operating Hours": "8760",
            "USAssetID": "A123",
            "System Category": "Chiller Plant",
            "Sub System Type": "Water-Cooled",
            "Sub System Classification": "Centrifugal",
            "Asset Size (Rounded up)": "800",
            "ID Unit": "Tons",
            "Service Life": "20",
            "Date Installed": "2020-01-15",
            "Floor": "1",
            "Area": "East Wing",
            "Room Number": "M101",
            "Notes": "Main building chiller",
        },
        {
            "Asset Name": "Air Handling Unit",
            "Asset Tag": "AHU-01",
            "Trade": "H",
            "Equip Served by": "RM-101",
            "Vendor": "Trane",
            "Manufacturer": "Trane",
            "Model Number": "T-AHU-5000",
            "Serial Number": "234567",
            "Size": "5000",
            "Unit": "CFM",
            "Motor HP": "7.5",
            "Estimated Operating Hours": "8760",
            "USAssetID": "A124",
            "System Category": "HVAC",
            "Sub System Type": "Air Handler",
            "Sub System Classification": "VAV",
            "Asset Size (Rounded up)": "5000",
            "ID Unit": "CFM",
            "Service Life": "15",
            "Date Installed": "2020-02-15",
            "Floor": "2",
            "Area": "North Wing",
            "Room Number": "M201",
            "Notes": "Primary AHU for offices",
        },
    ]

    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(data)
    csv_path = output_dir / "test_staging_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


def main():
    """Run the staging data classification example."""
    print("Staging Data Classification Example")
    print("==================================")

    # Create test staging data
    print("\nCreating test staging data...")
    staging_data_path = create_test_staging_data()
    print(f"Test staging data created at: {staging_data_path}")

    # Load staging data
    print("\nLoading staging data...")
    staging_df = pd.read_csv(staging_data_path)
    print(f"Loaded {len(staging_df)} equipment records")

    # Initialize and train the equipment classifier
    print("\nInitializing and training the Equipment Classifier...")
    classifier = EquipmentClassifier()
    classifier.train()

    # Process each equipment record
    print("\nClassifying equipment records...")
    results = []

    for i, (idx, row) in enumerate(staging_df.iterrows()):
        print(f"\nProcessing equipment {i+1}: {row['Asset Name']}")

        # Create description from relevant fields
        description_parts = []
        for field in [
            "Asset Name",
            "System Category",
            "Sub System Type",
            "Manufacturer",
            "Model Number",
        ]:
            if field in row and pd.notna(row[field]) and row[field] != "":
                description_parts.append(str(row[field]))

        description = " ".join(description_parts)
        service_life = (
            float(row.get("Service Life", 20.0))
            if pd.notna(row.get("Service Life", 0))
            else 20.0
        )
        asset_tag = str(row.get("Asset Tag", ""))

        # Get prediction with master DB mapping
        prediction = classifier.predict(description, service_life, asset_tag)

        # Print key results
        print(f"  Equipment Category: {prediction['Equipment_Category']}")
        print(f"  System Type: {prediction['System_Type']}")
        print(f"  Uniformat Class: {prediction['Uniformat_Class']}")
        print(f"  MasterFormat Class: {prediction['MasterFormat_Class']}")

        # Print master DB mapping
        print("\n  Master DB Mapping:")
        for key, value in prediction["master_db_mapping"].items():
            print(f"    {key}: {value}")

        # Add to results
        results.append({"original_data": row.to_dict(), "prediction": prediction})

    # Save results to JSON
    results_file = (
        Path(staging_data_path).parent / "staging_classification_results.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
