#!/usr/bin/env python
"""
Test the modular classification system with different input formats.
"""

import os
from pathlib import Path

import pandas as pd
from classify_equipment import process_any_input_file

# Create test data with different column names
test_data1 = pd.DataFrame(
    {
        "Asset Name": ["Centrifugal Chiller", "Air Handling Unit", "Boiler"],
        "Trade": ["H", "H", "H"],
        "System Category": ["Chiller Plant", "HVAC", "Heating Plant"],
        "Sub System Type": ["Water-Cooled", "Air Handler", "Hot Water"],
        "Manufacturer": ["York", "Trane", "Cleaver Brooks"],
        "Model Number": ["YK-8000", "TAHN-5000", "CB-200"],
        "Size": [800, 5000, 2500],
        "Unit": ["Tons", "CFM", "MBH"],
        "Service Life": [20, 15, 25],
    }
)

test_data2 = pd.DataFrame(
    {
        "Equipment Type": ["Pump", "Cooling Tower", "Fan"],
        "Discipline": ["P", "H", "H"],
        "System": ["Pumping System", "Cooling System", "Ventilation"],
        "Equipment Subtype": ["Centrifugal", "Open", "Centrifugal"],
        "Vendor": ["Grundfos", "SPX", "Cook"],
        "Model": ["CRE-5", "NC-8400", "CPS-3000"],
        "Capacity": [100, 900, 3000],
        "Capacity Unit": ["GPM", "Tons", "CFM"],
        "Expected Life (Years)": [15, 20, 15],
    }
)


def run_test():
    """Run the test with different input formats."""
    # Save test data
    output_dir = Path(__file__).resolve().parent / "test_output"
    os.makedirs(output_dir, exist_ok=True)

    test_file1 = output_dir / "test_data1.csv"
    test_file2 = output_dir / "test_data2.csv"

    test_data1.to_csv(test_file1, index=False)
    test_data2.to_csv(test_file2, index=False)

    # Process both test files
    print("Processing test file 1...")
    results1 = process_any_input_file(test_file1)

    print("\nProcessing test file 2...")
    results2 = process_any_input_file(test_file2)

    print("\nTest completed successfully!")

    return results1, results2


if __name__ == "__main__":
    run_test()
