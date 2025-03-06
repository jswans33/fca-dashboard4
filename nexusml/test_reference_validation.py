#!/usr/bin/env python
"""
Test Reference Data Validation

This script demonstrates how to use the reference data validation functionality
to ensure data quality across all reference data sources.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from nexusml.core.reference.manager import ReferenceManager


def test_reference_validation():
    """Test the reference data validation functionality."""
    print("Testing reference data validation...")

    # Create reference manager
    manager = ReferenceManager()

    # Load all reference data
    print("\nLoading reference data...")
    manager.load_all()

    # Validate all reference data
    print("\nValidating reference data...")
    validation_results = manager.validate_data()

    # Print validation results
    print("\nValidation Results:")
    print("==================")

    for source_name, result in validation_results.items():
        print(f"\n{source_name.upper()}:")
        print(f"  Loaded: {result['loaded']}")

        if result["issues"]:
            print(f"  Issues ({len(result['issues'])}):")
            for issue in result["issues"]:
                print(f"    - {issue}")
        else:
            print("  Issues: None")

        print("  Stats:")
        for stat_name, stat_value in result["stats"].items():
            # Format lists to be more readable
            if isinstance(stat_value, list) and len(stat_value) > 5:
                stat_display = f"{stat_value[:5]} ... ({len(stat_value)} total)"
            else:
                stat_display = stat_value
            print(f"    - {stat_name}: {stat_display}")

    # Save validation results to file
    output_file = project_root / "test_output" / "reference_validation_results.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(validation_results, f, indent=2, default=str)

    print(f"\nValidation results saved to {output_file}")

    # Return validation results
    return validation_results


def main():
    """Main function."""
    test_reference_validation()


if __name__ == "__main__":
    main()
