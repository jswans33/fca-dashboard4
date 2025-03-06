"""
Integrated Equipment Classifier Example

This example demonstrates the comprehensive equipment classification model that integrates:
1. Multiple classification systems (OmniClass, MasterFormat, Uniformat)
2. EAV (Entity-Attribute-Value) structure for flexible equipment attributes
3. ML capabilities to fill in missing attribute data
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add the parent directory to the path to import nexusml modules
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nexusml.core.eav_manager import EAVManager
from nexusml.core.model import EquipmentClassifier


def main():
    """Run the integrated equipment classifier example."""
    print("Integrated Equipment Classifier Example")
    print("=======================================")

    # Create output directory
    output_dir = Path(__file__).resolve().parent.parent / "output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the equipment classifier
    print("\nInitializing Equipment Classifier...")
    classifier = EquipmentClassifier()

    # Train the model
    print("\nTraining the model...")
    classifier.train()

    # Example equipment descriptions
    examples = [
        {
            "description": "Centrifugal chiller with 500 tons cooling capacity, 0.6 kW/ton efficiency, using R-134a refrigerant",
            "service_life": 20.0,
        },
        {
            "description": "Air handling unit with 10,000 CFM airflow, 2.5 inch WG static pressure, and MERV 13 filters",
            "service_life": 15.0,
        },
        {
            "description": "Hot water boiler with 2,000 MBH heating capacity, 85% efficiency, natural gas fired",
            "service_life": 25.0,
        },
        {
            "description": "Cooling tower with 600 tons capacity, 1,800 GPM flow rate, induced draft design",
            "service_life": 15.0,
        },
        {
            "description": "Centrifugal pump with 500 GPM flow rate, 60 ft head pressure, 5 HP motor",
            "service_life": 15.0,
        },
    ]

    # Make predictions for each example
    print("\nMaking predictions and generating EAV templates...")
    results = []

    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example['description'][:50]}...")

        # Make prediction
        prediction = classifier.predict(example["description"], example["service_life"])

        # Extract basic classification results
        basic_result = {
            "description": example["description"],
            "service_life": example["service_life"],
            "Equipment_Category": prediction["Equipment_Category"],
            "Uniformat_Class": prediction["Uniformat_Class"],
            "System_Type": prediction["System_Type"],
            "MasterFormat_Class": prediction["MasterFormat_Class"],
            "OmniClass_ID": prediction.get("OmniClass_ID", ""),
            "Uniformat_ID": prediction.get("Uniformat_ID", ""),
        }

        print(f"Predicted Equipment Category: {basic_result['Equipment_Category']}")
        print(f"Predicted MasterFormat Class: {basic_result['MasterFormat_Class']}")
        print(f"Predicted OmniClass ID: {basic_result['OmniClass_ID']}")

        # Get the attribute template
        template = prediction.get("attribute_template", {})

        # Try to extract attributes from the description
        equipment_type = prediction["Equipment_Category"]
        extracted_attributes = {}

        if hasattr(classifier, "predict_attributes"):
            extracted_attributes = classifier.predict_attributes(
                equipment_type, example["description"]
            )

            if extracted_attributes:
                print("\nExtracted attributes from description:")
                for attr, value in extracted_attributes.items():
                    print(f"  {attr}: {value}")

        # Add results to the list
        basic_result["extracted_attributes"] = extracted_attributes
        basic_result["attribute_template"] = template
        results.append(basic_result)

    # Save results to JSON file
    results_file = output_dir / "integrated_classifier_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Generate a complete EAV template example
    print("\nGenerating complete EAV template example...")
    eav_manager = EAVManager()

    # Get templates for different equipment types
    equipment_types = ["Chiller", "Air Handler", "Boiler", "Pump", "Cooling Tower"]
    templates = {}

    for eq_type in equipment_types:
        templates[eq_type] = eav_manager.generate_attribute_template(eq_type)

    # Save templates to JSON file
    templates_file = output_dir / "equipment_templates.json"
    with open(templates_file, "w") as f:
        json.dump(templates, f, indent=2)

    print(f"Equipment templates saved to {templates_file}")

    # Demonstrate attribute validation
    print("\nDemonstrating attribute validation...")

    # Example: Valid attributes for a chiller
    valid_attributes = {
        "cooling_capacity_tons": 500,
        "efficiency_kw_per_ton": 0.6,
        "refrigerant_type": "R-134a",
        "chiller_type": "Centrifugal",
    }

    # Example: Invalid attributes for a chiller (missing required, has unknown)
    invalid_attributes = {
        "cooling_capacity_tons": 500,
        "unknown_attribute": "value",
        "chiller_type": "Centrifugal",
    }

    # Validate attributes
    valid_result = eav_manager.validate_attributes("Chiller", valid_attributes)
    invalid_result = eav_manager.validate_attributes("Chiller", invalid_attributes)

    print("\nValid attributes validation result:")
    print(f"  Missing required: {valid_result['missing_required']}")
    print(f"  Unknown attributes: {valid_result['unknown']}")

    print("\nInvalid attributes validation result:")
    print(f"  Missing required: {invalid_result['missing_required']}")
    print(f"  Unknown attributes: {invalid_result['unknown']}")

    # Demonstrate filling missing attributes
    print("\nDemonstrating filling missing attributes...")

    # Example: Partial attributes for a chiller
    partial_attributes = {"cooling_capacity_tons": 500, "chiller_type": "Centrifugal"}

    # Description with additional information
    description = "Centrifugal chiller with 500 tons cooling capacity, 0.6 kW/ton efficiency, using R-134a refrigerant"

    # Fill missing attributes
    filled_attributes = eav_manager.fill_missing_attributes(
        "Chiller", partial_attributes, description, classifier
    )

    print("\nPartial attributes:")
    print(json.dumps(partial_attributes, indent=2))

    print("\nFilled attributes:")
    print(json.dumps(filled_attributes, indent=2))

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
