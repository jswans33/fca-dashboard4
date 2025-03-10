#!/usr/bin/env python
"""
Random Equipment Guessing Example

This script demonstrates how to use the equipment classifier model to make predictions
on random or user-provided equipment descriptions.
"""

import argparse
import random

from nexusml.core.model import EquipmentClassifier

# Sample equipment components for generating random descriptions
MANUFACTURERS = [
    "Trane",
    "Carrier",
    "York",
    "Daikin",
    "Johnson Controls",
    "Lennox",
    "Mitsubishi",
    "Greenheck",
    "Siemens",
    "Honeywell",
    "Grundfos",
    "Armstrong",
    "Bell & Gossett",
    "Caterpillar",
    "Cummins",
    "Kohler",
    "ABB",
    "Schneider",
    "Eaton",
    "GE",
]

EQUIPMENT_TYPES = [
    "Chiller",
    "Boiler",
    "Air Handler",
    "Rooftop Unit",
    "Heat Pump",
    "Fan Coil Unit",
    "Pump",
    "Cooling Tower",
    "Generator",
    "Transformer",
    "Switchgear",
    "Motor Control Center",
    "Variable Frequency Drive",
    "Exhaust Fan",
    "Fire Alarm Panel",
    "Fire Sprinkler",
    "Compressor",
    "Condenser",
    "Evaporator",
    "Heat Exchanger",
]

ATTRIBUTES = [
    "500 ton",
    "250 HP",
    "1000 kW",
    "480V",
    "2000 CFM",
    "50 GPM",
    "100 PSI",
    "3-phase",
    "high-efficiency",
    "variable-speed",
    "water-cooled",
    "air-cooled",
    "centrifugal",
    "reciprocating",
    "screw",
    "scroll",
    "plate and frame",
    "shell and tube",
    "vertical inline",
    "base mounted",
]

LOCATIONS = [
    "for mechanical room",
    "for rooftop installation",
    "for basement",
    "for outdoor use",
    "for data center",
    "for hospital",
    "for office building",
    "for school",
    "for industrial facility",
    "for commercial kitchen",
]


def generate_random_description():
    """Generate a random equipment description."""
    manufacturer = random.choice(MANUFACTURERS)
    equipment_type = random.choice(EQUIPMENT_TYPES)
    attributes = random.sample(ATTRIBUTES, k=random.randint(1, 3))
    location = random.choice(LOCATIONS)

    model = f"{manufacturer[0]}{random.randint(100, 9999)}"

    description = (
        f"{manufacturer} {model} {equipment_type} {' '.join(attributes)} {location}"
    )
    return description


def main():
    """Main function to demonstrate random equipment guessing."""
    parser = argparse.ArgumentParser(
        description="Test the equipment classifier with random descriptions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="nexusml/output/models/equipment_classifier_20250306_161707.pkl",
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of random samples to generate",
    )
    parser.add_argument(
        "--custom",
        type=str,
        default=None,
        help="Custom equipment description to classify",
    )
    args = parser.parse_args()

    # Load the model
    print(f"Loading model from {args.model_path}")
    classifier = EquipmentClassifier()
    classifier.load_model(args.model_path)
    print("Model loaded successfully\n")

    # Process custom description if provided
    if args.custom:
        print(f"Classifying custom description: {args.custom}")
        prediction = classifier.predict(args.custom)
        print_prediction(args.custom, prediction)
        print("\n" + "-" * 80 + "\n")

    # Generate and process random descriptions
    print(f"Generating {args.num_samples} random equipment descriptions:")
    for i in range(args.num_samples):
        description = generate_random_description()
        prediction = classifier.predict(description)
        print(f"\nRandom Sample #{i+1}:")
        print_prediction(description, prediction)
        print("-" * 80)


def print_prediction(description, prediction):
    """Print the prediction results in a readable format."""
    print(f"Description: {description}")
    print(f"Equipment Category: {prediction.get('category_name', 'Unknown')}")
    print(f"System Type: {prediction.get('mcaa_system_category', 'Unknown')}")
    print(f"Equipment Type: {prediction.get('Equipment_Type', 'Unknown')}")
    print(f"MasterFormat Class: {prediction.get('MasterFormat_Class', 'Unknown')}")


if __name__ == "__main__":
    main()
