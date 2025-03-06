#!/usr/bin/env python
"""
Modular Equipment Classification

This script takes input data with any column structure,
maps it to the expected model format, classifies it,
and outputs the results in a format ready for database import.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from nexusml.core.dynamic_mapper import DynamicFieldMapper
from nexusml.core.eav_manager import EAVManager
from nexusml.core.model import predict_with_enhanced_model, train_enhanced_model


def process_any_input_file(input_file, output_file=None, config_file=None):
    """
    Process equipment data with any column structure.

    Args:
        input_file: Path to input file (CSV, Excel)
        output_file: Path to output CSV file
        config_file: Path to classification config file
    """
    # Determine file type and load data
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext == ".csv":
        df = pd.read_csv(input_file)
    elif file_ext in [".xlsx", ".xls"]:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file type: {file_ext}")

    # Create dynamic mapper
    mapper = DynamicFieldMapper(config_file)

    # Map input data to model format
    print("Mapping input fields to model format...")
    mapped_df = mapper.map_dataframe(df)

    # Train model
    print("Training classification model...")
    model, _ = train_enhanced_model()

    # Get classification targets and DB field requirements
    classification_targets = mapper.get_classification_targets()
    db_field_mapping = mapper.get_required_db_fields()

    # Create EAV manager
    eav_manager = EAVManager()

    # Process each row
    results = []
    for i, (idx, row) in enumerate(mapped_df.iterrows()):
        # Create description from available text fields
        description_parts = []
        for field in [
            "Asset Category",
            "Equip Name ID",
            "Precon System",
            "Sub System Type",
        ]:
            if field in row and pd.notna(row[field]) and str(row[field]).strip():
                description_parts.append(str(row[field]))

        description = " ".join(description_parts)
        service_life = (
            float(row.get("Service Life", 0))
            if pd.notna(row.get("Service Life", 0))
            else 0.0
        )

        # Get prediction
        prediction = predict_with_enhanced_model(model, description, service_life)

        # Get EAV template info
        equipment_type = prediction.get("Equipment_Category", "Unknown")
        template = eav_manager.get_equipment_template(equipment_type)

        # Process for database integration
        db_fields = {}
        for target, field_info in db_field_mapping.items():
            if target in prediction:
                db_fields[target] = {
                    "value": prediction[target],
                    "table": field_info.get("table", ""),
                    "field": field_info.get("field", ""),
                    "id_field": field_info.get("id_field", ""),
                }

        # Combine all results
        result = {
            "original_data": df.iloc[i].to_dict(),
            "classification": prediction,
            "db_fields": db_fields,
            "eav_template": {
                "equipment_type": equipment_type,
                "required_attributes": template.get("required_attributes", []),
                "classification_ids": eav_manager.get_classification_ids(
                    equipment_type
                ),
            },
        }

        results.append(result)

        # Show progress
        if (i + 1) % 10 == 0 or i == len(mapped_df) - 1:
            print(f"Processed {i + 1}/{len(mapped_df)} items")

    # Determine output file
    if output_file is None:
        output_file = os.path.splitext(input_file)[0] + "_classified.json"

    # Save results based on extension
    out_ext = os.path.splitext(output_file)[1].lower()
    if out_ext == ".json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
    elif out_ext == ".csv":
        # Flatten results into a CSV-friendly format
        flat_results = []
        for result in results:
            flat_result = {}
            # Add original data
            for k, v in result["original_data"].items():
                flat_result[f"original_{k}"] = v
            # Add classifications
            for k, v in result["classification"].items():
                flat_result[f"class_{k}"] = v
            # Add DB mappings
            for target, info in result["db_fields"].items():
                flat_result[f"db_{target}_value"] = info["value"]
                flat_result[f"db_{target}_table"] = info["table"]
            flat_results.append(flat_result)

        pd.DataFrame(flat_results).to_csv(output_file, index=False)
    else:
        # Default to JSON
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Classification complete! Results saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify equipment data with any column structure"
    )
    parser.add_argument("input_file", help="Path to input file (CSV or Excel)")
    parser.add_argument("--output", "-o", help="Path to output file (JSON or CSV)")
    parser.add_argument(
        "--config", "-c", help="Path to classification configuration file"
    )

    args = parser.parse_args()
    process_any_input_file(args.input_file, args.output, args.config)
