"""
Example script demonstrating how to use the OmniClass generator in NexusML.

This script shows how to extract OmniClass data from Excel files and generate
descriptions using the Claude API.
"""

import os
from pathlib import Path

from nexusml import (
    OmniClassDescriptionGenerator,
    extract_omniclass_data,
    generate_descriptions,
)


def main():
    """Run the OmniClass generator example."""
    # Set up paths
    input_dir = "files/omniclass_tables"
    output_csv = "nexusml/ingest/generator/data/omniclass.csv"
    output_with_descriptions = "nexusml/ingest/generator/data/omniclass_with_descriptions.csv"

    # Extract OmniClass data from Excel files
    print(f"Extracting OmniClass data from {input_dir}...")
    df = extract_omniclass_data(input_dir=input_dir, output_file=output_csv, file_pattern="*.xlsx")
    print(f"Extracted {len(df)} OmniClass codes to {output_csv}")

    # Check if ANTHROPIC_API_KEY is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY environment variable not set.")
        print("Description generation will not work without an API key.")
        print("Please set the ANTHROPIC_API_KEY environment variable and try again.")
        return

    # Generate descriptions for a small subset of the data
    print("Generating descriptions for a sample of OmniClass codes...")
    result_df = generate_descriptions(
        input_file=output_csv,
        output_file=output_with_descriptions,
        start_index=0,
        end_index=5,  # Only process 5 rows for this example
        batch_size=5,
        description_column="Description",
    )

    print(f"Generated descriptions for {len(result_df)} OmniClass codes")
    print(f"Results saved to {output_with_descriptions}")

    # Display sample results
    print("\nSample results:")
    for _, row in result_df.head().iterrows():
        print(f"Code: {row['OmniClass_Code']}")
        print(f"Title: {row['OmniClass_Title']}")
        print(f"Description: {row['Description']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
