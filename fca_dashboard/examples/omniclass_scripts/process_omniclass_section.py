#!/usr/bin/env python
"""
Script to process a specific OmniClass section.

This script allows processing a specific OmniClass table or division
without having to run the entire dataset.

Usage:
    python fca_dashboard/examples/omniclass_scripts/process_omniclass_section.py [section_name]

Available sections:
    table21_elements - Table 21 (Elements)
    table22_work_results - Table 22 (Work Results)
    table23_products - Table 23 (Products)
    table22_div22_plumbing - Table 22, Division 22 (Plumbing)
    table22_div23_hvac - Table 22, Division 23 (HVAC)
    table23_div23_hvac - Table 23, Division 23 (HVAC Products)
    all - Process all sections
"""
import os
import subprocess
import sys
from datetime import datetime

# Configuration
OUTPUT_DIR = "fca_dashboard/generator/ingest/output"

# Define the ranges to process
SECTIONS = {
    "table21_elements": {
        "start": 2050,
        "end": 2690,
        "description": "Table 21 (Elements)"
    },
    "table22_work_results": {
        "start": 2690,
        "end": 11636,
        "description": "Table 22 (Work Results)"
    },
    "table23_products": {
        "start": 11636,
        "end": 18533,
        "description": "Table 23 (Products)"
    },
    "table22_div22_plumbing": {
        "start": 6510,
        "end": 6798,
        "description": "Table 22, Division 22 (Work Results - Plumbing)"
    },
    "table22_div23_hvac": {
        "start": 6798,
        "end": 7232,
        "description": "Table 22, Division 23 (Work Results - HVAC)"
    },
    "table23_div23_hvac": {
        "start": 14473,
        "end": 14663,
        "description": "Table 23, Division 23 (Products - HVAC)"
    }
}

def process_section(section_name):
    """Process a specific section."""
    if section_name not in SECTIONS:
        print(f"Error: Unknown section '{section_name}'")
        print("Available sections:")
        for name, info in SECTIONS.items():
            print(f"  {name} - {info['description']}")
        return False
    
    section = SECTIONS[section_name]
    output_file = f"{OUTPUT_DIR}/omniclass_{section_name}.csv"
    
    print("\n" + "="*80)
    print(f"Processing {section['description']}")
    print(f"Rows {section['start']} to {section['end']}")
    print(f"Output file: {output_file}")
    print("="*80 + "\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Build the command - use the Python from the virtual environment
    cmd = [
        ".venv/Scripts/python",
        "-m",
        "fca_dashboard.examples.omniclass_description_generator_example",
        f"--start={section['start']}",
        f"--end={section['end']}",
        f"--output-file={output_file}"
    ]
    
    # Run the command
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        print(f"Successfully processed {section['description']}")
        print(f"Output saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {section['description']}: {e}")
        return False

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python fca_dashboard/examples/omniclass_scripts/process_omniclass_section.py [section_name]")
        print("\nAvailable sections:")
        for name, info in SECTIONS.items():
            print(f"  {name} - {info['description']}")
        print("  all - Process all sections")
        return 1
    
    section_name = sys.argv[1]
    
    if section_name == "all":
        success = True
        for name in SECTIONS:
            if not process_section(name):
                success = False
        return 0 if success else 1
    else:
        return 0 if process_section(section_name) else 1

if __name__ == "__main__":
    sys.exit(main())