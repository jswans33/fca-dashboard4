#!/usr/bin/env python
"""
Improved script to run the OmniClass description generator for different tables and divisions,
saving each to a separate output file.

This script:
1. Creates an output directory if it doesn't exist
2. Processes each OmniClass table/division one at a time
3. Saves each to a separate output file
4. Provides clear progress information and error handling
"""
import os
import subprocess
import sys
import time
from datetime import datetime

# Configuration
OUTPUT_DIR = "fca_dashboard/generator/ingest/output"
LOG_FILE = f"omniclass_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Define the ranges to process
RANGES = [
    {
        "name": "table21_elements",
        "start": 2050,
        "end": 2690,
        "description": "Table 21 (Elements)"
    },
    {
        "name": "table22_work_results",
        "start": 2690,
        "end": 11636,
        "description": "Table 22 (Work Results)"
    },
    {
        "name": "table23_products",
        "start": 11636,
        "end": 18533,
        "description": "Table 23 (Products)"
    },
    {
        "name": "table22_div22_plumbing",
        "start": 6510,
        "end": 6798,
        "description": "Table 22, Division 22 (Work Results - Plumbing)"
    },
    {
        "name": "table22_div23_hvac",
        "start": 6798,
        "end": 7232,
        "description": "Table 22, Division 23 (Work Results - HVAC)"
    },
    {
        "name": "table23_div23_hvac",
        "start": 14473,
        "end": 14663,
        "description": "Table 23, Division 23 (Products - HVAC)"
    }
]

def log_message(message, also_print=True):
    """Log a message to the log file and optionally print it."""
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        f.write(log_line)
    if also_print:
        print(message)

def run_command(cmd):
    """Run a command and return the result."""
    try:
        log_message(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log_message(f"Command succeeded with output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        log_message(f"Command failed with error:\n{e.stderr}")
        return False

def main():
    """Main function to run the OmniClass description generator."""
    # Create log file
    with open(LOG_FILE, "w") as f:
        f.write(f"OmniClass Description Generator Run - {datetime.now()}\n\n")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        log_message(f"Created output directory: {OUTPUT_DIR}")
    
    # Process each range
    successful_ranges = []
    failed_ranges = []
    
    for i, range_info in enumerate(RANGES):
        output_file = f"{OUTPUT_DIR}/omniclass_{range_info['name']}.csv"
        
        log_message("\n" + "="*80)
        log_message(f"Processing {i+1}/{len(RANGES)}: {range_info['description']}")
        log_message(f"Rows {range_info['start']} to {range_info['end']}")
        log_message(f"Output file: {output_file}")
        log_message("="*80 + "\n")
        
        # Build the command - use the Python from the virtual environment
        cmd = [
            ".venv/Scripts/python",
            "-m",
            "fca_dashboard.examples.omniclass_description_generator_example",
            f"--start={range_info['start']}",
            f"--end={range_info['end']}",
            f"--output-file={output_file}"
        ]
        
        # Run the command
        success = run_command(cmd)
        
        if success:
            log_message(f"Successfully processed {range_info['description']}")
            log_message(f"Output saved to {output_file}")
            successful_ranges.append(range_info)
        else:
            log_message(f"Failed to process {range_info['description']}")
            failed_ranges.append(range_info)
        
        # Add a small delay between runs
        if i < len(RANGES) - 1:
            log_message("Waiting 2 seconds before next run...")
            time.sleep(2)
    
    # Print summary
    log_message("\n\n" + "="*80)
    log_message("SUMMARY")
    log_message("="*80)
    
    log_message(f"\nSuccessfully processed {len(successful_ranges)}/{len(RANGES)} ranges:")
    for range_info in successful_ranges:
        log_message(f"- {range_info['description']}")
    
    if failed_ranges:
        log_message(f"\nFailed to process {len(failed_ranges)}/{len(RANGES)} ranges:")
        for range_info in failed_ranges:
            log_message(f"- {range_info['description']}")
        log_message("\nCheck the log file for details on the failures.")
    
    log_message(f"\nLog file: {LOG_FILE}")
    log_message(f"Output files are in {OUTPUT_DIR}/")
    
    # Return appropriate exit code
    return 0 if not failed_ranges else 1

if __name__ == "__main__":
    sys.exit(main())