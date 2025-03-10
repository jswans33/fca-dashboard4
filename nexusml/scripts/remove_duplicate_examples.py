#!/usr/bin/env python
"""
Remove Duplicate Files

This script identifies and removes duplicate files between different directories:
1. Duplicate example scripts between fca_dashboard/examples and nexusml/examples
2. Duplicate output files between nexusml/output, nexusml/outputs, and outputs

It compares files by name and provides options to remove the duplicates.

Usage:
    python -m nexusml.scripts.remove_duplicate_examples [--dry-run] [--auto-remove] [--examples-only] [--outputs-only]

Options:
    --dry-run       Only show what would be removed, don't actually remove files
    --auto-remove   Automatically remove all identified duplicates without prompting
    --examples-only Only check for duplicate example files
    --outputs-only  Only check for duplicate output files
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("remove_duplicates")

# Define directories
FCA_EXAMPLES_DIR = Path("fca_dashboard/examples")
NEXUSML_EXAMPLES_DIR = Path("nexusml/examples")

# Output directories
PRIMARY_OUTPUT_DIR = Path("nexusml/output")
SECONDARY_OUTPUT_DIRS = [Path("nexusml/outputs"), Path("outputs")]

# List of subdirectories in nexusml/examples to check for duplicates
NEXUSML_SUBDIRS = [
    "data_loading",
    "evaluation",
    "feature_engineering",
    "legacy",
    "model_building",
    "pipeline",
    "prediction",
    "utilities",
    "visualization",
]


def find_duplicate_examples():
    """
    Find duplicate example files between fca_dashboard/examples and nexusml/examples.

    Returns:
        List of tuples containing (original_file_path, duplicate_file_path)
    """
    duplicates = []

    # Check if directories exist
    if not FCA_EXAMPLES_DIR.exists():
        logger.error(f"Source directory {FCA_EXAMPLES_DIR} does not exist")
        return duplicates

    if not NEXUSML_EXAMPLES_DIR.exists():
        logger.error(f"Target directory {NEXUSML_EXAMPLES_DIR} does not exist")
        return duplicates

    # Get all Python files in fca_dashboard/examples
    fca_files = [f for f in FCA_EXAMPLES_DIR.glob("*.py") if f.name != "__init__.py"]

    # Check for duplicates in the main nexusml/examples directory
    for fca_file in fca_files:
        nexusml_file = NEXUSML_EXAMPLES_DIR / fca_file.name
        if nexusml_file.exists():
            duplicates.append((fca_file, nexusml_file))

    # Check for duplicates in subdirectories
    for fca_file in fca_files:
        for subdir in NEXUSML_SUBDIRS:
            nexusml_subdir = NEXUSML_EXAMPLES_DIR / subdir
            if not nexusml_subdir.exists():
                continue

            nexusml_file = nexusml_subdir / fca_file.name
            if nexusml_file.exists():
                duplicates.append((fca_file, nexusml_file))

    return duplicates


def find_duplicate_outputs():
    """
    Find duplicate output files between nexusml/output, nexusml/outputs, and outputs.

    Returns:
        List of tuples containing (primary_file_path, duplicate_file_path)
    """
    duplicates = []

    # Check if primary output directory exists
    if not PRIMARY_OUTPUT_DIR.exists():
        logger.error(f"Primary output directory {PRIMARY_OUTPUT_DIR} does not exist")
        return duplicates

    # Get all files in the primary output directory (recursively)
    primary_files = list(PRIMARY_OUTPUT_DIR.glob("**/*"))
    primary_files = [f for f in primary_files if f.is_file()]

    # Check for duplicates in secondary output directories
    for secondary_dir in SECONDARY_OUTPUT_DIRS:
        if not secondary_dir.exists():
            logger.info(
                f"Secondary output directory {secondary_dir} does not exist, skipping"
            )
            continue

        # Get all files in the secondary output directory (recursively)
        secondary_files = list(secondary_dir.glob("**/*"))
        secondary_files = [f for f in secondary_files if f.is_file()]

        # Check for duplicates by name
        for primary_file in primary_files:
            for secondary_file in secondary_files:
                if primary_file.name == secondary_file.name:
                    # Check if the files have the same content
                    try:
                        if primary_file.read_bytes() == secondary_file.read_bytes():
                            duplicates.append((primary_file, secondary_file))
                    except Exception as e:
                        logger.warning(
                            f"Error comparing files {primary_file} and {secondary_file}: {e}"
                        )

    return duplicates


def remove_file(file_path, dry_run=False):
    """
    Remove a file.

    Args:
        file_path: Path to the file to remove
        dry_run: If True, only show what would be removed

    Returns:
        True if the file was removed or would be removed, False otherwise
    """
    try:
        if dry_run:
            logger.info(f"Would remove: {file_path}")
            return True
        else:
            logger.info(f"Removing: {file_path}")
            file_path.unlink()
            return True
    except Exception as e:
        logger.error(f"Error removing {file_path}: {e}")
        return False


def process_duplicates(duplicates, args, description="duplicates"):
    """
    Process a list of duplicate files.

    Args:
        duplicates: List of tuples containing (original_file_path, duplicate_file_path)
        args: Command-line arguments
        description: Description of the duplicates for logging

    Returns:
        Number of files removed
    """
    if not duplicates:
        logger.info(f"No {description} found")
        return 0

    logger.info(f"Found {len(duplicates)} {description}:")
    for i, (original_file, duplicate_file) in enumerate(duplicates, 1):
        logger.info(f"{i}. {duplicate_file} (duplicate of {original_file})")

    removed_count = 0

    if args.dry_run:
        logger.info("Dry run - no files will be removed")
        for _, duplicate_file in duplicates:
            if remove_file(duplicate_file, dry_run=True):
                removed_count += 1
        return removed_count

    if args.auto_remove:
        logger.info(f"Automatically removing all {description}")
        for _, duplicate_file in duplicates:
            if remove_file(duplicate_file):
                removed_count += 1
        return removed_count

    # Interactive mode
    logger.info(f"\nSelect which {description} to remove:")
    logger.info("  a - Remove all duplicates")
    logger.info("  n - Remove none (exit)")
    logger.info(
        "  Or enter comma-separated numbers to remove specific duplicates (e.g., '1,3,5')"
    )

    choice = input("Your choice: ").strip().lower()

    if choice == "a":
        for _, duplicate_file in duplicates:
            if remove_file(duplicate_file):
                removed_count += 1
    elif choice == "n":
        logger.info("No files removed")
    else:
        try:
            indices = [int(idx.strip()) for idx in choice.split(",") if idx.strip()]
            for idx in indices:
                if 1 <= idx <= len(duplicates):
                    _, duplicate_file = duplicates[idx - 1]
                    if remove_file(duplicate_file):
                        removed_count += 1
                else:
                    logger.warning(f"Invalid index: {idx}")
        except ValueError:
            logger.error("Invalid input. No files removed.")

    return removed_count


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Remove duplicate files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be removed, don't actually remove files",
    )
    parser.add_argument(
        "--auto-remove",
        action="store_true",
        help="Automatically remove all identified duplicates without prompting",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Only check for duplicate example files",
    )
    parser.add_argument(
        "--outputs-only",
        action="store_true",
        help="Only check for duplicate output files",
    )
    args = parser.parse_args()

    total_removed = 0

    # Process example duplicates
    if not args.outputs_only:
        logger.info("Finding duplicate example scripts...")
        example_duplicates = find_duplicate_examples()
        total_removed += process_duplicates(
            example_duplicates, args, description="duplicate example scripts"
        )

    # Process output duplicates
    if not args.examples_only:
        logger.info("\nFinding duplicate output files...")
        output_duplicates = find_duplicate_outputs()
        total_removed += process_duplicates(
            output_duplicates, args, description="duplicate output files"
        )

    logger.info(f"\nTotal files processed for removal: {total_removed}")


if __name__ == "__main__":
    main()
