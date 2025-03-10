#!/usr/bin/env python
"""
Update Output Paths

This script updates references to old output directories (nexusml/output/ and nexusml/nexusml/output/)
to use the standardized nexusml/output/ path in Python files.

Usage:
    python -m nexusml.scripts.update_output_paths [--dry-run] [--auto-update]

Options:
    --dry-run      Only show what would be updated, don't actually modify files
    --auto-update  Automatically update all files without prompting
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
logger = logging.getLogger("update_output_paths")

# Define old and new paths
PATH_MAPPINGS = {
    r"nexusml/output/": "nexusml/output/",
    r"nexusml/nexusml/output/": "nexusml/output/",
    r'"nexusml/output/': '"nexusml/output/',
    r"'nexusml/output/": "'nexusml/output/",
    r'"nexusml/nexusml/output/': '"nexusml/output/',
    r"'nexusml/nexusml/output/": "'nexusml/output/",
}

# Files to exclude (e.g., migration scripts that need to reference old paths)
EXCLUDE_FILES = [
    "nexusml/scripts/migrate_outputs.py",
    "nexusml/scripts/remove_duplicates.py",
    "nexusml/scripts/update_output_paths.py",
]


def find_files_with_old_paths() -> Dict[Path, List[Tuple[str, int, str]]]:
    """
    Find Python files that contain references to old output directories.

    Returns:
        Dictionary mapping file paths to lists of (line_number, line_content) tuples
    """
    files_with_old_paths = {}

    # Get all Python files in the nexusml directory
    python_files = list(project_root.glob("nexusml/**/*.py"))

    # Add Python files in the root directory
    python_files.extend(list(project_root.glob("*.py")))

    # Check each file for old path references
    for file_path in python_files:
        # Skip excluded files
        if any(str(file_path).endswith(exclude) for exclude in EXCLUDE_FILES):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                matches = []
                for i, line in enumerate(lines):
                    for old_path in PATH_MAPPINGS.keys():
                        if old_path in line:
                            matches.append((old_path, i + 1, line.rstrip()))

                if matches:
                    files_with_old_paths[file_path] = matches
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")

    return files_with_old_paths


def update_file(file_path: Path, dry_run: bool = False) -> bool:
    """
    Update references to old output directories in a file.

    Args:
        file_path: Path to the file to update
        dry_run: If True, only show what would be updated

    Returns:
        True if the file was updated or would be updated, False otherwise
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        updated_content = content
        for old_path, new_path in PATH_MAPPINGS.items():
            updated_content = updated_content.replace(old_path, new_path)

        if updated_content != content:
            if dry_run:
                logger.info(f"Would update: {file_path}")
                return True
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info(f"Updated: {file_path}")
                return True
        else:
            logger.info(f"No changes needed for: {file_path}")
            return False
    except Exception as e:
        logger.error(f"Error updating file {file_path}: {e}")
        return False


def process_files(files_with_old_paths: Dict[Path, List[Tuple[str, int, str]]], args):
    """
    Process files with old path references.

    Args:
        files_with_old_paths: Dictionary mapping file paths to lists of (old_path, line_number, line_content) tuples
        args: Command-line arguments

    Returns:
        Number of files updated
    """
    if not files_with_old_paths:
        logger.info("No files with old output paths found")
        return 0

    logger.info(f"Found {len(files_with_old_paths)} files with old output paths:")
    for file_path, matches in files_with_old_paths.items():
        logger.info(f"\n{file_path}:")
        for old_path, line_number, line_content in matches:
            logger.info(f"  Line {line_number}: {line_content}")
            logger.info(f"    Old path: {old_path}")
            logger.info(f"    New path: {PATH_MAPPINGS[old_path]}")

    updated_count = 0

    if args.dry_run:
        logger.info("\nDry run - no files will be updated")
        for file_path in files_with_old_paths.keys():
            if update_file(file_path, dry_run=True):
                updated_count += 1
        return updated_count

    if args.auto_update:
        logger.info("\nAutomatically updating all files")
        for file_path in files_with_old_paths.keys():
            if update_file(file_path):
                updated_count += 1
        return updated_count

    # Interactive mode
    logger.info("\nSelect which files to update:")
    logger.info("  a - Update all files")
    logger.info("  n - Update none (exit)")
    logger.info(
        "  Or enter comma-separated numbers to update specific files (e.g., '1,3,5')"
    )

    # Create a numbered list of files
    file_list = list(files_with_old_paths.keys())
    for i, file_path in enumerate(file_list, 1):
        logger.info(f"  {i}. {file_path}")

    choice = input("Your choice: ").strip().lower()

    if choice == "a":
        for file_path in file_list:
            if update_file(file_path):
                updated_count += 1
    elif choice == "n":
        logger.info("No files updated")
    else:
        try:
            indices = [int(idx.strip()) for idx in choice.split(",") if idx.strip()]
            for idx in indices:
                if 1 <= idx <= len(file_list):
                    file_path = file_list[idx - 1]
                    if update_file(file_path):
                        updated_count += 1
                else:
                    logger.warning(f"Invalid index: {idx}")
        except ValueError:
            logger.error("Invalid input. No files updated.")

    return updated_count


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(
        description="Update references to old output directories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be updated, don't actually modify files",
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Automatically update all files without prompting",
    )
    args = parser.parse_args()

    logger.info("Finding files with old output paths...")
    files_with_old_paths = find_files_with_old_paths()

    updated_count = process_files(files_with_old_paths, args)

    logger.info(f"\nTotal files processed for update: {updated_count}")


if __name__ == "__main__":
    main()
