#!/usr/bin/env python
"""
Output Directory Migration Script

This script consolidates multiple output directories into a single standardized
output directory structure. It moves files from:
- nexusml/nexusml/output/ -> nexusml/output/
- nexusml/output/ -> nexusml/output/

Usage:
    python -m nexusml.scripts.migrate_outputs
"""

import logging
import os
import shutil
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
logger = logging.getLogger("migrate_outputs")

# Define source and target directories
SOURCE_DIRS = [
    Path("nexusml/outputs"),  # nexusml/nexusml/output/ -> nexusml/output/
    Path("outputs"),  # nexusml/output/ -> nexusml/output/
]
TARGET_DIR = Path("nexusml/output")

# Define subdirectories to create in the target directory
SUBDIRS = [
    "models",
    "results",
    "evaluation",
    "model_cards",
    "temp",
    "logs",
]


def create_target_structure():
    """Create the target directory structure."""
    logger.info(f"Creating target directory structure in {TARGET_DIR}")

    # Create the main output directory
    TARGET_DIR.mkdir(exist_ok=True)

    # Create subdirectories
    for subdir in SUBDIRS:
        (TARGET_DIR / subdir).mkdir(exist_ok=True)
        logger.info(f"Created subdirectory: {TARGET_DIR / subdir}")


def migrate_files(source_dir, target_dir):
    """
    Migrate files from source directory to target directory.

    Args:
        source_dir: Source directory path
        target_dir: Target directory path
    """
    if not source_dir.exists():
        logger.info(f"Source directory {source_dir} does not exist, skipping")
        return

    logger.info(f"Migrating files from {source_dir} to {target_dir}")

    # Get all files and directories in the source directory
    for item in source_dir.glob("*"):
        # Determine target path
        if item.is_dir():
            # For directories, try to map to a corresponding subdirectory
            # or create a new one with the same name
            dir_name = item.name
            if dir_name in SUBDIRS:
                target_path = target_dir / dir_name
            else:
                target_path = target_dir / dir_name
                target_path.mkdir(exist_ok=True)
                logger.info(f"Created new subdirectory: {target_path}")

            # Recursively migrate files from this subdirectory
            migrate_files(item, target_path)
        else:
            # For files, copy directly to the target directory
            target_path = target_dir / item.name

            # Check if the file already exists in the target
            if target_path.exists():
                logger.warning(f"File {target_path} already exists, skipping")
                continue

            # Copy the file
            shutil.copy2(item, target_path)
            logger.info(f"Copied file: {item} -> {target_path}")


def main():
    """Main function to run the migration script."""
    logger.info("Starting output directory migration")

    # Create the target directory structure
    create_target_structure()

    # Migrate files from each source directory
    for source_dir in SOURCE_DIRS:
        migrate_files(source_dir, TARGET_DIR)

    logger.info("Output directory migration completed")
    logger.info(f"All outputs have been consolidated in {TARGET_DIR}")
    logger.info("Note: The original directories have not been deleted.")
    logger.info("After verifying the migration, you can manually delete them.")


if __name__ == "__main__":
    main()
