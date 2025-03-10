#!/usr/bin/env python
"""
Example Directory Migration Script

This script consolidates multiple example directories into a single standardized
example directory structure. It moves files from:
- examples/ -> nexusml/examples/
- fca_dashboard/examples/ -> nexusml/examples/

It also categorizes examples by functionality and adds README.md files to each
example directory.

Usage:
    python -m nexusml.scripts.migrate_examples
"""

import logging
import os
import re
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
logger = logging.getLogger("migrate_examples")

# Define source and target directories
SOURCE_DIRS = [
    Path("examples"),  # examples/ -> nexusml/examples/
    # Note: fca_dashboard examples should remain in place as they are legacy examples
]
TARGET_DIR = Path("nexusml/examples")

# Define categories for examples
CATEGORIES = {
    "data_loading": [
        "data",
        "load",
        "extract",
        "excel",
        "csv",
        "json",
        "xml",
        "database",
        "sql",
        "staging",
    ],
    "feature_engineering": [
        "feature",
        "engineering",
        "transform",
        "text",
        "numeric",
        "categorical",
        "normalize",
    ],
    "model_building": [
        "model",
        "build",
        "classifier",
        "regression",
        "random_forest",
        "svm",
        "neural",
    ],
    "pipeline": [
        "pipeline",
        "orchestrator",
        "factory",
        "component",
        "stage",
        "resolution",
    ],
    "evaluation": [
        "evaluation",
        "metrics",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion",
    ],
    "prediction": ["prediction", "predict", "inference", "classify"],
    "visualization": [
        "visualization",
        "visualize",
        "plot",
        "chart",
        "graph",
        "dashboard",
    ],
    "utilities": ["util", "utility", "helper", "tool", "script"],
    "legacy": ["legacy", "deprecated", "old", "archive"],
}


def create_target_structure():
    """Create the target directory structure."""
    logger.info(f"Creating target directory structure in {TARGET_DIR}")

    # Create the main examples directory
    TARGET_DIR.mkdir(exist_ok=True)

    # Create category subdirectories
    for category in CATEGORIES.keys():
        (TARGET_DIR / category).mkdir(exist_ok=True)
        logger.info(f"Created category subdirectory: {TARGET_DIR / category}")

        # Create README.md for each category
        readme_path = TARGET_DIR / category / "README.md"
        if not readme_path.exists():
            with open(readme_path, "w") as f:
                f.write(f"# {category.replace('_', ' ').title()} Examples\n\n")
                f.write(
                    f"This directory contains examples related to {category.replace('_', ' ')}.\n\n"
                )
                f.write("## Examples\n\n")
            logger.info(f"Created README.md for category: {category}")


def categorize_example(file_path):
    """
    Categorize an example file based on its name and content.

    Args:
        file_path: Path to the example file

    Returns:
        Category name
    """
    # Default category
    default_category = "utilities"

    # Get the file name without extension
    file_name = file_path.stem.lower()

    # Check if the file name matches any category keywords
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in file_name:
                return category

    # If no match in the file name, check the file content
    if file_path.suffix.lower() == ".py":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

                # Check if the content matches any category keywords
                for category, keywords in CATEGORIES.items():
                    for keyword in keywords:
                        if keyword.lower() in content:
                            return category
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")

    # If no match, return the default category
    return default_category


def migrate_example(source_path, target_dir):
    """
    Migrate an example file to the target directory.

    Args:
        source_path: Path to the source file
        target_dir: Path to the target directory
    """
    # Skip __init__.py files
    if source_path.name == "__init__.py":
        return

    # Skip directories
    if source_path.is_dir():
        return

    # Skip non-Python files
    if source_path.suffix.lower() not in [".py", ".ipynb", ".md"]:
        return

    # Categorize the example
    category = categorize_example(source_path)
    category_dir = target_dir / category

    # Create the target path
    target_path = category_dir / source_path.name

    # Check if the file already exists in the target
    if target_path.exists():
        logger.warning(f"File {target_path} already exists, skipping")
        return

    # Copy the file
    shutil.copy2(source_path, target_path)
    logger.info(f"Copied example: {source_path} -> {target_path}")

    # Update the README.md for the category
    readme_path = category_dir / "README.md"
    with open(readme_path, "a") as f:
        f.write(f"- [{source_path.name}](./{source_path.name}): ")

        # Try to extract a description from the file
        description = "Example file"
        if source_path.suffix.lower() == ".py":
            try:
                with open(source_path, "r", encoding="utf-8") as src_file:
                    content = src_file.read()

                    # Try to find a docstring
                    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
                    if docstring_match:
                        docstring = docstring_match.group(1).strip()
                        first_line = docstring.split("\n")[0].strip()
                        if first_line:
                            description = first_line
            except Exception as e:
                logger.warning(f"Could not extract description from {source_path}: {e}")

        f.write(f"{description}\n")


def migrate_examples():
    """Migrate examples from source directories to target directory."""
    logger.info("Starting example directory migration")

    # Create the target directory structure
    create_target_structure()

    # Migrate examples from each source directory
    for source_dir in SOURCE_DIRS:
        if not source_dir.exists():
            logger.info(f"Source directory {source_dir} does not exist, skipping")
            continue

        logger.info(f"Migrating examples from {source_dir}")

        # Get all files in the source directory
        for item in source_dir.glob("*"):
            if item.is_file():
                migrate_example(item, TARGET_DIR)
            elif item.is_dir() and item.name != "__pycache__":
                # For subdirectories, migrate each file
                for file_path in item.glob("*"):
                    if file_path.is_file():
                        migrate_example(file_path, TARGET_DIR)

    # Create a main README.md for the examples directory
    main_readme_path = TARGET_DIR / "README.md"
    with open(main_readme_path, "w") as f:
        f.write("# NexusML Examples\n\n")
        f.write("This directory contains examples for using the NexusML library.\n\n")
        f.write("## Categories\n\n")

        for category in sorted(CATEGORIES.keys()):
            category_dir = TARGET_DIR / category
            example_count = len(list(category_dir.glob("*.py")))
            if example_count > 0:
                f.write(
                    f"- [{category.replace('_', ' ').title()}](./{category}/): {example_count} examples\n"
                )

    logger.info("Example directory migration completed")
    logger.info(f"All examples have been consolidated in {TARGET_DIR}")
    logger.info("Note: The original directories have not been deleted.")
    logger.info("After verifying the migration, you can manually delete them.")


def main():
    """Main function to run the migration script."""
    migrate_examples()


if __name__ == "__main__":
    main()
