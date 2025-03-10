#!/usr/bin/env python3
"""
Script to update imports in example files to use the new directory structure.
"""

import os
import re
from pathlib import Path


def update_imports(file_path):
    """Update imports in a file to use the new directory structure."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return False

    # Define import mappings
    import_mappings = {
        r"from nexusml\.core\.pipeline\.": "from nexusml.src.pipeline.",
        r"from nexusml\.core\.feature_engineering\.": "from nexusml.src.features.",
        r"from nexusml\.core\.model_building\.": "from nexusml.src.models.",
        r"from nexusml\.core\.model_training\.": "from nexusml.src.models.training.",
        r"from nexusml\.core\.model_card\.": "from nexusml.src.models.cards.",
        r"from nexusml\.core\.di\.": "from nexusml.src.utils.di.",
        r"from nexusml\.core\.config\.": "from nexusml.src.utils.config.",
        r"from nexusml\.core\.cli\.": "from nexusml.src.utils.cli.",
        r"from nexusml\.core\.reference\.": "from nexusml.src.utils.reference.",
        r"from nexusml\.core\.validation\.": "from nexusml.src.utils.validation.",
        r"import nexusml\.core\.pipeline\.": "import nexusml.src.pipeline.",
        r"import nexusml\.core\.feature_engineering\.": "import nexusml.src.features.",
        r"import nexusml\.core\.model_building\.": "import nexusml.src.models.",
        r"import nexusml\.core\.model_training\.": "import nexusml.src.models.training.",
        r"import nexusml\.core\.model_card\.": "import nexusml.src.models.cards.",
        r"import nexusml\.core\.di\.": "import nexusml.src.utils.di.",
        r"import nexusml\.core\.config\.": "import nexusml.src.utils.config.",
        r"import nexusml\.core\.cli\.": "import nexusml.src.utils.cli.",
        r"import nexusml\.core\.reference\.": "import nexusml.src.utils.reference.",
        r"import nexusml\.core\.validation\.": "import nexusml.src.utils.validation.",
    }

    # Apply import mappings
    updated_content = content
    for pattern, replacement in import_mappings.items():
        updated_content = re.sub(pattern, replacement, updated_content)

    # Write updated content back to file
    if updated_content != content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            print(f"Updated imports in {file_path}")
            return True
        except Exception as e:
            print(f"Error writing to {file_path}: {e}")
            return False
    else:
        print(f"No imports to update in {file_path}")
        return False


def main():
    """Main function to update imports in all example files."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Get all Python files in the examples directory
    examples_dir = project_root / "examples"
    example_files = list(examples_dir.glob("**/*.py"))

    # Update imports in each file
    updated_files = 0
    for file_path in example_files:
        if update_imports(file_path):
            updated_files += 1

    print(f"Updated imports in {updated_files} files")


if __name__ == "__main__":
    main()
