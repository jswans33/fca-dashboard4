#!/usr/bin/env python3
"""
Verification script to test that the new directory structure works correctly.
"""

import importlib
import os
import sys
import traceback
from pathlib import Path


def check_directory(directory):
    """Check if a directory exists."""
    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"✅ Directory exists: {directory}")
        return True
    else:
        print(f"❌ Directory does not exist: {directory}")
        return False


def check_import(module_name):
    """Check if a module can be imported."""
    try:
        module = importlib.import_module(module_name)
        print(f"✅ Module imported successfully: {module_name}")
        return True, None
    except ImportError as e:
        print(f"❌ Failed to import module: {module_name}")
        print(f"   Error: {e}")
        return False, (module_name, str(e), traceback.format_exc())


def main():
    """Main function to verify the directory structure."""
    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Check directories
    directories = [
        "config",
        "data",
        "data/raw",
        "data/processed",
        "data/reference",
        "examples",
        "models",
        "models/baseline",
        "models/production",
        "notebooks",
        "outputs",
        "outputs/predictions",
        "outputs/evaluation",
        "outputs/model_cards",
        "scripts",
        "src",
        "src/data",
        "src/features",
        "src/models",
        "src/pipeline",
        "src/utils",
        "tests",
    ]

    print("Checking directories...")
    missing_dirs = []
    for directory in directories:
        dir_path = project_root / directory
        if not check_directory(dir_path):
            missing_dirs.append(directory)

    # Check imports
    print("\nChecking imports...")
    modules = [
        "nexusml.src.data",
        "nexusml.src.features",
        "nexusml.src.models",
        "nexusml.src.pipeline",
        "nexusml.src.utils",
    ]

    import_errors = []
    for module in modules:
        success, error = check_import(module)
        if not success:
            import_errors.append(error)

    # Print summary
    print("\nVerification Summary:")
    if not missing_dirs:
        print("✅ All directories exist")
    else:
        print("❌ Some directories are missing:")
        for directory in missing_dirs:
            print(f"   - {directory}")

    if not import_errors:
        print("✅ All modules can be imported")
    else:
        print("❌ Some modules cannot be imported:")
        for module_name, error_msg, _ in import_errors:
            print(f"   - {module_name}: {error_msg}")

    # Print detailed error information
    if import_errors:
        print("\nDetailed Import Errors:")
        for module_name, error_msg, traceback_str in import_errors:
            print(f"\n=== Error importing {module_name} ===")
            print(f"Error message: {error_msg}")
            print("Traceback:")
            print(traceback_str)

    if not missing_dirs and not import_errors:
        print("\n✅ Directory structure verification passed!")
        return 0
    else:
        print("\n❌ Directory structure verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
