"""
Classifier Verification Script

This script verifies that all necessary components are in place to run the NexusML examples.
It checks for required packages, data files, and module imports.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Union

import pandas as pd
from pandas import DataFrame


def get_package_version(package_name: str) -> str:
    """Get the version of a package in a type-safe way.

    Args:
        package_name: Name of the package

    Returns:
        Version string or "unknown" if version cannot be determined
    """
    try:
        # Try to get version directly from the module
        module = importlib.import_module(package_name)
        if hasattr(module, "__version__"):
            return str(module.__version__)

        # Fall back to importlib.metadata
        try:
            from importlib.metadata import version as get_version

            return str(get_version(package_name))
        except (ImportError, ModuleNotFoundError):
            # For Python < 3.8
            try:
                import pkg_resources

                return str(pkg_resources.get_distribution(package_name).version)
            except Exception:
                return "unknown"
    except Exception:
        return "unknown"


def read_csv_safe(filepath: Union[str, Path]) -> DataFrame:
    """Type-safe wrapper for pd.read_csv.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the CSV data
    """
    # Use type ignore to suppress Pylance warning about complex type
    return pd.read_csv(filepath)  # type: ignore


def check_package_versions():
    """Check if all required packages are installed and compatible."""
    print("Checking package versions...")
    all_ok = True

    # Check numpy
    try:
        version = get_package_version("numpy")
        print(f"✓ numpy: {version}")
    except Exception:
        print("✗ numpy: Not installed")
        all_ok = False

    # Check pandas
    try:
        version = get_package_version("pandas")
        print(f"✓ pandas: {version}")
    except Exception:
        print("✗ pandas: Not installed")
        all_ok = False

    # Check scikit-learn
    try:
        version = get_package_version("sklearn")
        print(f"✓ scikit-learn: {version}")
    except Exception:
        print("✗ scikit-learn: Not installed")
        all_ok = False

    # Check matplotlib
    try:
        version = get_package_version("matplotlib")
        print(f"✓ matplotlib: {version}")
    except Exception:
        print("✗ matplotlib: Not installed")
        all_ok = False

    # Check seaborn
    try:
        version = get_package_version("seaborn")
        print(f"✓ seaborn: {version}")
    except Exception:
        print("✗ seaborn: Not installed")
        all_ok = False

    # Check imbalanced-learn
    try:
        version = get_package_version("imblearn")
        print(f"✓ imbalanced-learn: {version}")
    except Exception:
        print("✗ imbalanced-learn: Not installed")
        all_ok = False

    return all_ok


def check_data_file():
    """Check if the training data file exists."""
    # Initialize data_path to None
    data_path = None

    # Try to load from settings
    try:
        import yaml

        # Check if we're running in the context of fca_dashboard
        try:
            from fca_dashboard.utils.path_util import get_config_path, resolve_path

            settings_path = get_config_path("settings.yml")
            with open(settings_path, "r") as file:
                settings = yaml.safe_load(file)

            data_path = settings.get("classifier", {}).get("data_paths", {}).get("training_data")
            if not data_path:
                # Fallback to default path in nexusml
                data_path = "nexusml/ingest/data/eq_ids.csv"

            # Resolve the path to ensure it exists
            data_path = str(resolve_path(data_path))
        except ImportError:
            # Not running in fca_dashboard context, use nexusml paths
            # Look for a config file in the nexusml directory
            settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
            if settings_path.exists():
                with open(settings_path, "r") as file:
                    settings = yaml.safe_load(file)
                data_path = settings.get("data_paths", {}).get("training_data")

            if not data_path:
                # Use default path in nexusml
                data_path = str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        # Use absolute path as fallback
        data_path = str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")

    print(f"\nChecking data file: {data_path}")
    if os.path.exists(data_path):
        print(f"✓ Data file exists: {data_path}")
        try:
            df = read_csv_safe(data_path)
            print(f"✓ Data file can be read: {len(df)} rows, {len(df.columns)} columns")
            return True
        except Exception as e:
            print(f"✗ Error reading data file: {e}")
            return False
    else:
        print(f"✗ Data file not found: {data_path}")
        return False


def check_module_imports():
    """Check if all required module imports work correctly."""
    print("\nChecking module imports...")
    all_ok = True

    modules_to_check = [
        ("nexusml.core.model", "train_enhanced_model"),
        ("nexusml.core.model", "predict_with_enhanced_model"),
        ("nexusml.core.data_preprocessing", "load_and_preprocess_data"),
        ("nexusml.core.feature_engineering", "enhance_features"),
    ]

    for module_name, attr_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, attr_name, None)
            if attr is not None:
                print(f"✓ Successfully imported {module_name}.{attr_name}")
            else:
                print(f"✗ Attribute {attr_name} not found in {module_name}")
                all_ok = False
        except ImportError as e:
            print(f"✗ Error importing {module_name}: {e}")
            all_ok = False
        except Exception as e:
            print(f"✗ Unexpected error with {module_name}: {e}")
            all_ok = False

    return all_ok


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("NEXUSML VERIFICATION")
    print("=" * 60)

    packages_ok = check_package_versions()
    data_ok = check_data_file()
    imports_ok = check_module_imports()

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Packages: {'✓ OK' if packages_ok else '✗ Issues found'}")
    print(f"Data file: {'✓ OK' if data_ok else '✗ Issues found'}")
    print(f"Module imports: {'✓ OK' if imports_ok else '✗ Issues found'}")

    if packages_ok and data_ok and imports_ok:
        print("\nAll checks passed! You can run the NexusML example with:")
        print("\n    python -m nexusml.examples.simple_example")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues before running the NexusML example.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
