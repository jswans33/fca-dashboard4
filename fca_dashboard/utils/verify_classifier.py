"""
Classifier Verification Script

This script verifies that all necessary components are in place to run the classifier example.
It checks for required packages, data files, and module imports.
"""

import os
import sys
import importlib
import pandas as pd
from pathlib import Path


def check_package_versions():
    """Check if all required packages are installed and compatible."""
    print("Checking package versions...")
    all_ok = True
    
    # Check numpy
    try:
        import numpy
        print(f"✓ numpy: {numpy.__version__}")
    except ImportError:
        print("✗ numpy: Not installed")
        all_ok = False
    
    # Check pandas
    try:
        import pandas
        print(f"✓ pandas: {pandas.__version__}")
    except ImportError:
        print("✗ pandas: Not installed")
        all_ok = False
    
    # Check scikit-learn
    try:
        import sklearn
        print(f"✓ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ scikit-learn: Not installed")
        all_ok = False
    
    # Check matplotlib
    try:
        import matplotlib
        print(f"✓ matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib: Not installed")
        all_ok = False
    
    # Check seaborn
    try:
        import seaborn
        print(f"✓ seaborn: {seaborn.__version__}")
    except ImportError:
        print("✗ seaborn: Not installed")
        all_ok = False
    
    # Check imbalanced-learn
    try:
        import imblearn
        print(f"✓ imbalanced-learn: {imblearn.__version__}")
    except ImportError:
        print("✗ imbalanced-learn: Not installed")
        all_ok = False
    
    return all_ok


def check_data_file():
    """Check if the training data file exists."""
    # Try to load from settings
    try:
        import yaml
        from fca_dashboard.utils.path_util import get_config_path, resolve_path
        
        settings_path = get_config_path("settings.yml")
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
            
        data_path = settings.get('classifier', {}).get('data_paths', {}).get('training_data')
        if not data_path:
            # Fallback to default path
            data_path = "fca_dashboard/classifier/ingest/eq_ids.csv"
        
        # Resolve the path to ensure it exists
        data_path = str(resolve_path(data_path))
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        # Use absolute path as fallback
        data_path = str(Path(__file__).resolve().parent.parent / "classifier" / "ingest" / "eq_ids.csv")
    
    print(f"\nChecking data file: {data_path}")
    if os.path.exists(data_path):
        print(f"✓ Data file exists: {data_path}")
        try:
            df = pd.read_csv(data_path)
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
        ('fca_dashboard.classifier', 'train_enhanced_model'),
        ('fca_dashboard.classifier', 'predict_with_enhanced_model'),
        ('fca_dashboard.utils.path_util', 'get_config_path'),
        ('fca_dashboard.utils.path_util', 'resolve_path'),
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
    print("CLASSIFIER EXAMPLE VERIFICATION")
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
        print("\nAll checks passed! You can run the classifier example with:")
        print("\n    python -m fca_dashboard.examples.classifier_example")
        return 0
    else:
        print("\nSome checks failed. Please fix the issues before running the classifier example.")
        return 1


if __name__ == "__main__":
    sys.exit(main())