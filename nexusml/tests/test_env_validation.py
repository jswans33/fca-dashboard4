"""
Test script for environment variable validation functionality.

This script tests the environment variable validation functionality in the
ConfigurationValidator class.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import nexusml
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nexusml.config.validation import validate_environment_variables


def test_env_validation():
    """Test environment variable validation functionality."""
    # Set valid environment variables for testing
    os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR"] = "/custom/output/path"
    os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL"] = "false"

    # Set invalid environment variables for testing
    os.environ["NEXUSML_CONFIG_NONEXISTENT_CONFIG_KEY"] = "value"  # Non-existent config
    os.environ["NEXUSML_CONFIG_NEXUSML_NONEXISTENT_KEY"] = "value"  # Non-existent key
    os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL"] = (
        "not-a-boolean"  # Invalid type
    )

    # Validate environment variables
    results = validate_environment_variables()

    # Print results
    print("\nEnvironment Variable Validation Results:")
    print("----------------------------------------")
    for env_var, is_valid in results.items():
        print(f"{env_var}: {'Valid' if is_valid else 'Invalid'}")

    # Check that the validation results are correct
    assert results["NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR"] is True
    assert results["NEXUSML_CONFIG_NONEXISTENT_CONFIG_KEY"] is False
    assert (
        results["NEXUSML_CONFIG_NEXUSML_NONEXISTENT_KEY"] is True
    )  # This is now valid because we create the key

    print("\nEnvironment variable validation test completed!")


def cleanup():
    """Clean up environment variables."""
    # Remove environment variables
    del os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR"]
    del os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL"]
    del os.environ["NEXUSML_CONFIG_NONEXISTENT_CONFIG_KEY"]
    del os.environ["NEXUSML_CONFIG_NEXUSML_NONEXISTENT_KEY"]


if __name__ == "__main__":
    try:
        test_env_validation()
    finally:
        cleanup()
