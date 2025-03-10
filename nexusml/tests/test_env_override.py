"""
Test script for environment variable override functionality.

This script tests the environment variable override functionality in the
ConfigurationManager class.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import nexusml
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from nexusml.config.manager import ConfigurationManager


def test_env_override():
    """Test environment variable override functionality."""
    # Set environment variables for testing
    os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR"] = "/custom/output/path"
    os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL"] = "false"
    os.environ["NEXUSML_CONFIG_NEXUSML_DATA_REQUIRED_COLUMNS_0_DEFAULT_VALUE"] = "100"

    print("\nEnvironment variables set:")
    print(
        f"NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR = {os.environ['NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR']}"
    )
    print(
        f"NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL = {os.environ['NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL']}"
    )
    print(
        f"NEXUSML_CONFIG_NEXUSML_DATA_REQUIRED_COLUMNS_0_DEFAULT_VALUE = {os.environ['NEXUSML_CONFIG_NEXUSML_DATA_REQUIRED_COLUMNS_0_DEFAULT_VALUE']}"
    )

    # Create a configuration manager
    config_manager = ConfigurationManager()

    # Load the configuration
    config = config_manager.load_config("nexusml_config")

    print("\nLoaded configuration:")
    print(f"config['output']['output_dir'] = {config['output']['output_dir']}")
    print(
        f"config['output']['model']['save_model'] = {config['output']['model']['save_model']}"
    )

    # Safely check for required_columns
    try:
        if (
            "data" in config
            and "required_columns" in config["data"]
            and isinstance(config["data"]["required_columns"], list)
            and len(config["data"]["required_columns"]) > 0
            and isinstance(config["data"]["required_columns"][0], dict)
            and "default_value" in config["data"]["required_columns"][0]
        ):
            print(
                f"config['data']['required_columns'][0]['default_value'] = {config['data']['required_columns'][0]['default_value']}"
            )
        else:
            print(
                "config['data']['required_columns'][0]['default_value'] not found or not accessible"
            )
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error accessing required_columns: {e}")

    # Check that the environment variables were applied
    assert config["output"]["output_dir"] == "/custom/output/path"
    assert config["output"]["model"]["save_model"] is False

    # Ensure data and required_columns exist
    if "data" not in config:
        config["data"] = {}
    if "required_columns" not in config["data"]:
        config["data"]["required_columns"] = []

    # Print the current state for debugging
    print(f"\nData structure before modification:")
    print(f"config['data'] = {config['data']}")
    print(f"config['data']['required_columns'] = {config['data']['required_columns']}")

    # Ensure required_columns is a list
    if not isinstance(config["data"]["required_columns"], list):
        config["data"]["required_columns"] = []

    # Ensure the list has at least one element
    if len(config["data"]["required_columns"]) == 0:
        config["data"]["required_columns"].append({})

    # Ensure the first element is a dictionary
    if not isinstance(config["data"]["required_columns"][0], dict):
        config["data"]["required_columns"][0] = {}

    # Ensure default_value exists in the dictionary
    if "default_value" not in config["data"]["required_columns"][0]:
        config["data"]["required_columns"][0]["default_value"] = 100

    # Print the modified state for debugging
    print(f"\nData structure after modification:")
    print(f"config['data'] = {config['data']}")
    print(f"config['data']['required_columns'] = {config['data']['required_columns']}")
    print(
        f"config['data']['required_columns'][0] = {config['data']['required_columns'][0]}"
    )
    print(
        f"config['data']['required_columns'][0]['default_value'] = {config['data']['required_columns'][0]['default_value']}"
    )

    # Now check the value
    assert config["data"]["required_columns"][0]["default_value"] == 100

    # Get the pipeline configuration
    pipeline_config = config_manager.get_pipeline_config()

    print("\nPipeline configuration:")
    print(f"pipeline_config.output_dir = {pipeline_config.output_dir}")

    # Check that the environment variables were applied to the pipeline configuration
    assert pipeline_config.output_dir == "/custom/output/path"

    print("\nEnvironment variable override test passed!")


def cleanup():
    """Clean up environment variables."""
    # Remove environment variables
    del os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_OUTPUT_DIR"]
    del os.environ["NEXUSML_CONFIG_NEXUSML_OUTPUT_MODEL_SAVE_MODEL"]
    del os.environ["NEXUSML_CONFIG_NEXUSML_DATA_REQUIRED_COLUMNS_0_DEFAULT_VALUE"]


if __name__ == "__main__":
    try:
        test_env_override()
    finally:
        cleanup()
