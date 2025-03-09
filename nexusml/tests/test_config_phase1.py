#!/usr/bin/env python
"""
Test Script for Phase 1: Configuration Centralization

This script verifies that the configuration management, interfaces, and path resolution
are all functioning as expected after the Phase 1 refactoring.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import the refactored modules
from nexusml.config import (
    get_project_root,
    get_configuration_manager,
    get_resolved_data_path,
    get_resolved_config_path,
    get_output_path,
    get_reference_path,
    ModelCardConfig,
    validate_all_configs,
    validate_config_compatibility,
)
from nexusml.config.manager import ConfigurationManager
from nexusml.config.paths import PathResolver, get_path_resolver
from nexusml.config.schemas import validate_config, available_schemas

def test_configuration_manager():
    """Test the ConfigurationManager functionality."""
    print("\n=== Testing ConfigurationManager ===")
    
    # Get the singleton instance
    config_manager = get_configuration_manager()
    print(f"ConfigurationManager instance: {config_manager}")
    
    # Load data configuration
    try:
        data_config = config_manager.get_data_config()
        print(f"Loaded data configuration successfully")
        print(f"Required columns: {data_config.required_columns[:5]}...")
        print(f"Source columns: {data_config.source_columns[:5]}...")
        print(f"Critical columns: {data_config.critical_columns}")
    except Exception as e:
        print(f"Error loading data configuration: {e}")
    
    # Load feature configuration
    try:
        feature_config = config_manager.get_feature_config()
        print(f"Loaded feature configuration successfully")
        print(f"Text combinations: {feature_config.text_combinations[:2]}...")
        print(f"Numeric columns: {feature_config.numeric_columns[:2]}...")
        print(f"EAV integration enabled: {feature_config.eav_integration_enabled}")
    except Exception as e:
        print(f"Error loading feature configuration: {e}")
    
    # Test configuration validation
    try:
        is_valid = config_manager.validate_config("production_data_config")
        print(f"Data configuration validation: {'Passed' if is_valid else 'Failed'}")
        
        is_valid = config_manager.validate_config("feature_config")
        print(f"Feature configuration validation: {'Passed' if is_valid else 'Failed'}")
    except Exception as e:
        print(f"Error validating configuration: {e}")
    
    # Test configuration merging
    try:
        merged_config = config_manager.merge_configs("production_data_config", "feature_config")
        print(f"Merged configuration successfully")
        print(f"Merged config keys: {list(merged_config.keys())[:5]}...")
    except Exception as e:
        print(f"Error merging configurations: {e}")

def test_path_resolver():
    """Test the PathResolver functionality."""
    print("\n=== Testing PathResolver ===")
    
    # Get the singleton instance
    path_resolver = get_path_resolver()
    print(f"PathResolver instance: {path_resolver}")
    
    # Test path resolution
    try:
        # Resolve a relative path
        relative_path = "config/production_data_config.yml"
        resolved_path = path_resolver.resolve_path(relative_path)
        print(f"Resolved relative path: {resolved_path}")
        print(f"Path exists: {resolved_path.exists()}")
        
        # Resolve an absolute path
        absolute_path = str(get_project_root() / "config" / "feature_config.yml")
        resolved_path = path_resolver.resolve_path(absolute_path)
        print(f"Resolved absolute path: {resolved_path}")
        print(f"Path exists: {resolved_path.exists()}")
        
        # Get data path
        data_path = path_resolver.get_data_path("training_data")
        print(f"Data path: {data_path}")
        
        # Get config path
        config_path = path_resolver.get_config_path("production_data_config")
        print(f"Config path: {config_path}")
        print(f"Path exists: {config_path.exists()}")
        
        # Get output path
        output_path = path_resolver.get_output_path("models")
        print(f"Output path: {output_path}")
        
        # Get reference path
        reference_path = path_resolver.get_reference_path("omniclass")
        print(f"Reference path: {reference_path}")
    except Exception as e:
        print(f"Error resolving paths: {e}")

def test_convenience_functions():
    """Test the convenience functions for path resolution."""
    print("\n=== Testing Convenience Functions ===")
    
    try:
        # Get data path
        data_path = get_resolved_data_path("training_data")
        print(f"Data path: {data_path}")
        
        # Get config path
        config_path = get_resolved_config_path("production_data_config")
        print(f"Config path: {config_path}")
        
        # Get output path
        output_path = get_output_path("models")
        print(f"Output path: {output_path}")
        
        # Get reference path
        reference_path = get_reference_path("omniclass")
        print(f"Reference path: {reference_path}")
    except Exception as e:
        print(f"Error using convenience functions: {e}")

def test_schema_validation():
    """Test the schema validation functionality."""
    print("\n=== Testing Schema Validation ===")
    
    # List available schemas
    print(f"Available schemas: {available_schemas}")
    
    # Load and validate data configuration
    try:
        config_manager = get_configuration_manager()
        data_config = config_manager.load_config("production_data_config")
        
        is_valid = validate_config(data_config, "data_config_schema")
        print(f"Data configuration validation: {'Passed' if is_valid else 'Failed'}")
    except Exception as e:
        print(f"Error validating data configuration: {e}")
    
    # Load and validate feature configuration
    try:
        feature_config = config_manager.load_config("feature_config")
        
        is_valid = validate_config(feature_config, "feature_config_schema")
        print(f"Feature configuration validation: {'Passed' if is_valid else 'Failed'}")
    except Exception as e:
        print(f"Error validating feature configuration: {e}")

def test_model_card_config():
    """Test the ModelCardConfig functionality."""
    print("\n=== Testing ModelCardConfig ===")
    
    # Get the configuration manager
    config_manager = get_configuration_manager()
    
    # Load model card configuration
    try:
        model_card_config = config_manager.get_model_card_config()
        print(f"Loaded model card configuration successfully")
        print(f"Model name: {model_card_config.model_name}")
        print(f"Model version: {model_card_config.model_version}")
        
        # Test inputs and outputs
        print(f"Inputs: {[input['name'] for input in model_card_config.inputs]}")
        print(f"Outputs: {[output['name'] for output in model_card_config.outputs]}")
        
        # Test field access
        print(f"Required fields: {[field['name'] for field in model_card_config.required_fields][:5]}...")
        print(f"Target fields: {[field['name'] for field in model_card_config.target_fields]}")
        
        # Test field lookup
        equipment_tag_field = model_card_config.get_field_by_name("equipment_tag")
        if equipment_tag_field:
            print(f"Equipment tag field: {equipment_tag_field['name']}")
            print(f"Equipment tag description: {model_card_config.get_field_description('equipment_tag')}")
            print(f"Equipment tag example: {model_card_config.get_field_example('equipment_tag')}")
            print(f"Equipment tag data type: {model_card_config.get_field_data_type('equipment_tag')}")
            print(f"Equipment tag required: {model_card_config.is_field_required('equipment_tag')}")
            print(f"Equipment tag target: {model_card_config.is_field_target('equipment_tag')}")
        
        # Test MCAA ID mapping
        print(f"MCAA ID for 'HVAC Equipment': {model_card_config.get_mcaaid_for_system_category('HVAC Equipment')}")
        
        # Test standard categories
        print(f"Standard categories: {model_card_config.standard_categories[:5]}...")
        
        # Test technical specifications
        print(f"Hyperparameters: {model_card_config.hyperparameters}")
        print(f"Text combinations: {model_card_config.text_combinations}")
        
        # Test reference data paths
        print(f"Omniclass reference data path: {model_card_config.get_reference_data_path('omniclass')}")
        
    except Exception as e:
        print(f"Error loading model card configuration: {e}")

def test_config_validation():
    """Test the configuration validation functionality."""
    print("\n=== Testing Configuration Validation ===")
    
    # Validate all configurations
    try:
        results = validate_all_configs()
        print("Validation results:")
        for config_name, is_valid in results.items():
            print(f"  {config_name}: {'Passed' if is_valid else 'Failed'}")
        
        # Validate configuration compatibility
        is_compatible = validate_config_compatibility()
        print(f"Configuration compatibility: {'Passed' if is_compatible else 'Failed'}")
        
    except Exception as e:
        print(f"Error validating configurations: {e}")

def main():
    """Main function to run all tests."""
    print("=== Phase 1 Verification Tests ===")
    print(f"Project root: {get_project_root()}")
    
    # Run all tests
    test_configuration_manager()
    test_path_resolver()
    test_convenience_functions()
    test_schema_validation()
    test_model_card_config()
    test_config_validation()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()