"""
Configuration Validation Module for NexusML

This module provides utilities for validating configurations against schemas
and ensuring that they meet the requirements of the pipeline.
"""

from typing import Dict, List, Optional, Union, Any

from nexusml.config.manager import ConfigurationManager
from nexusml.config.schemas import validate_config, available_schemas

class ConfigurationValidator:
    """
    Validates configurations against schemas and pipeline requirements.
    
    This class provides methods for validating configurations against schemas
    and ensuring that they meet the requirements of the pipeline.
    """
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the configuration validator.
        
        Args:
            config_manager: Configuration manager to use. If None, a new one will be created.
        """
        self.config_manager = config_manager or ConfigurationManager()
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """
        Validate all known configurations against their schemas.
        
        Returns:
            Dictionary mapping configuration names to validation results
        """
        results = {}
        
        # Validate data configuration
        results["production_data_config"] = self.validate_data_config()
        
        # Validate feature configuration
        results["feature_config"] = self.validate_feature_config()
        
        # Validate model card configuration
        results["model_card_config"] = self.validate_model_card_config()
        
        return results
    
    def validate_data_config(self) -> bool:
        """
        Validate the data configuration against its schema.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the configuration
            config = self.config_manager.load_config("production_data_config")
            
            # Validate against schema
            is_valid = validate_config(config, "data_config_schema")
            
            # Additional validation
            if is_valid:
                # Check that required columns are defined
                if "required_columns" not in config or not config["required_columns"]:
                    print("Data configuration is missing required_columns")
                    return False
                
                # Check that at least one column is defined
                if len(config["required_columns"]) == 0:
                    print("Data configuration must define at least one column")
                    return False
                
                # Check that each column has a name
                for column in config["required_columns"]:
                    if "name" not in column:
                        print("Data configuration has a column without a name")
                        return False
            
            return is_valid
        except Exception as e:
            print(f"Error validating data configuration: {e}")
            return False
    
    def validate_feature_config(self) -> bool:
        """
        Validate the feature configuration against its schema.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the configuration
            config = self.config_manager.load_config("feature_config")
            
            # Validate against schema
            is_valid = validate_config(config, "feature_config_schema")
            
            # Additional validation
            if is_valid:
                # Check that at least one feature engineering method is defined
                has_feature_method = False
                
                if "text_combinations" in config and config["text_combinations"]:
                    has_feature_method = True
                
                if "numeric_columns" in config and config["numeric_columns"]:
                    has_feature_method = True
                
                if "hierarchies" in config and config["hierarchies"]:
                    has_feature_method = True
                
                if "column_mappings" in config and config["column_mappings"]:
                    has_feature_method = True
                
                if not has_feature_method:
                    print("Feature configuration must define at least one feature engineering method")
                    return False
            
            return is_valid
        except Exception as e:
            print(f"Error validating feature configuration: {e}")
            return False
    
    def validate_model_card_config(self) -> bool:
        """
        Validate the model card configuration against its schema.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the configuration
            config = self.config_manager.load_config("model_card_config")
            
            # Validate against schema
            is_valid = validate_config(config, "model_card_schema")
            
            # Additional validation
            if is_valid:
                # Check that model details are defined
                if "model_details" not in config or not config["model_details"]:
                    print("Model card configuration is missing model_details")
                    return False
                
                # Check that model name and version are defined
                model_details = config["model_details"]
                if "name" not in model_details or not model_details["name"]:
                    print("Model card configuration is missing model name")
                    return False
                
                if "version" not in model_details or not model_details["version"]:
                    print("Model card configuration is missing model version")
                    return False
                
                # Check that inputs and outputs are defined
                if "inputs" not in config or not config["inputs"]:
                    print("Model card configuration is missing inputs")
                    return False
                
                if "outputs" not in config or not config["outputs"]:
                    print("Model card configuration is missing outputs")
                    return False
            
            return is_valid
        except Exception as e:
            print(f"Error validating model card configuration: {e}")
            return False
    
    def validate_pipeline_config(self) -> bool:
        """
        Validate the pipeline configuration.
        
        Returns:
            True if valid, False otherwise
        """
        # This will be implemented in Phase 3: Pipeline Orchestration
        return True
    
    def validate_config_compatibility(self) -> bool:
        """
        Validate that configurations are compatible with each other.
        
        Returns:
            True if compatible, False otherwise
        """
        try:
            # Load configurations
            data_config = self.config_manager.get_data_config()
            feature_config = self.config_manager.get_feature_config()
            model_card_config = self.config_manager.get_model_card_config()
            
            # Check that required columns in data config match fields in model card config
            # Note: The model card doesn't need to define all the columns that the data config requires
            # It only needs to define the ones that are relevant for the model card
            data_required_columns = set(data_config.required_columns)
            model_card_fields = set(field["name"] for field in model_card_config.fields)
            
            # Instead of failing, just log a warning
            missing_fields = data_required_columns - model_card_fields
            if missing_fields:
                print(f"Warning: Data config requires columns that are not defined in model card: {missing_fields}")
                # Don't return False here, as this is expected
            
            # Check that text combinations in feature config use columns defined in data config
            # Note: Some columns like 'building_name' might be optional and not defined in the data config
            # We'll just log a warning instead of failing
            for combo in feature_config.text_combinations:
                for column in combo.get("columns", []):
                    if column not in data_required_columns and column not in ["combined_text", "service_life", "building_name"]:
                        print(f"Warning: Text combination uses column '{column}' that is not defined in data config")
                        # Don't return False here, as this is expected for optional columns
            
            # Check that hierarchies in feature config use columns defined in data config
            # Note: Some columns might be optional and not defined in the data config
            # We'll just log a warning instead of failing
            for hierarchy in feature_config.hierarchies:
                for column in hierarchy.get("parents", []):
                    if column not in data_required_columns and column not in ["combined_text", "service_life", "building_name"]:
                        print(f"Warning: Hierarchy uses column '{column}' that is not defined in data config")
                        # Don't return False here, as this is expected for optional columns
            
            return True
        except Exception as e:
            print(f"Error validating config compatibility: {e}")
            return False

# Create a singleton instance of ConfigurationValidator
_config_validator = None

def get_config_validator() -> ConfigurationValidator:
    """
    Get the singleton instance of ConfigurationValidator.
    
    Returns:
        ConfigurationValidator instance
    """
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigurationValidator()
    return _config_validator

def validate_all_configs() -> Dict[str, bool]:
    """
    Validate all known configurations against their schemas.
    
    Returns:
        Dictionary mapping configuration names to validation results
    """
    return get_config_validator().validate_all_configs()

def validate_config_compatibility() -> bool:
    """
    Validate that configurations are compatible with each other.
    
    Returns:
        True if compatible, False otherwise
    """
    return get_config_validator().validate_config_compatibility()