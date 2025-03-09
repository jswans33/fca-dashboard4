"""
Configuration Manager Module for NexusML

This module provides a centralized approach to configuration management,
implementing the ConfigurationManager class that loads and manages all configuration
files with type-safe access and validation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast
import json
import os
import yaml

from nexusml.config import get_project_root
from nexusml.config.sections import ConfigSection

T = TypeVar('T')

class DataConfig(ConfigSection):
    """Configuration for data handling."""
    
    @property
    def required_columns(self) -> list:
        """Get the required columns."""
        return [col['name'] for col in self.data.get('required_columns', [])]
    
    @property
    def source_columns(self) -> list:
        """Get only the source columns (not derived during feature engineering)."""
        return [
            col['name'] for col in self.data.get('required_columns', [])
            if not col['name'].startswith(('Equipment_', 'Uniformat_', 'System_', 'combined_', 'service_life'))
        ]
    
    @property
    def target_columns(self) -> list:
        """Get only the target columns (derived during feature engineering)."""
        return [
            col['name'] for col in self.data.get('required_columns', [])
            if col['name'].startswith(('Equipment_', 'Uniformat_', 'System_', 'combined_', 'service_life'))
        ]
    
    @property
    def critical_columns(self) -> list:
        """Get the critical columns that must not have missing values."""
        return ["equipment_tag", "category_name", "mcaa_system_category"]
    
    def get_column_default(self, column_name: str) -> Any:
        """Get the default value for a column."""
        for col in self.data.get('required_columns', []):
            if col['name'] == column_name:
                return col.get('default_value')
        return None
    
    def get_column_data_type(self, column_name: str) -> str:
        """Get the data type for a column."""
        for col in self.data.get('required_columns', []):
            if col['name'] == column_name:
                return col.get('data_type', 'str')
        return 'str'

class FeatureConfig(ConfigSection):
    """Configuration for feature engineering."""
    
    @property
    def text_combinations(self) -> list:
        """Get text combination configurations."""
        return self.data.get('text_combinations', [])
    
    @property
    def numeric_columns(self) -> list:
        """Get numeric column configurations."""
        return self.data.get('numeric_columns', [])
    
    @property
    def hierarchies(self) -> list:
        """Get hierarchy configurations."""
        return self.data.get('hierarchies', [])
    
    @property
    def column_mappings(self) -> list:
        """Get column mapping configurations."""
        return self.data.get('column_mappings', [])
    
    @property
    def classification_systems(self) -> list:
        """Get classification system configurations."""
        return self.data.get('classification_systems', [])
    
    @property
    def eav_integration_enabled(self) -> bool:
        """Check if EAV integration is enabled."""
        eav_config = self.data.get('eav_integration', {})
        return eav_config.get('enabled', False)

class ModelConfig(ConfigSection):
    """Configuration for model building and training."""
    
    @property
    def model_type(self) -> str:
        """Get the model type."""
        return self.data.get('model_type', 'random_forest')
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get the model hyperparameters."""
        return self.data.get('hyperparameters', {})
    
    @property
    def evaluation_metrics(self) -> list:
        """Get the evaluation metrics."""
        return self.data.get('evaluation_metrics', ['accuracy', 'f1'])
    
    @property
    def cross_validation(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return self.data.get('cross_validation', {'enabled': False, 'folds': 5})

class PipelineConfig(ConfigSection):
    """Configuration for pipeline orchestration."""
    
    @property
    def stages(self) -> list:
        """Get the pipeline stages."""
        return self.data.get('stages', [])
    
    @property
    def components(self) -> Dict[str, str]:
        """Get the component implementations to use."""
        return self.data.get('components', {})
    
    @property
    def output_dir(self) -> str:
        """Get the output directory."""
        return self.data.get('output_dir', 'outputs')
    
    @property
    def visualizations_enabled(self) -> bool:
        """Check if visualizations are enabled."""
        return self.data.get('visualizations', {}).get('enabled', False)

class ConfigurationManager:
    """Manager for all configuration files."""
    
    def __init__(self):
        self.root = get_project_root()
        self.config_dir = self.root / "config"
        self.configs = {}
        
        # Environment-specific configuration
        self.environment = os.environ.get('NEXUSML_ENV', 'production')
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            Configuration as a dictionary
        
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
        """
        # Check if config is already loaded
        if name in self.configs:
            return self.configs[name]
        
        # Try environment-specific config first
        if self.environment != 'production':
            env_path = self.config_dir / f"{name}.{self.environment}.yml"
            if env_path.exists():
                with open(env_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                self.configs[name] = config
                return config
        
        # Try standard config
        path = self.config_dir / f"{name}.yml"
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
        
        self.configs[name] = config
        return config
    
    def get_config_section(self, name: str, section_class: Type[T]) -> T:
        """
        Get a typed configuration section.
        
        Args:
            name: Name of the configuration file (without extension)
            section_class: Class to instantiate with the configuration
            
        Returns:
            Instance of section_class initialized with the configuration
        """
        config = self.load_config(name)
        return section_class(config)
    
    def get_data_config(self, name: str = "production_data_config") -> DataConfig:
        """
        Get the data configuration.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            DataConfig instance
        """
        return self.get_config_section(name, DataConfig)
    
    def get_feature_config(self, name: str = "feature_config") -> FeatureConfig:
        """
        Get the feature configuration.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            FeatureConfig instance
        """
        return self.get_config_section(name, FeatureConfig)
    
    def get_model_config(self, name: str = "classification_config") -> ModelConfig:
        """
        Get the model configuration.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            ModelConfig instance
        """
        return self.get_config_section(name, ModelConfig)
    
    def get_pipeline_config(self, name: str = "nexusml_config") -> PipelineConfig:
        """
        Get the pipeline configuration.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            PipelineConfig instance
        """
        return self.get_config_section(name, PipelineConfig)
    
    def get_model_card_config(self, name: str = "model_card_config") -> Any:
        """
        Get the model card configuration.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            ModelCardConfig instance
        """
        # Import here to avoid circular import
        from nexusml.config.model_card import ModelCardConfig
        return self.get_config_section(name, ModelCardConfig)
    
    def merge_configs(self, base_name: str, override_name: str) -> Dict[str, Any]:
        """
        Merge two configurations, with the override taking precedence.
        
        Args:
            base_name: Name of the base configuration file
            override_name: Name of the configuration file with overrides
            
        Returns:
            Merged configuration dictionary
        """
        base_config = self.load_config(base_name)
        try:
            override_config = self.load_config(override_name)
        except FileNotFoundError:
            # If override doesn't exist, just return the base config
            return base_config
        
        # Deep merge the dictionaries
        return self._deep_merge(base_config, override_config)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary with overrides
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override or add the value
                result[key] = value
                
        return result
    
    def validate_config(self, name: str, schema_name: str = None) -> bool:
        """
        Validate a configuration against a JSON Schema.
        
        Args:
            name: Name of the configuration file
            schema_name: Name of the schema file (defaults to {name}_schema)
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the configuration
            config = self.load_config(name)
            
            # Determine schema name if not provided
            if schema_name is None:
                # Map configuration names to schema names
                schema_mapping = {
                    "production_data_config": "data_config_schema",
                    "data_config": "data_config_schema",
                    "feature_config": "feature_config_schema",
                    "classification_config": "model_config_schema",
                    "nexusml_config": "pipeline_config_schema",
                    "model_card_config": "model_card_schema",
                }
                schema_name = schema_mapping.get(name, f"{name}_schema")
            
            # Use the schema validation functionality from the schemas package
            from nexusml.config.schemas import validate_config as validate_with_schema
            return validate_with_schema(config, schema_name)
            
        except Exception as e:
            # Validation failed
            print(f"Configuration validation failed: {e}")
            return False