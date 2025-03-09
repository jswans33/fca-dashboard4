"""
YAML Configuration Implementations Module for NexusML

This module provides concrete implementations of configuration interfaces
using YAML files as the underlying storage mechanism.
"""

from typing import Any, Dict, List, Optional, Union
import yaml
from pathlib import Path

from nexusml.config import get_project_root
from nexusml.config.interfaces import (
    ConfigInterface,
    DataConfigInterface,
    FeatureConfigInterface,
    ModelConfigInterface,
    PipelineConfigInterface,
)

class YamlConfigBase(ConfigInterface):
    """Base class for YAML-based configurations."""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize with configuration data.
        
        Args:
            data: Configuration data dictionary
        """
        self.data = data
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        return self.data.get(key, default)
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value
            default: Default value if path is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        
        # Navigate through the nested dictionary
        current = self.data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
                
        return current
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'YamlConfigBase':
        """
        Create a configuration instance from a YAML file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Configuration instance
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            yaml.YAMLError: If the file contains invalid YAML
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        return cls(data)
    
    @classmethod
    def from_config_name(cls, config_name: str) -> 'YamlConfigBase':
        """
        Create a configuration instance from a configuration name.
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Configuration instance
        """
        root = get_project_root()
        config_path = root / "config" / f"{config_name}.yml"
        return cls.from_file(config_path)

class YamlDataConfig(YamlConfigBase, DataConfigInterface):
    """YAML-based implementation of DataConfigInterface."""
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns.
        
        Returns:
            List of required column names
        """
        return [col['name'] for col in self.data.get('required_columns', [])]
    
    def get_source_columns(self) -> List[str]:
        """
        Get the list of source columns (not derived during feature engineering).
        
        Returns:
            List of source column names
        """
        return [
            col['name'] for col in self.data.get('required_columns', [])
            if not col['name'].startswith(('Equipment_', 'Uniformat_', 'System_', 'combined_', 'service_life'))
        ]
    
    def get_target_columns(self) -> List[str]:
        """
        Get the list of target columns (derived during feature engineering).
        
        Returns:
            List of target column names
        """
        return [
            col['name'] for col in self.data.get('required_columns', [])
            if col['name'].startswith(('Equipment_', 'Uniformat_', 'System_', 'combined_', 'service_life'))
        ]
    
    def get_critical_columns(self) -> List[str]:
        """
        Get the list of critical columns that must not have missing values.
        
        Returns:
            List of critical column names
        """
        return ["equipment_tag", "category_name", "mcaa_system_category"]
    
    def get_column_default(self, column_name: str) -> Any:
        """
        Get the default value for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Default value for the column
        """
        for col in self.data.get('required_columns', []):
            if col['name'] == column_name:
                return col.get('default_value')
        return None
    
    def get_column_data_type(self, column_name: str) -> str:
        """
        Get the data type for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Data type of the column
        """
        for col in self.data.get('required_columns', []):
            if col['name'] == column_name:
                return col.get('data_type', 'str')
        return 'str'

class YamlFeatureConfig(YamlConfigBase, FeatureConfigInterface):
    """YAML-based implementation of FeatureConfigInterface."""
    
    def get_text_combinations(self) -> List[Dict[str, Any]]:
        """
        Get text combination configurations.
        
        Returns:
            List of text combination configurations
        """
        return self.data.get('text_combinations', [])
    
    def get_numeric_columns(self) -> List[Dict[str, Any]]:
        """
        Get numeric column configurations.
        
        Returns:
            List of numeric column configurations
        """
        return self.data.get('numeric_columns', [])
    
    def get_hierarchies(self) -> List[Dict[str, Any]]:
        """
        Get hierarchy configurations.
        
        Returns:
            List of hierarchy configurations
        """
        return self.data.get('hierarchies', [])
    
    def get_column_mappings(self) -> List[Dict[str, str]]:
        """
        Get column mapping configurations.
        
        Returns:
            List of column mapping configurations
        """
        return self.data.get('column_mappings', [])
    
    def get_classification_systems(self) -> List[Dict[str, Any]]:
        """
        Get classification system configurations.
        
        Returns:
            List of classification system configurations
        """
        return self.data.get('classification_systems', [])
    
    def is_eav_integration_enabled(self) -> bool:
        """
        Check if EAV integration is enabled.
        
        Returns:
            True if EAV integration is enabled, False otherwise
        """
        eav_config = self.data.get('eav_integration', {})
        return eav_config.get('enabled', False)

class YamlModelConfig(YamlConfigBase, ModelConfigInterface):
    """YAML-based implementation of ModelConfigInterface."""
    
    def get_model_type(self) -> str:
        """
        Get the model type.
        
        Returns:
            Model type
        """
        return self.data.get('model_type', 'random_forest')
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the model hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.data.get('hyperparameters', {})
    
    def get_evaluation_metrics(self) -> List[str]:
        """
        Get the evaluation metrics.
        
        Returns:
            List of evaluation metrics
        """
        return self.data.get('evaluation_metrics', ['accuracy', 'f1'])
    
    def get_cross_validation_config(self) -> Dict[str, Any]:
        """
        Get cross-validation configuration.
        
        Returns:
            Cross-validation configuration
        """
        return self.data.get('cross_validation', {'enabled': False, 'folds': 5})

class YamlPipelineConfig(YamlConfigBase, PipelineConfigInterface):
    """YAML-based implementation of PipelineConfigInterface."""
    
    def get_stages(self) -> List[str]:
        """
        Get the pipeline stages.
        
        Returns:
            List of pipeline stage names
        """
        return self.data.get('stages', [])
    
    def get_components(self) -> Dict[str, str]:
        """
        Get the component implementations to use.
        
        Returns:
            Dictionary mapping component types to implementation names
        """
        return self.data.get('components', {})
    
    def get_output_dir(self) -> str:
        """
        Get the output directory.
        
        Returns:
            Output directory path
        """
        return self.data.get('output_dir', 'outputs')
    
    def is_visualizations_enabled(self) -> bool:
        """
        Check if visualizations are enabled.
        
        Returns:
            True if visualizations are enabled, False otherwise
        """
        return self.data.get('visualizations', {}).get('enabled', False)