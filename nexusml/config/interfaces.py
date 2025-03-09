"""
Configuration Interfaces Module for NexusML

This module defines interfaces for different configuration types,
following the Interface Segregation Principle to provide focused interfaces
for each configuration concern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

class ConfigInterface(ABC):
    """Base interface for all configuration types."""
    
    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        pass
    
    @abstractmethod
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value
            default: Default value if path is not found
            
        Returns:
            Configuration value or default
        """
        pass

class DataConfigInterface(ConfigInterface):
    """Interface for data configuration."""
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns.
        
        Returns:
            List of required column names
        """
        pass
    
    @abstractmethod
    def get_source_columns(self) -> List[str]:
        """
        Get the list of source columns (not derived during feature engineering).
        
        Returns:
            List of source column names
        """
        pass
    
    @abstractmethod
    def get_target_columns(self) -> List[str]:
        """
        Get the list of target columns (derived during feature engineering).
        
        Returns:
            List of target column names
        """
        pass
    
    @abstractmethod
    def get_critical_columns(self) -> List[str]:
        """
        Get the list of critical columns that must not have missing values.
        
        Returns:
            List of critical column names
        """
        pass
    
    @abstractmethod
    def get_column_default(self, column_name: str) -> Any:
        """
        Get the default value for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Default value for the column
        """
        pass
    
    @abstractmethod
    def get_column_data_type(self, column_name: str) -> str:
        """
        Get the data type for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Data type of the column
        """
        pass

class FeatureConfigInterface(ConfigInterface):
    """Interface for feature engineering configuration."""
    
    @abstractmethod
    def get_text_combinations(self) -> List[Dict[str, Any]]:
        """
        Get text combination configurations.
        
        Returns:
            List of text combination configurations
        """
        pass
    
    @abstractmethod
    def get_numeric_columns(self) -> List[Dict[str, Any]]:
        """
        Get numeric column configurations.
        
        Returns:
            List of numeric column configurations
        """
        pass
    
    @abstractmethod
    def get_hierarchies(self) -> List[Dict[str, Any]]:
        """
        Get hierarchy configurations.
        
        Returns:
            List of hierarchy configurations
        """
        pass
    
    @abstractmethod
    def get_column_mappings(self) -> List[Dict[str, str]]:
        """
        Get column mapping configurations.
        
        Returns:
            List of column mapping configurations
        """
        pass
    
    @abstractmethod
    def get_classification_systems(self) -> List[Dict[str, Any]]:
        """
        Get classification system configurations.
        
        Returns:
            List of classification system configurations
        """
        pass
    
    @abstractmethod
    def is_eav_integration_enabled(self) -> bool:
        """
        Check if EAV integration is enabled.
        
        Returns:
            True if EAV integration is enabled, False otherwise
        """
        pass

class ModelConfigInterface(ConfigInterface):
    """Interface for model building and training configuration."""
    
    @abstractmethod
    def get_model_type(self) -> str:
        """
        Get the model type.
        
        Returns:
            Model type
        """
        pass
    
    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get the model hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        pass
    
    @abstractmethod
    def get_evaluation_metrics(self) -> List[str]:
        """
        Get the evaluation metrics.
        
        Returns:
            List of evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_cross_validation_config(self) -> Dict[str, Any]:
        """
        Get cross-validation configuration.
        
        Returns:
            Cross-validation configuration
        """
        pass

class PipelineConfigInterface(ConfigInterface):
    """Interface for pipeline orchestration configuration."""
    
    @abstractmethod
    def get_stages(self) -> List[str]:
        """
        Get the pipeline stages.
        
        Returns:
            List of pipeline stage names
        """
        pass
    
    @abstractmethod
    def get_components(self) -> Dict[str, str]:
        """
        Get the component implementations to use.
        
        Returns:
            Dictionary mapping component types to implementation names
        """
        pass
    
    @abstractmethod
    def get_output_dir(self) -> str:
        """
        Get the output directory.
        
        Returns:
            Output directory path
        """
        pass
    
    @abstractmethod
    def is_visualizations_enabled(self) -> bool:
        """
        Check if visualizations are enabled.
        
        Returns:
            True if visualizations are enabled, False otherwise
        """
        pass