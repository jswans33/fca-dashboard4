"""
Configuration Sections Module for NexusML

This module provides base classes for configuration sections, which are used
to provide type-safe access to configuration values.
"""

from typing import Any, Dict, Generic, TypeVar

T = TypeVar('T')

class ConfigSection(Generic[T]):
    """Base class for configuration sections."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.data.get(key, default)
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'training_data.default_path')
            default: Default value to return if the key is not found
            
        Returns:
            The configuration value or the default
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