"""
Model Card Configuration Module for NexusML

This module provides access to the model card configuration, which contains
information about the model, its inputs and outputs, data format, and technical
specifications.
"""

from typing import Any, Dict, List, Optional, Union, cast

from nexusml.config.sections import ConfigSection

class ModelCardConfig(ConfigSection):
    """Configuration for model card information."""
    
    @property
    def model_details(self) -> Dict[str, Any]:
        """Get the model details."""
        return self.data.get('model_details', {})
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return str(self.model_details.get('name', 'Unknown Model'))
    
    @property
    def model_version(self) -> str:
        """Get the model version."""
        return str(self.model_details.get('version', '0.0.0'))
    
    @property
    def inputs(self) -> List[Dict[str, Any]]:
        """Get the model inputs."""
        return list(self.data.get('inputs', []))
    
    @property
    def outputs(self) -> List[Dict[str, Any]]:
        """Get the model outputs."""
        return list(self.data.get('outputs', []))
    
    @property
    def data_format(self) -> Dict[str, Any]:
        """Get the data format information."""
        return dict(self.data.get('data_format', {}))
    
    @property
    def fields(self) -> List[Dict[str, Any]]:
        """Get the field definitions."""
        return list(self.data_format.get('fields', []))
    
    @property
    def required_fields(self) -> List[Dict[str, Any]]:
        """Get the required field definitions."""
        return [field for field in self.fields if field.get('is_required', False)]
    
    @property
    def target_fields(self) -> List[Dict[str, Any]]:
        """Get the target field definitions."""
        return [field for field in self.fields if field.get('is_target', False)]
    
    @property
    def mcaaid_mapping(self) -> Dict[str, str]:
        """Get the MCAA ID mapping."""
        return dict(self.data.get('mcaaid_mapping', {}))
    
    @property
    def standard_categories(self) -> List[str]:
        """Get the standard equipment categories."""
        return list(self.data.get('standard_categories', []))
    
    @property
    def technical_specifications(self) -> Dict[str, Any]:
        """Get the technical specifications."""
        return dict(self.data.get('technical_specifications', {}))
    
    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get the model hyperparameters."""
        return dict(self.technical_specifications.get('hyperparameters', {}))
    
    @property
    def feature_engineering(self) -> Dict[str, Any]:
        """Get the feature engineering specifications."""
        return dict(self.technical_specifications.get('feature_engineering', {}))
    
    @property
    def text_combinations(self) -> List[Dict[str, Any]]:
        """Get the text combination configurations."""
        return list(self.feature_engineering.get('text_combinations', []))
    
    @property
    def hierarchical_categories(self) -> List[Dict[str, Any]]:
        """Get the hierarchical category configurations."""
        return list(self.feature_engineering.get('hierarchical_categories', []))
    
    @property
    def reference_data(self) -> Dict[str, str]:
        """Get the reference data paths."""
        # Ensure all values are strings
        result = {}
        for key, value in self.data.get('reference_data', {}).items():
            result[str(key)] = str(value)
        return result
    
    def get_field_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a field definition by name.
        
        Args:
            name: Name of the field
            
        Returns:
            Field definition or None if not found
        """
        for field in self.fields:
            if field.get('name') == name:
                return field
        return None
    
    def get_field_description(self, name: str) -> str:
        """
        Get the description of a field.
        
        Args:
            name: Name of the field
            
        Returns:
            Description of the field or empty string if not found
        """
        field = self.get_field_by_name(name)
        return field.get('description', '') if field else ''
    
    def get_field_example(self, name: str) -> str:
        """
        Get an example value for a field.
        
        Args:
            name: Name of the field
            
        Returns:
            Example value for the field or empty string if not found
        """
        field = self.get_field_by_name(name)
        return field.get('example', '') if field else ''
    
    def get_field_data_type(self, name: str) -> str:
        """
        Get the data type of a field.
        
        Args:
            name: Name of the field
            
        Returns:
            Data type of the field or 'string' if not found
        """
        field = self.get_field_by_name(name)
        return field.get('data_type', 'string') if field else 'string'
    
    def is_field_required(self, name: str) -> bool:
        """
        Check if a field is required.
        
        Args:
            name: Name of the field
            
        Returns:
            True if the field is required, False otherwise
        """
        field = self.get_field_by_name(name)
        return field.get('is_required', False) if field else False
    
    def is_field_target(self, name: str) -> bool:
        """
        Check if a field is a target for prediction.
        
        Args:
            name: Name of the field
            
        Returns:
            True if the field is a target, False otherwise
        """
        field = self.get_field_by_name(name)
        return field.get('is_target', False) if field else False
    
    def get_mcaaid_for_system_category(self, system_category: str) -> str:
        """
        Get the MCAA ID for a system category.
        
        Args:
            system_category: System category
            
        Returns:
            MCAA ID for the system category or empty string if not found
        """
        return self.mcaaid_mapping.get(system_category, '')
    
    def get_reference_data_path(self, reference_type: str) -> str:
        """
        Get the path to a reference data file.
        
        Args:
            reference_type: Type of reference data
            
        Returns:
            Path to the reference data file or empty string if not found
        """
        return self.reference_data.get(f"{reference_type}_file", '')