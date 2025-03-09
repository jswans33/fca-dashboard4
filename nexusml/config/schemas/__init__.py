"""
Configuration Schemas Package for NexusML

This package provides JSON Schema definitions for validating configuration files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

# Dictionary to cache loaded schemas
_schemas: Dict[str, dict] = {}

def get_schema_path(schema_name: str) -> Path:
    """
    Get the path to a schema file.
    
    Args:
        schema_name: Name of the schema file (without extension)
        
    Returns:
        Path to the schema file
    """
    schemas_dir = Path(__file__).resolve().parent
    return schemas_dir / f"{schema_name}.json"

def load_schema(schema_name: str) -> Optional[dict]:
    """
    Load a JSON Schema from file.
    
    Args:
        schema_name: Name of the schema file (without extension)
        
    Returns:
        Schema as a dictionary, or None if the schema file doesn't exist
    """
    # Check if schema is already loaded
    if schema_name in _schemas:
        return _schemas[schema_name]
    
    # Get the schema path
    schema_path = get_schema_path(schema_name)
    
    # Check if the schema file exists
    if not schema_path.exists():
        return None
    
    # Load the schema
    try:
        with open(schema_path, 'r') as f:
            schema = json.load(f)
        
        # Cache the schema
        _schemas[schema_name] = schema
        
        return schema
    except Exception as e:
        print(f"Error loading schema {schema_name}: {e}")
        return None

def validate_config(config: dict, schema_name: str) -> bool:
    """
    Validate a configuration against a JSON Schema.
    
    Args:
        config: Configuration to validate
        schema_name: Name of the schema file (without extension)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import jsonschema
        
        # Load the schema
        schema = load_schema(schema_name)
        if schema is None:
            print(f"Schema {schema_name} not found")
            return False
        
        # Validate the configuration
        jsonschema.validate(config, schema)
        return True
    except ImportError:
        # jsonschema not installed, skip validation
        return True
    except Exception as e:
        # Validation failed
        print(f"Configuration validation failed: {e}")
        return False

# Export available schemas
available_schemas = [
    path.stem for path in Path(__file__).resolve().parent.glob("*.json")
]