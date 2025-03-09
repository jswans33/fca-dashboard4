"""
Centralized Configuration Module for NexusML

This module provides a unified approach to configuration management,
handling both standalone usage and integration with fca_dashboard.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union, cast, Optional, Callable

import yaml  # type: ignore

# Default paths
DEFAULT_PATHS = {
    "training_data": "ingest/data/eq_ids.csv",
    "output_dir": "outputs",
    "config_file": "config/settings.yml",
}

# Configuration file paths
CONFIG_FILES = {
    "production_data_config": "config/production_data_config.yml",
    "feature_config": "config/feature_config.yml",
    "classification_config": "config/classification_config.yml",
    "reference_config": "config/reference_config.yml",
    "nexusml_config": "config/nexusml_config.yml",
}

# Try to load from fca_dashboard if available (only once at import time)
try:
    from fca_dashboard.utils.path_util import get_config_path, resolve_path

    FCA_DASHBOARD_AVAILABLE = True
    # Store the imported functions to avoid "possibly unbound" errors
    FCA_GET_CONFIG_PATH = get_config_path  # type: ignore
    FCA_RESOLVE_PATH = resolve_path  # type: ignore
except ImportError:
    FCA_DASHBOARD_AVAILABLE = False
    # Define dummy functions that will never be called
    FCA_GET_CONFIG_PATH = None  # type: ignore
    FCA_RESOLVE_PATH = None  # type: ignore


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


# Import new functionality
from nexusml.config.manager import ConfigurationManager
from nexusml.config.interfaces import (
    ConfigInterface,
    DataConfigInterface,
    FeatureConfigInterface,
    ModelConfigInterface,
    PipelineConfigInterface,
)
from nexusml.config.implementations import (
    YamlConfigBase,
    YamlDataConfig,
    YamlFeatureConfig,
    YamlModelConfig,
    YamlPipelineConfig,
)
from nexusml.config.model_card import ModelCardConfig
from nexusml.config.paths import (
    PathResolver,
    get_path_resolver,
    resolve_path,
    get_data_path as get_resolved_data_path,
    get_config_path as get_resolved_config_path,
    get_output_path,
    get_reference_path,
)
from nexusml.config.validation import (
    ConfigurationValidator,
    get_config_validator,
    validate_all_configs,
    validate_config_compatibility,
)

# For backward compatibility, import the compatibility functions
from nexusml.config.compatibility import (
    get_data_path,
    get_output_dir,
    load_settings,
    get_config_file_path,
    get_config_value,
    get_config_manager,
)

# Create a singleton instance of ConfigurationManager
_config_manager = None

def get_configuration_manager() -> ConfigurationManager:
    """
    Get the singleton instance of ConfigurationManager.
    
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

# Export public API
__all__ = [
    # Core functionality
    'get_project_root',
    'DEFAULT_PATHS',
    'CONFIG_FILES',
    
    # New configuration management
    'ConfigurationManager',
    'ConfigInterface',
    'DataConfigInterface',
    'FeatureConfigInterface',
    'ModelConfigInterface',
    'PipelineConfigInterface',
    'YamlConfigBase',
    'YamlDataConfig',
    'YamlFeatureConfig',
    'YamlModelConfig',
    'YamlPipelineConfig',
    'ModelCardConfig',
    'get_configuration_manager',
    
    # Path management
    'PathResolver',
    'get_path_resolver',
    'resolve_path',
    'get_resolved_data_path',
    'get_resolved_config_path',
    'get_output_path',
    'get_reference_path',
    
    # Configuration validation
    'ConfigurationValidator',
    'get_config_validator',
    'validate_all_configs',
    'validate_config_compatibility',
    
    # Backward compatibility
    'get_data_path',
    'get_output_dir',
    'load_settings',
    'get_config_file_path',
    'get_config_value',
    'get_config_manager',
]
