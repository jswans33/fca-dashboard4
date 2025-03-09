"""
Configuration Compatibility Module for NexusML

This module provides backward compatibility with existing code that uses
the old configuration access patterns.
"""

import os
from pathlib import Path
from typing import Any, Dict, Union, cast

from nexusml.config import get_project_root, DEFAULT_PATHS, CONFIG_FILES
from nexusml.config.manager import ConfigurationManager

# Singleton instance of ConfigurationManager
_config_manager = None

def get_config_manager() -> ConfigurationManager:
    """
    Get the singleton instance of ConfigurationManager.
    
    Returns:
        ConfigurationManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def load_settings() -> Dict[str, Any]:
    """
    Load settings from the configuration file.
    
    This function provides backward compatibility with the old load_settings function.
    
    Returns:
        Configuration settings as a dictionary
    """
    # Try to find a settings file
    try:
        from fca_dashboard.utils.path_util import get_config_path
        FCA_DASHBOARD_AVAILABLE = True
        try:
            settings_path = cast(Union[str, Path], get_config_path("settings.yml"))
        except Exception:
            settings_path = None
    except ImportError:
        FCA_DASHBOARD_AVAILABLE = False
        settings_path = get_project_root() / DEFAULT_PATHS["config_file"]
    
    # Check environment variable as fallback
    if not settings_path or not os.path.exists(str(settings_path)):
        settings_path_str = os.environ.get("NEXUSML_CONFIG", "")
        settings_path = Path(settings_path_str) if settings_path_str else None
    
    # Try to load settings
    if settings_path and os.path.exists(str(settings_path)):
        try:
            import yaml
            with open(settings_path, "r") as file:
                return yaml.safe_load(file) or {}
        except Exception as e:
            print(f"Warning: Could not load settings from {settings_path}: {e}")
    
    # Return default settings
    return {
        "nexusml": {
            "data_paths": {
                "training_data": str(
                    get_project_root() / "ingest" / "data" / "eq_ids.csv"
                ),
                "output_dir": str(get_project_root() / "outputs"),
            }
        }
    }

def get_data_path(path_key: str = "training_data") -> Union[str, Path]:
    """
    Get a data path from config or defaults.
    
    This function provides backward compatibility with the old get_data_path function.
    
    Args:
        path_key: Key for the path in the configuration
        
    Returns:
        Resolved path as string or Path object
    """
    root = get_project_root()
    
    # Try to load settings
    settings = load_settings()
    
    # Check in nexusml section first, then classifier section for backward compatibility
    nexusml_settings = settings.get("nexusml", {})
    classifier_settings = settings.get("classifier", {})
    
    # Merge settings, preferring nexusml if available
    merged_settings = {**classifier_settings, **nexusml_settings}
    
    # Get path from settings
    path = merged_settings.get("data_paths", {}).get(path_key)
    
    if not path:
        # Use default path
        path = os.path.join(str(root), DEFAULT_PATHS.get(path_key, ""))
    
    # If running in fca_dashboard context and path is not absolute, resolve it
    try:
        from fca_dashboard.utils.path_util import resolve_path
        FCA_DASHBOARD_AVAILABLE = True
        if FCA_DASHBOARD_AVAILABLE and not os.path.isabs(path):
            try:
                return cast(Union[str, Path], resolve_path(path))
            except Exception:
                # Fall back to local resolution
                return os.path.join(str(root), path)
    except ImportError:
        FCA_DASHBOARD_AVAILABLE = False
    
    # If path is not absolute, make it relative to project root
    if not os.path.isabs(path):
        return os.path.join(str(root), path)
    
    return path

def get_output_dir() -> Union[str, Path]:
    """
    Get the output directory path.
    
    This function provides backward compatibility with the old get_output_dir function.
    
    Returns:
        Path to the output directory as string or Path object
    """
    return get_data_path("output_dir")

def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using a dot-separated path.
    
    This function provides backward compatibility with the old get_config_value function.
    
    Args:
        key_path: Dot-separated path to the config value
        default: Default value to return if the key is not found
        
    Returns:
        The configuration value or the default
    """
    settings = load_settings()
    keys = key_path.split(".")
    
    # Navigate through the nested dictionary
    current = settings
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def get_config_file_path(config_name: str) -> Path:
    """
    Get the path to a specific configuration file.
    
    This function provides backward compatibility with the old get_config_file_path function.
    
    Args:
        config_name: Name of the configuration file (e.g., 'production_data_config')
        
    Returns:
        Path to the configuration file
    """
    root = get_project_root()
    
    # Get the relative path from CONFIG_FILES
    if config_name in CONFIG_FILES:
        relative_path = CONFIG_FILES[config_name]
    else:
        # Default to the config directory
        relative_path = f"config/{config_name}.yml"
    
    # Return the absolute path
    return root / relative_path