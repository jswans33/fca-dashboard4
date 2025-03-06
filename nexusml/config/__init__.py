"""
Centralized Configuration Module for NexusML

This module provides a unified approach to configuration management,
handling both standalone usage and integration with fca_dashboard.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import yaml

# Default paths
DEFAULT_PATHS = {
    "training_data": "ingest/data/eq_ids.csv",
    "output_dir": "outputs",
    "config_file": "config/settings.yml",
}

# Try to load from fca_dashboard if available (only once at import time)
try:
    from fca_dashboard.utils.path_util import get_config_path, resolve_path

    FCA_DASHBOARD_AVAILABLE = True
    # Store the imported functions to avoid "possibly unbound" errors
    FCA_GET_CONFIG_PATH = get_config_path
    FCA_RESOLVE_PATH = resolve_path
except ImportError:
    FCA_DASHBOARD_AVAILABLE = False
    # Define dummy functions that will never be called
    FCA_GET_CONFIG_PATH = None
    FCA_RESOLVE_PATH = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_path(path_key: str = "training_data") -> Union[str, Path]:
    """
    Get a data path from config or defaults.

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
    if (
        FCA_DASHBOARD_AVAILABLE
        and not os.path.isabs(path)
        and FCA_RESOLVE_PATH is not None
    ):
        try:
            return cast(Union[str, Path], FCA_RESOLVE_PATH(path))
        except Exception:
            # Fall back to local resolution
            return os.path.join(str(root), path)

    # If path is not absolute, make it relative to project root
    if not os.path.isabs(path):
        return os.path.join(str(root), path)

    return path


def get_output_dir() -> Union[str, Path]:
    """
    Get the output directory path.

    Returns:
        Path to the output directory as string or Path object
    """
    return get_data_path("output_dir")


def load_settings() -> Dict[str, Any]:
    """
    Load settings from the configuration file.

    Returns:
        Configuration settings as a dictionary
    """
    # Try to find a settings file
    if FCA_DASHBOARD_AVAILABLE and FCA_GET_CONFIG_PATH is not None:
        try:
            settings_path = cast(Union[str, Path], FCA_GET_CONFIG_PATH("settings.yml"))
        except Exception:
            settings_path = None
    else:
        settings_path = get_project_root() / DEFAULT_PATHS["config_file"]

    # Check environment variable as fallback
    if not settings_path or not os.path.exists(str(settings_path)):
        settings_path_str = os.environ.get("NEXUSML_CONFIG", "")
        settings_path = Path(settings_path_str) if settings_path_str else None

    # Try to load settings
    if settings_path and os.path.exists(str(settings_path)):
        try:
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


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using a dot-separated path.

    Args:
        key_path: Dot-separated path to the config value (e.g., 'nexusml.data_paths.training_data')
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
