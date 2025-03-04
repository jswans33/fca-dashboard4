"""
Environment and configuration utilities.

This module provides functions to safely access environment variables
and check the current running environment. It integrates with the application's
settings module for consistent configuration access.
"""

import os
from typing import Any

from fca_dashboard.config.settings import settings

# The environment variable name used to determine the current environment
ENV_VAR_NAME = "ENVIRONMENT"


def get_env_var(key: str, fallback: Any = None) -> Any:
    """
    Safely access environment variables with an optional fallback value.
    
    This function first checks if the environment variable is set directly in
    the OS environment. If not found, it attempts to retrieve it from the
    application settings. If still not found, it returns the fallback value.
    
    Args:
        key: The name of the environment variable to retrieve
        fallback: The value to return if the environment variable is not set
        
    Returns:
        The value of the environment variable if it exists, otherwise the fallback value
    """
    # First check OS environment variables
    value = os.environ.get(key)
    
    # If not found in OS environment, check application settings
    if value is None:
        # Look for the key in the env section of settings
        value = settings.get(f"env.{key}")
        
        # If still not found, look for it at the top level
        if value is None:
            value = settings.get(key)
    
    # If still not found, return the fallback
    if value is None:
        return fallback
        
    return value


def is_dev() -> bool:
    """
    Check if the current environment is development.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is development.
    
    Returns:
        True if the current environment is development, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["development", "dev"]


def is_prod() -> bool:
    """
    Check if the current environment is production.
    
    This function checks the environment variable specified by ENV_VAR_NAME
    to determine if the current environment is production.
    
    Returns:
        True if the current environment is production, False otherwise
    """
    env = str(get_env_var(ENV_VAR_NAME, "")).lower()
    return env in ["production", "prod"]


def get_environment() -> str:
    """
    Get the current environment name.
    
    Returns:
        The current environment name (e.g., 'development', 'production', 'staging')
        or 'unknown' if not set
    """
    return str(get_env_var(ENV_VAR_NAME, "unknown")).lower()
