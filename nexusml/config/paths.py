"""
Path Management Module for NexusML

This module provides utilities for path resolution across different environments,
handling both absolute and relative paths, and providing a consistent API for
accessing paths in the project.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Union, cast

from nexusml.config import get_project_root, DEFAULT_PATHS

# Try to load from fca_dashboard if available
try:
    from fca_dashboard.utils.path_util import resolve_path as fca_resolve_path
    FCA_DASHBOARD_AVAILABLE = True
except ImportError:
    FCA_DASHBOARD_AVAILABLE = False
    fca_resolve_path = None


class PathResolver:
    """
    Resolves paths across different environments.
    
    This class provides a unified API for resolving paths, handling both
    absolute and relative paths, and providing context-specific path resolution.
    """
    
    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the path resolver.
        
        Args:
            root_dir: Root directory for path resolution. If None, uses the project root.
        """
        self.root_dir = Path(root_dir) if root_dir else get_project_root()
        self.path_cache: Dict[str, Path] = {}
        
        # Environment-specific configuration
        self.environment = os.environ.get('NEXUSML_ENV', 'production')
    
    def resolve_path(self, path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        Resolve a path relative to the root directory or a specified base directory.
        
        Args:
            path: Path to resolve
            base_dir: Base directory for relative paths. If None, uses the root directory.
            
        Returns:
            Resolved path
        """
        # If path is already absolute, return it
        if os.path.isabs(str(path)):
            return Path(path)
        
        # If base_dir is provided, use it as the base
        if base_dir:
            base = Path(base_dir)
            if not base.is_absolute():
                base = self.root_dir / base
            return base / path
        
        # Try to use fca_dashboard's resolve_path if available
        if FCA_DASHBOARD_AVAILABLE and fca_resolve_path:
            try:
                return cast(Path, fca_resolve_path(path))
            except Exception:
                # Fall back to local resolution
                pass
        
        # Resolve relative to root directory
        return self.root_dir / path
    
    def get_data_path(self, path_key: str = "training_data") -> Path:
        """
        Get a data path from the configuration.
        
        Args:
            path_key: Key for the path in the configuration
            
        Returns:
            Resolved path
        """
        # Check if path is already cached
        cache_key = f"data_{path_key}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Get path from configuration
        from nexusml.config.compatibility import get_data_path
        path = get_data_path(path_key)
        
        # Resolve and cache the path
        resolved_path = self.resolve_path(path)
        self.path_cache[cache_key] = resolved_path
        
        return resolved_path
    
    def get_config_path(self, config_name: str) -> Path:
        """
        Get the path to a configuration file.
        
        Args:
            config_name: Name of the configuration file (without extension)
            
        Returns:
            Resolved path to the configuration file
        """
        # Check if path is already cached
        cache_key = f"config_{config_name}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Try environment-specific config first
        if self.environment != 'production':
            env_path = self.root_dir / "config" / f"{config_name}.{self.environment}.yml"
            if env_path.exists():
                self.path_cache[cache_key] = env_path
                return env_path
        
        # Use standard config path
        path = self.root_dir / "config" / f"{config_name}.yml"
        self.path_cache[cache_key] = path
        
        return path
    
    def get_output_path(self, output_type: str = "models") -> Path:
        """
        Get the path to an output directory.
        
        Args:
            output_type: Type of output (e.g., "models", "visualizations")
            
        Returns:
            Resolved path to the output directory
        """
        # Check if path is already cached
        cache_key = f"output_{output_type}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Get base output directory
        output_dir = self.get_data_path("output_dir")
        
        # Create type-specific output directory
        type_dir = output_dir / output_type
        
        # Create directory if it doesn't exist
        type_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache and return the path
        self.path_cache[cache_key] = type_dir
        return type_dir
    
    def get_reference_path(self, reference_type: str) -> Path:
        """
        Get the path to a reference data file.
        
        Args:
            reference_type: Type of reference data (e.g., "omniclass", "uniformat")
            
        Returns:
            Resolved path to the reference data file
        """
        # Check if path is already cached
        cache_key = f"reference_{reference_type}"
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Define reference data paths
        reference_paths = {
            "omniclass": "files/omniclass_tables/omniclass_23.csv",
            "uniformat": "files/uniformat/uniformat_ii.csv",
            "masterformat": "files/masterformat/masterformat_2018.csv",
            "mcaa": "files/mcaa-glossary/mcaa_glossary.csv",
        }
        
        try:
            # Get path for the specified reference type
            if reference_type in reference_paths:
                path = self.resolve_path(reference_paths[reference_type])
            else:
                # Default to a subdirectory in the reference directory
                path = self.resolve_path(f"files/{reference_type}/{reference_type}.csv")
                
            # Check if the path exists
            if not path.exists():
                # Try alternative locations
                alt_paths = [
                    self.root_dir / "files" / reference_type / f"{reference_type}.csv",
                    self.root_dir.parent / "files" / reference_type / f"{reference_type}.csv",
                    self.root_dir.parent / "fca_dashboard" / "files" / reference_type / f"{reference_type}.csv"
                ]
                
                for alt_path in alt_paths:
                    if alt_path.exists():
                        path = alt_path
                        break
            
            # Cache and return the path
            self.path_cache[cache_key] = path
            return path
            
        except Exception as e:
            print(f"Failed to resolve reference path for {reference_type}: {e}")
            # Return a default path
            default_path = self.root_dir / "files" / reference_type / f"{reference_type}.csv"
            self.path_cache[cache_key] = default_path
            return default_path
    
    def clear_cache(self) -> None:
        """Clear the path cache."""
        self.path_cache.clear()


# Create a singleton instance of PathResolver
_path_resolver = None

def get_path_resolver() -> PathResolver:
    """
    Get the singleton instance of PathResolver.
    
    Returns:
        PathResolver instance
    """
    global _path_resolver
    if _path_resolver is None:
        _path_resolver = PathResolver()
    return _path_resolver

def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path relative to the root directory or a specified base directory.
    
    This function provides a convenient way to resolve paths without creating a PathResolver instance.
    
    Args:
        path: Path to resolve
        base_dir: Base directory for relative paths. If None, uses the root directory.
        
    Returns:
        Resolved path
    """
    return get_path_resolver().resolve_path(path, base_dir)

def get_data_path(path_key: str = "training_data") -> Path:
    """
    Get a data path from the configuration.
    
    This function provides a convenient way to get data paths without creating a PathResolver instance.
    
    Args:
        path_key: Key for the path in the configuration
        
    Returns:
        Resolved path
    """
    return get_path_resolver().get_data_path(path_key)

def get_config_path(config_name: str) -> Path:
    """
    Get the path to a configuration file.
    
    This function provides a convenient way to get configuration paths without creating a PathResolver instance.
    
    Args:
        config_name: Name of the configuration file (without extension)
        
    Returns:
        Resolved path to the configuration file
    """
    return get_path_resolver().get_config_path(config_name)

def get_output_path(output_type: str = "models") -> Path:
    """
    Get the path to an output directory.
    
    This function provides a convenient way to get output paths without creating a PathResolver instance.
    
    Args:
        output_type: Type of output (e.g., "models", "visualizations")
        
    Returns:
        Resolved path to the output directory
    """
    return get_path_resolver().get_output_path(output_type)

def get_reference_path(reference_type: str) -> Path:
    """
    Get the path to a reference data file.
    
    This function provides a convenient way to get reference paths without creating a PathResolver instance.
    
    Args:
        reference_type: Type of reference data (e.g., "omniclass", "uniformat")
        
    Returns:
        Resolved path to the reference data file
    """
    return get_path_resolver().get_reference_path(reference_type)