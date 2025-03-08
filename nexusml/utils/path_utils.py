"""
Path Utilities for NexusML

This module provides robust path handling utilities for the NexusML package,
ensuring consistent path resolution across different execution contexts
(scripts, notebooks, etc.)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        Path object pointing to the project root directory
    """
    # Assuming this module is in nexusml/utils/
    module_dir = Path(__file__).resolve().parent
    # Go up two levels to get to the project root (nexusml)
    return module_dir.parent.parent


def get_nexusml_root() -> Path:
    """
    Get the absolute path to the nexusml package root directory.
    
    Returns:
        Path object pointing to the nexusml package root
    """
    # Assuming this module is in nexusml/utils/
    module_dir = Path(__file__).resolve().parent
    # Go up one level to get to the nexusml package root
    return module_dir.parent


def ensure_nexusml_in_path() -> None:
    """
    Ensure that the nexusml package is in the Python path.
    This is useful for notebooks and scripts that need to import nexusml.
    """
    project_root = str(get_project_root())
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path")
    
    # Also add the parent directory of nexusml to support direct imports
    parent_dir = str(get_project_root().parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        print(f"Added {parent_dir} to Python path")


def resolve_path(path: Union[str, Path], relative_to: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve a path to an absolute path.
    
    Args:
        path: The path to resolve
        relative_to: The directory to resolve relative paths against.
                    If None, uses the current working directory.
    
    Returns:
        Resolved absolute Path object
    """
    if relative_to is None:
        return Path(path).resolve()
    else:
        return Path(relative_to).resolve() / Path(path)


def find_data_files(
    search_paths: Optional[List[Union[str, Path]]] = None,
    file_extensions: List[str] = [".xlsx", ".csv"],
    recursive: bool = False
) -> Dict[str, str]:
    """
    Find data files in the specified search paths.
    
    Args:
        search_paths: List of paths to search. If None, uses default locations.
        file_extensions: List of file extensions to include
        recursive: Whether to search recursively in subdirectories
    
    Returns:
        Dictionary mapping file names to their full paths
    """
    if search_paths is None:
        # Default locations to search
        project_root = get_project_root()
        search_paths = [
            project_root / "data",
            project_root / "examples",
            project_root.parent / "examples",
            project_root.parent / "uploads",
        ]
    
    data_files: Dict[str, str] = {}
    for base_path in search_paths:
        base_path = Path(base_path)
        if not base_path.exists():
            continue
            
        if recursive:
            # Recursive search
            for ext in file_extensions:
                for file_path in base_path.glob(f"**/*{ext}"):
                    if file_path.is_file():
                        data_files[file_path.name] = str(file_path)
        else:
            # Non-recursive search
            for file_path in base_path.iterdir():
                if file_path.is_file() and any(str(file_path).endswith(ext) for ext in file_extensions):
                    data_files[file_path.name] = str(file_path)
    
    return data_files


# Add a convenience function to initialize the environment for notebooks
def setup_notebook_environment() -> Dict[str, str]:
    """
    Set up the environment for Jupyter notebooks.
    This ensures that the nexusml package can be imported correctly.
    
    Returns:
        Dictionary of useful paths for notebooks
    """
    ensure_nexusml_in_path()
    
    # Create and return common paths that might be useful in notebooks
    paths: Dict[str, str] = {
        "project_root": str(get_project_root()),
        "nexusml_root": str(get_nexusml_root()),
        "data_dir": str(get_project_root() / "data"),
        "examples_dir": str(get_project_root() / "examples"),
        "outputs_dir": str(get_project_root() / "outputs"),
    }
    return paths


if __name__ == "__main__":
    # If run as a script, print the project root and ensure nexusml is in path
    print(f"Project root: {get_project_root()}")
    print(f"NexusML root: {get_nexusml_root()}")
    ensure_nexusml_in_path()
    print(f"Python path: {sys.path}")