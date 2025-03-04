"""
Unit tests for the path utility module.

This module contains tests for the path utility functions in the
fca_dashboard.utils.path_util module.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fca_dashboard.utils.path_util import get_config_path, get_logs_path, get_root_dir, resolve_path


def test_get_root_dir() -> None:
    """Test that get_root_dir returns a Path object to the project root."""
    root_dir = get_root_dir()
    assert isinstance(root_dir, Path)
    # Check that the directory exists
    assert root_dir.exists()
    # Check that it contains expected project files/directories
    assert (root_dir / "fca_dashboard").exists()
    assert (root_dir / "setup.py").exists() or (root_dir / "pyproject.toml").exists()


def test_get_config_path_default() -> None:
    """Test get_config_path with default filename."""
    config_path = get_config_path()
    assert isinstance(config_path, Path)
    assert config_path.name == "settings.yml"
    # Use os.path.join to handle platform-specific path separators
    assert os.path.join("fca_dashboard", "config") in str(config_path)


def test_get_config_path_custom() -> None:
    """Test get_config_path with custom filename."""
    custom_filename = "custom_settings.yml"
    config_path = get_config_path(custom_filename)
    assert isinstance(config_path, Path)
    assert config_path.name == custom_filename
    # Use os.path.join to handle platform-specific path separators
    assert os.path.join("fca_dashboard", "config") in str(config_path)


@patch("fca_dashboard.utils.path_util.logger")
def test_get_config_path_nonexistent(mock_logger: Any) -> None:
    """Test get_config_path with a nonexistent file."""
    nonexistent_file = "nonexistent_file.yml"
    config_path = get_config_path(nonexistent_file)
    assert isinstance(config_path, Path)
    assert config_path.name == nonexistent_file
    # Check that a warning was logged
    mock_logger.warning.assert_called_once()


def test_get_logs_path_default() -> None:
    """Test get_logs_path with default filename."""
    logs_path = get_logs_path()
    assert isinstance(logs_path, Path)
    assert logs_path.name == "fca_dashboard.log"
    assert "logs" in str(logs_path)
    # Check that the logs directory exists
    assert logs_path.parent.exists()


def test_get_logs_path_custom() -> None:
    """Test get_logs_path with custom filename."""
    custom_filename = "custom.log"
    logs_path = get_logs_path(custom_filename)
    assert isinstance(logs_path, Path)
    assert logs_path.name == custom_filename
    assert "logs" in str(logs_path)


def test_resolve_path_absolute() -> None:
    """Test resolve_path with an absolute path."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        # Test with absolute path
        resolved_path = resolve_path(temp_path)
        assert resolved_path == temp_path
    finally:
        # Clean up
        os.unlink(temp_path)


def test_resolve_path_existing_relative() -> None:
    """Test resolve_path with an existing relative path."""
    # Create a temporary file in the current directory
    with tempfile.NamedTemporaryFile(dir=".", delete=False) as temp_file:
        temp_name = Path(temp_file.name).name

    try:
        # Test with relative path that exists
        resolved_path = resolve_path(temp_name)
        assert resolved_path.is_absolute()
        assert resolved_path.name == temp_name
    finally:
        # Clean up
        os.unlink(temp_name)


@patch("fca_dashboard.utils.path_util.logger")
def test_resolve_path_nonexistent(mock_logger: Any) -> None:
    """Test resolve_path with a nonexistent path."""
    nonexistent_path = "nonexistent_file.txt"
    resolved_path = resolve_path(nonexistent_path)
    assert isinstance(resolved_path, Path)
    assert resolved_path.name == nonexistent_path
    # Check that a warning message was logged (changed from debug to warning)
    mock_logger.warning.assert_called()


def test_resolve_path_with_base_dir() -> None:
    """Test resolve_path with a base directory."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file in the temporary directory
        temp_file_path = Path(temp_dir) / "test_file.txt"
        with open(temp_file_path, "w") as f:
            f.write("test")

        # Test resolving the file relative to the base directory
        resolved_path = resolve_path("test_file.txt", base_dir=Path(temp_dir))
        assert resolved_path.is_absolute()
        assert resolved_path.name == "test_file.txt"
        assert resolved_path.parent == Path(temp_dir).resolve()


def test_resolve_path_with_fca_dashboard_subdir() -> None:
    """Test resolve_path with a path in the fca_dashboard subdirectory."""
    # Mock a base directory with an fca_dashboard subdirectory
    with tempfile.TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)
        fca_dir = base_dir / "fca_dashboard"
        fca_dir.mkdir()

        # Create a file in the fca_dashboard subdirectory
        test_file = fca_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test")

        # Test resolving the file
        resolved_path = resolve_path("test_file.txt", base_dir=base_dir)
        assert resolved_path.is_absolute()
        assert resolved_path.name == "test_file.txt"
        assert resolved_path.parent == fca_dir.resolve()
