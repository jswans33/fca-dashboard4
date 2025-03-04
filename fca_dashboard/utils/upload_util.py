"""
File upload utility module for the FCA Dashboard application.

This module provides functionality for uploading files to specified destinations,
with features like duplicate handling, error handling, and logging.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Union

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


class FileUploadError(FCADashboardError):
    """Exception raised for errors during file upload operations."""
    pass


def upload_file(source: Union[str, Path], destination_dir: Union[str, Path], target_filename: str = None) -> bool:
    """
    Upload a file by copying it to the destination directory.
    
    Args:
        source: Path to the file to upload. Can be absolute or relative.
        destination_dir: Directory where the file should be uploaded. Can be absolute or relative.
        target_filename: Optional filename to use in the destination. If None, uses the source filename.
        
    Returns:
        True if the file was successfully uploaded.
        
    Raises:
        FileNotFoundError: If the source file does not exist.
        FileNotFoundError: If the destination directory does not exist.
        FileUploadError: If an error occurs during the upload process.
    """
    logger = get_logger("upload_util")
    
    # Resolve paths to handle relative paths correctly
    source_path = resolve_path(source)
    dest_dir_path = resolve_path(destination_dir)
    
    # Validate source file
    if not source_path.is_file():
        logger.error(f"Source file not found: {source_path}")
        raise FileNotFoundError(f"Source file not found: {source_path}")
    
    # Validate destination directory
    if not dest_dir_path.is_dir():
        logger.error(f"Destination directory not found: {dest_dir_path}")
        raise FileNotFoundError(f"Destination directory not found: {dest_dir_path}")
    
    # Get the filename
    filename = target_filename if target_filename else source_path.name
    dest_path = dest_dir_path / filename
    
    # Handle duplicate filenames
    if dest_path.exists():
        logger.info(f"File already exists at destination: {dest_path}")
        # Generate a new filename with timestamp
        base_name = dest_path.stem
        extension = dest_path.suffix
        timestamp = int(time.time())
        new_filename = f"{base_name}_{timestamp}{extension}"
        dest_path = dest_dir_path / new_filename
        logger.info(f"Renamed to: {dest_path}")
    
    try:
        # Copy the file to the destination
        logger.info(f"Uploading file {source_path} to {dest_path}")
        shutil.copy(source_path, dest_path)
        logger.info(f"Successfully uploaded file to {dest_path}")
        return True
    except Exception as e:
        error_msg = f"Error uploading file {source_path} to {dest_path}: {str(e)}"
        logger.error(error_msg)
        raise FileUploadError(error_msg) from e