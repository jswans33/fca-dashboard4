"""
Pipeline utility functions for the FCA Dashboard application.

This module provides utility functions for common pipeline operations such as
clearing output directories, managing pipeline state, and other pipeline-related
tasks that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

from fca_dashboard.config.settings import settings
from fca_dashboard.utils.logging_config import get_logger
from fca_dashboard.utils.path_util import resolve_path


class PipelineUtilError(Exception):
    """Exception raised for errors in pipeline utility operations."""
    pass


def get_pipeline_output_dir(pipeline_name: str) -> Path:
    """
    Get the output directory for a specific pipeline.

    Args:
        pipeline_name: Name of the pipeline (e.g., 'medtronics', 'wichita')

    Returns:
        Path object representing the pipeline's output directory

    Raises:
        PipelineUtilError: If the pipeline settings or output directory cannot be found
    """
    logger = get_logger("pipeline_util")
    
    # Get the output directory from settings
    output_dir = settings.get(f"{pipeline_name}.output_dir", None)
    
    if not output_dir:
        error_msg = f"Output directory not configured for pipeline: {pipeline_name}"
        logger.error(error_msg)
        raise PipelineUtilError(error_msg)
    
    # Resolve the path
    try:
        resolved_path = resolve_path(output_dir)
        logger.debug(f"Resolved output directory for {pipeline_name}: {resolved_path}")
        return resolved_path
    except Exception as e:
        error_msg = f"Failed to resolve output directory for pipeline {pipeline_name}: {str(e)}"
        logger.error(error_msg)
        raise PipelineUtilError(error_msg) from e


def clear_output_directory(
    pipeline_name: str,
    preserve_files: Optional[List[str]] = None,
    preserve_extensions: Optional[List[str]] = None,
    dry_run: bool = False
) -> List[str]:
    """
    Clear the output directory for a specific pipeline.

    Args:
        pipeline_name: Name of the pipeline (e.g., 'medtronics', 'wichita')
        preserve_files: List of filenames to preserve (not delete)
        preserve_extensions: List of file extensions to preserve (e.g., ['.db', '.log'])
        dry_run: If True, only log what would be deleted without actually deleting

    Returns:
        List of files that were deleted or would be deleted (if dry_run=True)

    Raises:
        PipelineUtilError: If the pipeline settings or output directory cannot be found
    """
    logger = get_logger("pipeline_util")
    
    # Initialize lists
    preserve_files = preserve_files or []
    preserve_extensions = preserve_extensions or []
    deleted_files = []
    
    try:
        # Get the output directory
        output_dir = get_pipeline_output_dir(pipeline_name)
        
        # Check if the directory exists
        if not output_dir.exists():
            error_msg = f"Output directory does not exist: {output_dir}"
            logger.error(error_msg)
            raise PipelineUtilError(error_msg)
        
        logger.info(f"Clearing output directory for pipeline {pipeline_name}: {output_dir}")
        
        # Walk through the directory and collect files to delete
        for root, dirs, files in os.walk(str(output_dir)):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(output_dir)
                
                # Check if file should be preserved
                if file in preserve_files:
                    logger.debug(f"Preserving file (by name): {rel_path}")
                    continue
                
                # Check if extension should be preserved
                if any(file.endswith(ext) for ext in preserve_extensions):
                    logger.debug(f"Preserving file (by extension): {rel_path}")
                    continue
                
                # Add to the list of files to delete
                deleted_files.append(str(rel_path))
                
                if not dry_run:
                    try:
                        file_path.unlink()
                        logger.debug(f"Deleted file: {rel_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete file {rel_path}: {str(e)}")
        
        # Remove empty directories (if not dry run)
        if not dry_run:
            for root, dirs, files in os.walk(str(output_dir), topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if not any(dir_path.iterdir()):
                        try:
                            dir_path.rmdir()
                            logger.debug(f"Removed empty directory: {dir_path.relative_to(output_dir)}")
                        except Exception as e:
                            logger.warning(f"Failed to remove directory {dir_path.relative_to(output_dir)}: {str(e)}")
        
        # Log summary
        if dry_run:
            logger.info(f"Dry run: Would delete {len(deleted_files)} files from {output_dir}")
        else:
            logger.info(f"Deleted {len(deleted_files)} files from {output_dir}")
        
        return deleted_files
    
    except PipelineUtilError:
        # Re-raise PipelineUtilError exceptions
        raise
    except Exception as e:
        # Wrap other exceptions
        error_msg = f"Error clearing output directory for pipeline {pipeline_name}: {str(e)}"
        logger.error(error_msg)
        raise PipelineUtilError(error_msg) from e