"""
Unit tests for the pipeline utility module.

This module contains tests for the pipeline utility functions that handle
operations like clearing output directories and managing pipeline state.
"""

import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from fca_dashboard.utils.error_handler import FCADashboardError
from fca_dashboard.utils.path_util import get_root_dir


# Import the module under test (will be implemented after tests)
# This allows us to mock it properly in tests
@pytest.fixture
def pipeline_util():
    """Import and return the pipeline_util module."""
    with mock.patch("fca_dashboard.utils.pipeline_util.get_logger"):
        from fca_dashboard.utils import pipeline_util
        return pipeline_util


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory structure for testing."""
    # Create a mock pipeline output directory
    pipeline_dir = tmp_path / "pipeline" / "test_pipeline"
    pipeline_dir.mkdir(parents=True)
    
    # Create some test files
    test_files = [
        "file1.txt",
        "file2.csv",
        "important.db",
        "subdir/nested_file.txt"
    ]
    
    for file_path in test_files:
        full_path = pipeline_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text("test content")
    
    return pipeline_dir


class TestClearOutputDirectory:
    """Tests for the clear_output_directory function."""
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    def test_clear_output_directory_not_found(self, mock_settings, pipeline_util):
        """Test handling of non-existent output directory."""
        # Setup
        mock_settings.get.return_value = "/nonexistent/path"
        
        # Execute and verify
        with pytest.raises(pipeline_util.PipelineUtilError):
            pipeline_util.clear_output_directory("nonexistent_pipeline")
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    @mock.patch("fca_dashboard.utils.pipeline_util.resolve_path")
    def test_clear_output_directory_all_files(self, mock_resolve_path, mock_settings, 
                                             pipeline_util, temp_output_dir):
        """Test clearing all files in an output directory."""
        # Setup
        mock_settings.get.return_value = str(temp_output_dir)
        mock_resolve_path.return_value = temp_output_dir
        
        # Execute
        deleted_files = pipeline_util.clear_output_directory("test_pipeline")
        
        # Verify
        assert len(deleted_files) == 4  # All 4 files should be deleted
        assert not list(temp_output_dir.glob("**/*.*"))  # No files should remain
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    @mock.patch("fca_dashboard.utils.pipeline_util.resolve_path")
    def test_clear_output_directory_preserve_files(self, mock_resolve_path, mock_settings, 
                                                  pipeline_util, temp_output_dir):
        """Test clearing directory while preserving specific files."""
        # Setup
        mock_settings.get.return_value = str(temp_output_dir)
        mock_resolve_path.return_value = temp_output_dir
        
        # Execute
        deleted_files = pipeline_util.clear_output_directory(
            "test_pipeline", 
            preserve_files=["file1.txt"]
        )
        
        # Verify
        assert len(deleted_files) == 3  # 3 files should be deleted
        assert (temp_output_dir / "file1.txt").exists()  # Preserved file should exist
        assert not (temp_output_dir / "file2.csv").exists()  # Other files should be deleted
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    @mock.patch("fca_dashboard.utils.pipeline_util.resolve_path")
    def test_clear_output_directory_preserve_extensions(self, mock_resolve_path, mock_settings, 
                                                      pipeline_util, temp_output_dir):
        """Test clearing directory while preserving specific file extensions."""
        # Setup
        mock_settings.get.return_value = str(temp_output_dir)
        mock_resolve_path.return_value = temp_output_dir
        
        # Execute
        deleted_files = pipeline_util.clear_output_directory(
            "test_pipeline", 
            preserve_extensions=[".db"]
        )
        
        # Verify
        assert len(deleted_files) == 3  # 3 files should be deleted
        assert (temp_output_dir / "important.db").exists()  # .db file should be preserved
        assert not (temp_output_dir / "file1.txt").exists()  # Other files should be deleted
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    @mock.patch("fca_dashboard.utils.pipeline_util.resolve_path")
    def test_clear_output_directory_dry_run(self, mock_resolve_path, mock_settings, 
                                           pipeline_util, temp_output_dir):
        """Test dry run mode which should not delete any files."""
        # Setup
        mock_settings.get.return_value = str(temp_output_dir)
        mock_resolve_path.return_value = temp_output_dir
        
        # Get initial file count
        initial_files = list(temp_output_dir.glob("**/*.*"))
        
        # Execute
        deleted_files = pipeline_util.clear_output_directory(
            "test_pipeline", 
            dry_run=True
        )
        
        # Verify
        assert len(deleted_files) == 4  # Should report 4 files would be deleted
        assert len(list(temp_output_dir.glob("**/*.*"))) == len(initial_files)  # No files should be deleted


class TestGetPipelineOutputDir:
    """Tests for the get_pipeline_output_dir function."""
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    @mock.patch("fca_dashboard.utils.pipeline_util.resolve_path")
    def test_get_pipeline_output_dir_exists(self, mock_resolve_path, mock_settings, pipeline_util):
        """Test getting an existing pipeline output directory."""
        # Setup
        mock_settings.get.return_value = "/path/to/output"
        mock_resolve_path.return_value = Path("/resolved/path/to/output")
        
        # Execute
        output_dir = pipeline_util.get_pipeline_output_dir("test_pipeline")
        
        # Verify
        assert output_dir == Path("/resolved/path/to/output")
        mock_settings.get.assert_called_with("test_pipeline.output_dir", None)
    
    @mock.patch("fca_dashboard.utils.pipeline_util.settings")
    def test_get_pipeline_output_dir_not_configured(self, mock_settings, pipeline_util):
        """Test handling when pipeline output directory is not configured."""
        # Setup
        mock_settings.get.return_value = None
        
        # Execute and verify
        with pytest.raises(pipeline_util.PipelineUtilError):
            pipeline_util.get_pipeline_output_dir("nonexistent_pipeline")