"""
Unit tests for the base pipeline module.

This module contains tests for the BasePipeline class that provides common
functionality for all data pipelines.
"""

import os
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from fca_dashboard.pipelines import BasePipeline


class TestPipeline(BasePipeline):
    """Test implementation of BasePipeline for testing."""
    
    def __init__(self):
        """Initialize the test pipeline."""
        super().__init__("test_pipeline")
    
    def extract(self):
        """Extract test data."""
        # Create a simple test DataFrame
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Test 1", "Test 2", "Test 3"],
            "value": [10.5, 20.5, 30.5]
        })
    
    def analyze(self, df):
        """Analyze test data."""
        # Store simple analysis results
        self.analysis_results = {
            "unique_values": {
                "name": {
                    "count": 3,
                    "null_count": 0,
                    "null_percentage": 0.0,
                    "values": ["Test 1", "Test 2", "Test 3"],
                    "value_counts": {"Test 1": 1, "Test 2": 1, "Test 3": 1}
                }
            }
        }
        return self.analysis_results
    
    def validate(self, df):
        """Validate test data."""
        # Store simple validation results
        self.validation_results = {
            "missing_values": {
                "id": 0.0,
                "name": 0.0,
                "value": 0.0
            },
            "duplicate_rows": {
                "duplicate_count": 0,
                "duplicate_indices": []
            }
        }
        return self.validation_results
    
    def export(self, df):
        """Export test data."""
        # Return a mock database path
        return os.path.join(self.output_dir, "test_pipeline.db")


@pytest.fixture
def test_pipeline():
    """Create a test pipeline instance."""
    return TestPipeline()


class TestBasePipeline:
    """Tests for the BasePipeline class."""
    
    @mock.patch("fca_dashboard.pipelines.base_pipeline.clear_output_directory")
    def test_clear_output_directory(self, mock_clear_output, test_pipeline):
        """Test clearing the output directory."""
        # Setup
        mock_clear_output.return_value = ["file1.txt", "file2.csv"]
        
        # Execute
        result = test_pipeline.clear_output_directory(preserve_db=True)
        
        # Verify
        assert result == ["file1.txt", "file2.csv"]
        mock_clear_output.assert_called_once_with(
            "test_pipeline",
            preserve_files=None,
            preserve_extensions=[".db"]
        )
    
    @mock.patch("fca_dashboard.pipelines.base_pipeline.get_pipeline_output_dir")
    @mock.patch("fca_dashboard.pipelines.base_pipeline.os.makedirs")
    def test_prepare_output_directory(self, mock_makedirs, mock_get_dir, test_pipeline):
        """Test preparing the output directory."""
        # Setup
        mock_get_dir.return_value = Path("/test/output/dir")
        
        # Execute
        result = test_pipeline.prepare_output_directory()
        
        # Verify
        assert result == Path("/test/output/dir")
        mock_get_dir.assert_called_once_with("test_pipeline")
        mock_makedirs.assert_called_once_with(Path("/test/output/dir"), exist_ok=True)
    
    @mock.patch("fca_dashboard.pipelines.base_pipeline.BasePipeline.prepare_output_directory")
    def test_save_reports(self, mock_prepare_dir, test_pipeline, tmp_path):
        """Test saving reports."""
        # Setup
        mock_prepare_dir.return_value = tmp_path
        df = test_pipeline.extract()
        test_pipeline.analyze(df)
        test_pipeline.validate(df)
        
        # Execute
        report_paths = test_pipeline.save_reports(df)
        
        # Verify
        assert "analysis_report" in report_paths
        assert "validation_report" in report_paths
        assert os.path.exists(report_paths["analysis_report"])
        assert os.path.exists(report_paths["validation_report"])
    
    @mock.patch("fca_dashboard.pipelines.base_pipeline.BasePipeline.clear_output_directory")
    @mock.patch("fca_dashboard.pipelines.base_pipeline.get_pipeline_output_dir")
    @mock.patch("fca_dashboard.pipelines.base_pipeline.os.makedirs")
    def test_run(self, mock_makedirs, mock_get_dir, mock_clear_output, test_pipeline, tmp_path):
        """Test running the pipeline."""
        # Setup
        mock_clear_output.return_value = []
        mock_get_dir.return_value = tmp_path
        
        # Execute with clear_output=True
        result = test_pipeline.run(clear_output=True)
        
        # Verify
        assert result["status"] == "success"
        assert "data" in result
        assert result["data"]["rows"] == 3
        assert result["data"]["columns"] == 3
        mock_clear_output.assert_called_once()
        
        # Reset mocks
        mock_clear_output.reset_mock()
        
        # Execute with clear_output=False
        result = test_pipeline.run(clear_output=False)
        
        # Verify
        assert result["status"] == "success"
        assert "data" in result
        mock_clear_output.assert_not_called()
    
    @mock.patch.object(TestPipeline, "extract")
    def test_run_extract_error(self, mock_extract, test_pipeline):
        """Test running the pipeline with an extraction error."""
        # Setup
        mock_extract.side_effect = ValueError("Test error")
        
        # Execute
        result = test_pipeline.run()
        
        # Verify
        assert result["status"] == "error"
        assert "Error extracting data" in result["message"]
    
    @mock.patch.object(TestPipeline, "analyze")
    def test_run_analyze_error(self, mock_analyze, test_pipeline):
        """Test running the pipeline with an analysis error."""
        # Setup
        mock_analyze.side_effect = ValueError("Test error")
        
        # Execute
        result = test_pipeline.run()
        
        # Verify
        assert result["status"] == "error"
        assert "Error analyzing data" in result["message"]