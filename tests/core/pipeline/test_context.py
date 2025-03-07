"""
Unit tests for the PipelineContext class.
"""

import logging
import time
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from nexusml.core.pipeline.context import PipelineContext


class TestPipelineContext:
    """Tests for the PipelineContext class."""

    def test_initialization(self):
        """Test that the context is initialized correctly."""
        context = PipelineContext()
        assert context.data == {}
        assert context.metrics == {}
        assert context.logs == []
        assert context.start_time is None
        assert context.end_time is None
        assert context.status == "initialized"
        assert context.config == {}

    def test_initialization_with_config(self):
        """Test that the context is initialized correctly with a config."""
        config = {"key": "value"}
        context = PipelineContext(config=config)
        assert context.config == config

    def test_start(self):
        """Test that the start method sets the start time and status."""
        context = PipelineContext()
        context.start()
        assert context.start_time is not None
        assert context.status == "running"

    def test_end(self):
        """Test that the end method sets the end time and status."""
        context = PipelineContext()
        context.start()
        context.end("completed")
        assert context.end_time is not None
        assert context.status == "completed"
        assert "total_execution_time" in context.metrics

    def test_end_without_start(self):
        """Test that the end method works even if start was not called."""
        context = PipelineContext()
        context.end("completed")
        assert context.end_time is not None
        assert context.status == "completed"
        assert "total_execution_time" not in context.metrics

    def test_start_component(self):
        """Test that the start_component method sets the current component."""
        context = PipelineContext()
        context.start_component("test_component")
        assert context._current_component == "test_component"
        assert context._component_start_time is not None

    def test_end_component(self):
        """Test that the end_component method records the execution time."""
        context = PipelineContext()
        context.start_component("test_component")
        time.sleep(0.01)  # Small delay to ensure execution time is measurable
        context.end_component()
        assert "test_component" in context._component_execution_times
        assert context._component_execution_times["test_component"] > 0

    def test_end_component_without_start(self):
        """Test that the end_component method works even if start_component was not called."""
        context = PipelineContext()
        context.end_component()
        assert context._component_execution_times == {}

    def test_get_component_execution_times(self):
        """Test that the get_component_execution_times method returns a copy of the execution times."""
        context = PipelineContext()
        context.start_component("test_component")
        context.end_component()
        execution_times = context.get_component_execution_times()
        assert "test_component" in execution_times
        # Modify the returned dictionary
        execution_times["new_component"] = 1.0
        # Check that the original dictionary is not modified
        assert "new_component" not in context._component_execution_times

    def test_set_and_get(self):
        """Test that the set and get methods work correctly."""
        context = PipelineContext()
        context.set("key", "value")
        assert context.get("key") == "value"
        assert "key" in context._modified_keys
        assert "key" in context._accessed_keys

    def test_get_with_default(self):
        """Test that the get method returns the default value if the key is not found."""
        context = PipelineContext()
        assert context.get("key", "default") == "default"
        assert "key" in context._accessed_keys

    def test_has(self):
        """Test that the has method returns True if the key exists."""
        context = PipelineContext()
        context.set("key", "value")
        assert context.has("key") is True
        assert context.has("nonexistent") is False

    def test_add_metric(self):
        """Test that the add_metric method adds a metric to the metrics collection."""
        context = PipelineContext()
        context.add_metric("key", "value")
        assert context.metrics["key"] == "value"

    def test_get_metrics(self):
        """Test that the get_metrics method returns a copy of the metrics."""
        context = PipelineContext()
        context.add_metric("key", "value")
        metrics = context.get_metrics()
        assert metrics["key"] == "value"
        # Modify the returned dictionary
        metrics["new_key"] = "new_value"
        # Check that the original dictionary is not modified
        assert "new_key" not in context.metrics

    def test_log(self):
        """Test that the log method adds a log entry to the logs collection."""
        context = PipelineContext()
        context.log("INFO", "Test message")
        assert len(context.logs) == 1
        assert context.logs[0]["level"] == "INFO"
        assert context.logs[0]["message"] == "Test message"

    def test_get_logs(self):
        """Test that the get_logs method returns a copy of the logs."""
        context = PipelineContext()
        context.log("INFO", "Test message")
        logs = context.get_logs()
        assert len(logs) == 1
        assert logs[0]["level"] == "INFO"
        assert logs[0]["message"] == "Test message"
        # Modify the returned list
        logs.append({"level": "ERROR", "message": "New message"})
        # Check that the original list is not modified
        assert len(context.logs) == 1

    def test_get_execution_summary(self):
        """Test that the get_execution_summary method returns a summary of the execution."""
        context = PipelineContext()
        context.start()
        context.set("key", "value")
        context.add_metric("metric_key", "metric_value")
        context.start_component("test_component")
        context.end_component()
        context.end("completed")
        summary = context.get_execution_summary()
        assert summary["status"] == "completed"
        assert summary["metrics"]["metric_key"] == "metric_value"
        assert "test_component" in summary["component_execution_times"]
        assert "key" in summary["modified_keys"]
        assert "key" in summary["accessed_keys"]
        assert "start_time" in summary
        assert "end_time" in summary
        assert "total_execution_time" in summary

    @pytest.fixture
    def temp_csv_file(self, tmp_path):
        """Create a temporary CSV file for testing."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        file_path = tmp_path / "test.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def test_save_data(self, temp_csv_file, tmp_path):
        """Test that the save_data method saves data to a file."""
        context = PipelineContext()
        df = pd.read_csv(temp_csv_file)
        output_path = tmp_path / "output.csv"
        context.save_data("data_path", df, output_path)
        assert output_path.exists()
        assert context.get("data_path") == str(output_path)
        # Check that the saved data is correct
        saved_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df, saved_df)

    def test_load_data_csv(self, temp_csv_file):
        """Test that the load_data method loads data from a CSV file."""
        context = PipelineContext()
        df = context.load_data(temp_csv_file)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["col1", "col2"]

    def test_load_data_nonexistent_file(self):
        """Test that the load_data method raises an error for nonexistent files."""
        context = PipelineContext()
        with pytest.raises(FileNotFoundError):
            context.load_data("nonexistent.csv")

    def test_load_data_unsupported_format(self, tmp_path):
        """Test that the load_data method raises an error for unsupported file formats."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("test")
        context = PipelineContext()
        with pytest.raises(ValueError):
            context.load_data(file_path)

    def test_logging_integration(self):
        """Test that the context integrates with the logging system."""
        logger = mock.MagicMock(spec=logging.Logger)
        context = PipelineContext(logger=logger)
        context.log("info", "Test message")
        logger.info.assert_called_once_with("Test message")
