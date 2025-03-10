#!/usr/bin/env python
"""
Tests for the Training Arguments Module

This module contains tests for the TrainingArguments class and related utilities.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from nexusml.core.cli.training_args import TrainingArguments, parse_args, setup_logging


class TestTrainingArguments:
    """Tests for the TrainingArguments class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            args = TrainingArguments(data_path=temp_file.name)

            # Check default values
            assert args.test_size == 0.3
            assert args.random_state == 42
            assert args.sampling_strategy == "direct"
            assert args.optimize_hyperparameters is False
            assert args.output_dir == "nexusml/output/models"
            assert args.model_name == "equipment_classifier"
            assert args.log_level == "INFO"
            assert args.visualize is False
            assert args.use_orchestrator is True

    def test_validation_data_path(self):
        """Test validation of data_path."""
        # Test with non-existent data_path
        with pytest.raises(ValueError, match="Data path does not exist"):
            TrainingArguments(data_path="non_existent_file.csv")

    def test_validation_feature_config_path(self):
        """Test validation of feature_config_path."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Test with non-existent feature_config_path
            with pytest.raises(ValueError, match="Feature config path does not exist"):
                TrainingArguments(
                    data_path=temp_file.name,
                    feature_config_path="non_existent_file.yml",
                )

    def test_validation_reference_config_path(self):
        """Test validation of reference_config_path."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Test with non-existent reference_config_path
            with pytest.raises(ValueError, match="Reference config path does not exist"):
                TrainingArguments(
                    data_path=temp_file.name,
                    reference_config_path="non_existent_file.yml",
                )

    def test_validation_test_size(self):
        """Test validation of test_size."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Test with invalid test_size (too small)
            with pytest.raises(ValueError, match="Test size must be between 0 and 1"):
                TrainingArguments(data_path=temp_file.name, test_size=0)

            # Test with invalid test_size (too large)
            with pytest.raises(ValueError, match="Test size must be between 0 and 1"):
                TrainingArguments(data_path=temp_file.name, test_size=1)

    def test_validation_sampling_strategy(self):
        """Test validation of sampling_strategy."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Test with invalid sampling_strategy
            with pytest.raises(ValueError, match="Sampling strategy must be one of"):
                TrainingArguments(data_path=temp_file.name, sampling_strategy="invalid_strategy")

    def test_validation_log_level(self):
        """Test validation of log_level."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Test with invalid log_level
            with pytest.raises(ValueError, match="Log level must be one of"):
                TrainingArguments(data_path=temp_file.name, log_level="INVALID_LEVEL")

    def test_output_dir_creation(self):
        """Test that output_dir is created if it doesn't exist."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Create a temporary directory for output_dir
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = os.path.join(temp_dir, "new_output_dir")

                # Verify that output_dir doesn't exist yet
                assert not os.path.exists(output_dir)

                # Create TrainingArguments with the new output_dir
                args = TrainingArguments(data_path=temp_file.name, output_dir=output_dir)

                # Verify that output_dir was created
                assert os.path.exists(output_dir)
                assert os.path.isdir(output_dir)

    def test_to_dict(self):
        """Test the to_dict method."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            args = TrainingArguments(
                data_path=temp_file.name,
                test_size=0.2,
                random_state=123,
                optimize_hyperparameters=True,
            )

            # Convert to dictionary
            args_dict = args.to_dict()

            # Check that the dictionary contains the expected values
            assert args_dict["data_path"] == temp_file.name
            assert args_dict["test_size"] == 0.2
            assert args_dict["random_state"] == 123
            assert args_dict["optimize_hyperparameters"] is True


class TestParseArgs:
    """Tests for the parse_args function."""

    @mock.patch("argparse.ArgumentParser.parse_args")
    def test_parse_args(self, mock_parse_args):
        """Test that parse_args returns a TrainingArguments object."""
        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Mock the return value of argparse.ArgumentParser.parse_args
            mock_args = mock.Mock()
            mock_args.data_path = temp_file.name
            mock_args.feature_config_path = None
            mock_args.reference_config_path = None
            mock_args.test_size = 0.3
            mock_args.random_state = 42
            mock_args.sampling_strategy = "direct"
            mock_args.optimize_hyperparameters = False
            mock_args.output_dir = "nexusml/output/models"
            mock_args.model_name = "equipment_classifier"
            mock_args.log_level = "INFO"
            mock_args.visualize = False
            mock_args.use_orchestrator = True
            mock_parse_args.return_value = mock_args

            # Call parse_args
            args = parse_args()

            # Check that parse_args returns a TrainingArguments object
            assert isinstance(args, TrainingArguments)
            assert args.data_path == temp_file.name
            assert args.test_size == 0.3
            assert args.random_state == 42
            assert args.sampling_strategy == "direct"
            assert args.optimize_hyperparameters is False
            assert args.output_dir == "nexusml/output/models"
            assert args.model_name == "equipment_classifier"
            assert args.log_level == "INFO"
            assert args.visualize is False
            assert args.use_orchestrator is True


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_setup_logging(self):
        """Test that setup_logging returns a logger."""
        # Call setup_logging
        logger = setup_logging(log_level="INFO")

        # Check that setup_logging returns a logger
        assert logger is not None
        assert logger.name == "model_training"
        assert logger.level == 20  # INFO level

        # Check that the logger has handlers
        assert len(logger.handlers) > 0

        # Check that the log directory was created
        assert os.path.exists("logs")
        assert os.path.isdir("logs")

    def test_setup_logging_with_level(self):
        """Test that setup_logging sets the correct log level."""
        # Call setup_logging with DEBUG level
        logger = setup_logging(log_level="DEBUG")

        # Check that the logger has the correct level
        assert logger.level == 10  # DEBUG level

        # Call setup_logging with WARNING level
        logger = setup_logging(log_level="WARNING")

        # Check that the logger has the correct level
        assert logger.level == 30  # WARNING level
