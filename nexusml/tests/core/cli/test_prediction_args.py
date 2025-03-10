"""
Tests for the prediction pipeline argument parsing module.
"""

import argparse
import logging
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest

from nexusml.core.cli.prediction_args import PredictionArgumentParser


class TestPredictionArgumentParser(unittest.TestCase):
    """Test cases for the PredictionArgumentParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = PredictionArgumentParser()

        # Create temporary test files
        self.test_model_path = "test_model.pkl"
        self.test_input_file = "test_input.csv"
        self.test_feature_config = "test_feature_config.yml"

        # Create empty files for testing
        Path(self.test_model_path).touch()
        Path(self.test_input_file).touch()
        Path(self.test_feature_config).touch()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary test files
        for file_path in [self.test_model_path, self.test_input_file, self.test_feature_config]:
            if Path(file_path).exists():
                Path(file_path).unlink()

    def test_default_arguments(self):
        """Test parsing with default arguments."""
        args = self.parser.parse_args([f"--input-file={self.test_input_file}"])

        self.assertEqual(args.model_path, "nexusml/output/models/equipment_classifier_latest.pkl")
        self.assertEqual(args.input_file, self.test_input_file)
        self.assertEqual(args.output_file, "prediction_results.csv")
        self.assertEqual(args.log_level, "INFO")
        self.assertEqual(args.description_column, "Description")
        self.assertEqual(args.service_life_column, "Service Life")
        self.assertEqual(args.asset_tag_column, "Asset Tag")
        self.assertIsNone(args.feature_config_path)
        self.assertFalse(args.use_orchestrator)

    def test_custom_arguments(self):
        """Test parsing with custom arguments."""
        args = self.parser.parse_args(
            [
                f"--model-path={self.test_model_path}",
                f"--input-file={self.test_input_file}",
                "--output-file=custom_output.csv",
                "--log-level=DEBUG",
                "--description-column=CustomDesc",
                "--service-life-column=CustomLife",
                "--asset-tag-column=CustomTag",
                f"--feature-config-path={self.test_feature_config}",
                "--use-orchestrator",
            ]
        )

        self.assertEqual(args.model_path, self.test_model_path)
        self.assertEqual(args.input_file, self.test_input_file)
        self.assertEqual(args.output_file, "custom_output.csv")
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(args.description_column, "CustomDesc")
        self.assertEqual(args.service_life_column, "CustomLife")
        self.assertEqual(args.asset_tag_column, "CustomTag")
        self.assertEqual(args.feature_config_path, self.test_feature_config)
        self.assertTrue(args.use_orchestrator)

    def test_required_arguments(self):
        """Test that required arguments are enforced."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])

    def test_parse_args_to_dict(self):
        """Test conversion of parsed args to dictionary."""
        args_dict = self.parser.parse_args_to_dict([f"--input-file={self.test_input_file}"])

        self.assertIsInstance(args_dict, dict)
        self.assertEqual(args_dict["input_file"], self.test_input_file)
        self.assertEqual(args_dict["model_path"], "nexusml/output/models/equipment_classifier_latest.pkl")

    def test_validate_args_valid(self):
        """Test validation with valid arguments."""
        args = self.parser.parse_args(
            [
                f"--model-path={self.test_model_path}",
                f"--input-file={self.test_input_file}",
                f"--feature-config-path={self.test_feature_config}",
            ]
        )

        # This should not raise an exception
        self.parser.validate_args(args)

    def test_validate_args_invalid_model_path(self):
        """Test validation with invalid model path."""
        args = self.parser.parse_args(["--model-path=nonexistent_model.pkl", f"--input-file={self.test_input_file}"])

        with self.assertRaises(ValueError) as context:
            self.parser.validate_args(args)

        self.assertIn("Model file not found", str(context.exception))

    def test_validate_args_invalid_input_file(self):
        """Test validation with invalid input file."""
        args = self.parser.parse_args([f"--model-path={self.test_model_path}", "--input-file=nonexistent_input.csv"])

        with self.assertRaises(ValueError) as context:
            self.parser.validate_args(args)

        self.assertIn("Input file not found", str(context.exception))

    def test_validate_args_invalid_feature_config(self):
        """Test validation with invalid feature config path."""
        args = self.parser.parse_args(
            [
                f"--model-path={self.test_model_path}",
                f"--input-file={self.test_input_file}",
                "--feature-config-path=nonexistent_config.yml",
            ]
        )

        with self.assertRaises(ValueError) as context:
            self.parser.validate_args(args)

        self.assertIn("Feature config file not found", str(context.exception))

    def test_validate_args_invalid_log_level(self):
        """Test validation with invalid log level."""
        args = self.parser.parse_args([f"--model-path={self.test_model_path}", f"--input-file={self.test_input_file}"])

        # Modify the log level to an invalid value
        args.log_level = "INVALID_LEVEL"

        with self.assertRaises(ValueError) as context:
            self.parser.validate_args(args)

        self.assertIn("Invalid log level", str(context.exception))

    @mock.patch("logging.FileHandler")
    @mock.patch("logging.StreamHandler")
    @mock.patch("logging.basicConfig")
    def test_setup_logging(self, mock_basic_config, mock_stream_handler, mock_file_handler):
        """Test logging setup."""
        args = self.parser.parse_args([f"--input-file={self.test_input_file}", "--log-level=DEBUG"])

        logger = self.parser.setup_logging(args)

        # Check that the logger was created
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "equipment_prediction")

        # Check that basicConfig was called with the right level
        mock_basic_config.assert_called_once()
        call_args = mock_basic_config.call_args[1]
        self.assertEqual(call_args["level"], logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
