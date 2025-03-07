"""
Tests for the updated prediction pipeline entry point.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from nexusml.predict_v2 import create_orchestrator, main, run_legacy_prediction, run_orchestrator_prediction


class TestPredictV2(unittest.TestCase):
    """Test cases for the updated prediction pipeline entry point."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.test_model_path = "test_model.pkl"
        self.test_input_file = "test_input.csv"
        self.test_output_file = "test_output.csv"
        self.test_feature_config = "test_feature_config.yml"

        # Create empty files for testing
        Path(self.test_model_path).touch()

        # Create a simple test input CSV
        test_data = pd.DataFrame(
            {
                "equipment_tag": ["AHU-01", "CHW-01"],
                "manufacturer": ["Trane", "Carrier"],
                "model": ["M-1000", "C-2000"],
                "description": ["Air Handling Unit", "Chiller"],
            }
        )
        test_data.to_csv(self.test_input_file, index=False)

        # Create a dummy feature config file
        Path(self.test_feature_config).touch()

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary test files
        for file_path in [self.test_model_path, self.test_input_file, self.test_output_file, self.test_feature_config]:
            if Path(file_path).exists():
                Path(file_path).unlink()

    @mock.patch("nexusml.predict_v2.PredictionArgumentParser")
    @mock.patch("nexusml.predict_v2.run_legacy_prediction")
    @mock.patch("nexusml.predict_v2.run_orchestrator_prediction")
    def test_main_legacy_mode(self, mock_run_orchestrator, mock_run_legacy, mock_parser_class):
        """Test main function with legacy mode."""
        # Set up mock parser
        mock_parser = mock.MagicMock()
        mock_parser_class.return_value = mock_parser

        # Set up mock args
        mock_args = mock.MagicMock()
        mock_args.use_orchestrator = False
        mock_parser.parse_args.return_value = mock_args

        # Set up mock logger
        mock_logger = mock.MagicMock()
        mock_parser.setup_logging.return_value = mock_logger

        # Call main function
        main()

        # Verify that the legacy prediction function was called
        mock_run_legacy.assert_called_once_with(mock_args, mock_logger)
        mock_run_orchestrator.assert_not_called()

    @mock.patch("nexusml.predict_v2.PredictionArgumentParser")
    @mock.patch("nexusml.predict_v2.run_legacy_prediction")
    @mock.patch("nexusml.predict_v2.run_orchestrator_prediction")
    def test_main_orchestrator_mode(self, mock_run_orchestrator, mock_run_legacy, mock_parser_class):
        """Test main function with orchestrator mode."""
        # Set up mock parser
        mock_parser = mock.MagicMock()
        mock_parser_class.return_value = mock_parser

        # Set up mock args
        mock_args = mock.MagicMock()
        mock_args.use_orchestrator = True
        mock_parser.parse_args.return_value = mock_args

        # Set up mock logger
        mock_logger = mock.MagicMock()
        mock_parser.setup_logging.return_value = mock_logger

        # Call main function
        main()

        # Verify that the orchestrator prediction function was called
        mock_run_orchestrator.assert_called_once_with(mock_args, mock_logger)
        mock_run_legacy.assert_not_called()

    def test_run_legacy_prediction(self):
        """Test legacy prediction function."""
        # Set up mock args
        mock_args = mock.MagicMock()
        mock_args.model_path = self.test_model_path
        mock_args.input_file = self.test_input_file
        mock_args.output_file = self.test_output_file
        mock_args.description_column = "Description"
        mock_args.service_life_column = "Service Life"
        mock_args.asset_tag_column = "Asset Tag"

        # Set up mock logger
        mock_logger = mock.MagicMock()

        # Call legacy prediction function with mocked dependencies
        with mock.patch("nexusml.core.model.EquipmentClassifier") as mock_classifier_class:
            # Set up mock classifier
            mock_classifier = mock.MagicMock()
            mock_classifier_class.return_value = mock_classifier

            # Set up mock prediction result
            mock_prediction = {
                "category_name": "HVAC",
                "mcaa_system_category": "Mechanical",
                "Equipment_Type": "Air Handling",
                "System_Subtype": "Cooling",
            }
            mock_classifier.predict_from_row.return_value = mock_prediction

            with mock.patch(
                "nexusml.core.data_mapper.map_staging_to_model_input", return_value=pd.read_csv(self.test_input_file)
            ):
                with mock.patch(
                    "nexusml.core.feature_engineering.GenericFeatureEngineer"
                ) as mock_feature_engineer_class:
                    mock_feature_engineer = mock.MagicMock()
                    mock_feature_engineer_class.return_value = mock_feature_engineer
                    mock_feature_engineer.transform.return_value = pd.read_csv(self.test_input_file)

                    run_legacy_prediction(mock_args, mock_logger)

        # Verify that the model was loaded
        mock_classifier.load_model.assert_called_once_with(self.test_model_path)

        # Verify that predictions were made
        assert mock_classifier.predict_from_row.call_count > 0

        # Verify that the output file was created
        assert Path(self.test_output_file).exists()

    @mock.patch("nexusml.predict_v2.create_orchestrator")
    def test_run_orchestrator_prediction(self, mock_create_orchestrator):
        """Test orchestrator prediction function."""
        # Set up mock orchestrator
        mock_orchestrator = mock.MagicMock()
        mock_create_orchestrator.return_value = mock_orchestrator

        # Set up mock prediction result
        mock_predictions = pd.DataFrame(
            {
                "category_name": ["HVAC", "HVAC"],
                "mcaa_system_category": ["Mechanical", "Mechanical"],
                "Equipment_Type": ["Air Handling", "Chiller"],
                "System_Subtype": ["Cooling", "Cooling"],
            }
        )
        mock_orchestrator.predict.return_value = mock_predictions

        # Set up mock execution summary
        mock_summary = {
            "status": "completed",
            "component_execution_times": {
                "data_loading": 0.1,
                "data_preprocessing": 0.2,
                "feature_engineering": 0.3,
                "prediction": 0.4,
            },
            "total_execution_time": 1.0,
        }
        mock_orchestrator.get_execution_summary.return_value = mock_summary

        # Set up mock args
        mock_args = mock.MagicMock()
        mock_args.model_path = self.test_model_path
        mock_args.input_file = self.test_input_file
        mock_args.output_file = self.test_output_file
        mock_args.feature_config_path = self.test_feature_config
        mock_args.description_column = "Description"
        mock_args.service_life_column = "Service Life"
        mock_args.asset_tag_column = "Asset Tag"

        # Set up mock logger
        mock_logger = mock.MagicMock()

        # Call orchestrator prediction function
        run_orchestrator_prediction(mock_args, mock_logger)

        # Verify that the orchestrator was created
        mock_create_orchestrator.assert_called_once_with(mock_logger)

        # Verify that predictions were made
        mock_orchestrator.predict.assert_called_once_with(
            model_path=self.test_model_path,
            data=mock.ANY,  # Can't directly compare DataFrames in assert_called_once_with
            output_path=self.test_output_file,
            feature_config_path=self.test_feature_config,
            description_column="Description",
            service_life_column="Service Life",
            asset_tag_column="Asset Tag",
        )

        # Verify that the execution summary was retrieved
        mock_orchestrator.get_execution_summary.assert_called_once()

    @mock.patch("nexusml.predict_v2.ComponentRegistry")
    @mock.patch("nexusml.predict_v2.DIContainer")
    @mock.patch("nexusml.predict_v2.PipelineFactory")
    @mock.patch("nexusml.predict_v2.PipelineContext")
    @mock.patch("nexusml.predict_v2.PipelineOrchestrator")
    def test_create_orchestrator(
        self, mock_orchestrator_class, mock_context_class, mock_factory_class, mock_container_class, mock_registry_class
    ):
        """Test orchestrator creation."""
        # Set up mocks
        mock_registry = mock.MagicMock()
        mock_registry_class.return_value = mock_registry

        mock_container = mock.MagicMock()
        mock_container_class.return_value = mock_container

        mock_factory = mock.MagicMock()
        mock_factory_class.return_value = mock_factory

        mock_context = mock.MagicMock()
        mock_context_class.return_value = mock_context

        mock_orchestrator = mock.MagicMock()
        mock_orchestrator_class.return_value = mock_orchestrator

        # Set up mock logger
        mock_logger = mock.MagicMock()

        # Call create_orchestrator function
        result = create_orchestrator(mock_logger)

        # Verify that the orchestrator was created with the right components
        mock_registry_class.assert_called_once()
        mock_container_class.assert_called_once()
        mock_factory_class.assert_called_once_with(mock_registry, mock_container)
        mock_context_class.assert_called_once()
        mock_orchestrator_class.assert_called_once_with(mock_factory, mock_context, mock_logger)

        # Verify that the function returned the orchestrator
        self.assertEqual(result, mock_orchestrator)


if __name__ == "__main__":
    unittest.main()
