#!/usr/bin/env python
"""
Tests for the Train Model Pipeline v2

This module contains tests for the train_model_pipeline_v2.py module.
"""

import os
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.cli.training_args import TrainingArguments
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.train_model_pipeline_v2 import (
    create_orchestrator,
    main,
    make_sample_prediction_with_orchestrator,
    train_with_orchestrator,
)


class TestCreateOrchestrator:
    """Tests for the create_orchestrator function."""

    def test_create_orchestrator(self):
        """Test that create_orchestrator returns a PipelineOrchestrator."""
        # Create a mock logger
        logger = mock.MagicMock()

        # Call create_orchestrator
        orchestrator = create_orchestrator(logger)

        # Check that create_orchestrator returns a PipelineOrchestrator
        assert isinstance(orchestrator, PipelineOrchestrator)
        assert orchestrator.factory is not None
        assert orchestrator.context is not None
        assert orchestrator.logger is logger


class TestTrainWithOrchestrator:
    """Tests for the train_with_orchestrator function."""

    @mock.patch("nexusml.train_model_pipeline_v2.create_orchestrator")
    def test_train_with_orchestrator(self, mock_create_orchestrator):
        """Test that train_with_orchestrator calls the orchestrator correctly."""
        # Create a mock logger
        logger = mock.MagicMock()

        # Create a mock orchestrator
        mock_orchestrator = mock.MagicMock()
        mock_create_orchestrator.return_value = mock_orchestrator

        # Create a mock model and metrics
        mock_model = mock.MagicMock(spec=Pipeline)
        mock_metrics = {"accuracy": 0.9, "f1": 0.85}
        mock_orchestrator.train_model.return_value = (mock_model, mock_metrics)

        # Set up the mock model with steps attribute
        mock_model.steps = [("classifier", mock.MagicMock(spec=RandomForestClassifier))]
        classifier = mock_model.steps[0][1]
        classifier.classes_ = np.array(["HVAC", "Plumbing"])
        classifier.n_classes_ = 2
        classifier.n_features_in_ = 1
        classifier.n_outputs_ = 1

        # Add predict method to the mock model that returns the correct shape
        def mock_predict(X):
            # Return a 2D array with the correct number of columns
            return np.array(
                [["HVAC", "D3010", "Mechanical", "AHU", "Cooling"]] * len(X)
            )

        mock_model.predict = mock_predict
        classifier.predict = mock_predict

        # Create dummy estimators for RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier

        dummy_estimator = mock.MagicMock(spec=DecisionTreeClassifier)
        dummy_estimator.tree_ = mock.MagicMock()
        dummy_estimator.classes_ = classifier.classes_
        dummy_estimator.n_outputs_ = 1
        dummy_estimator.n_classes_ = 2
        dummy_estimator.n_features_in_ = 1
        classifier.estimators_ = [dummy_estimator]

        # Create a mock execution summary
        mock_summary = {
            "status": "success",
            "component_execution_times": {
                "data_loader": 0.1,
                "preprocessor": 0.2,
                "feature_engineer": 0.3,
                "model_builder": 0.4,
                "model_trainer": 0.5,
                "model_evaluator": 0.6,
            },
            "total_execution_time": 2.1,
        }
        mock_orchestrator.get_execution_summary.return_value = mock_summary

        # Create a mock context with engineered_data
        mock_orchestrator.context.get.return_value = pd.DataFrame(
            {
                "combined_text": ["text1", "text2"],
                "service_life": [10, 20],
                "category_name": ["HVAC", "Plumbing"],
                "mcaa_system_category": ["Mechanical", "Plumbing"],
                "Equipment_Type": ["AHU", "Pump"],
                "System_Subtype": ["Cooling", "Water"],
                "uniformat_code": ["D3010", "D2010"],
            }
        )

        # Create a temporary file for data_path
        with tempfile.NamedTemporaryFile() as temp_file:
            # Create training arguments
            args = TrainingArguments(
                data_path=temp_file.name,
                visualize=True,
            )

            # Call train_with_orchestrator
            model, metrics, viz_paths = train_with_orchestrator(args, logger)

            # Check that create_orchestrator was called
            mock_create_orchestrator.assert_called_once_with(logger)

            # Check that orchestrator.train_model was called with the correct arguments
            mock_orchestrator.train_model.assert_called_once_with(
                data_path=args.data_path,
                feature_config_path=args.feature_config_path,
                test_size=args.test_size,
                random_state=args.random_state,
                optimize_hyperparameters=args.optimize_hyperparameters,
                output_dir=args.output_dir,
                model_name=args.model_name,
            )

            # Check that orchestrator.get_execution_summary was called
            mock_orchestrator.get_execution_summary.assert_called_once()

            # Check that make_sample_prediction_with_orchestrator was called
            # (indirectly through the mock)
            assert mock_orchestrator.predict.call_count > 0

            # Check that the function returns the expected values
            assert model is mock_model
            assert metrics is mock_metrics
            assert viz_paths is not None


class TestMakeSamplePredictionWithOrchestrator:
    """Tests for the make_sample_prediction_with_orchestrator function."""

    def test_make_sample_prediction_with_orchestrator(self):
        """Test that make_sample_prediction_with_orchestrator calls the orchestrator correctly."""
        # Create a mock logger
        logger = mock.MagicMock()

        # Create a mock orchestrator
        mock_orchestrator = mock.MagicMock()

        # Create a mock model
        mock_model = mock.MagicMock(spec=Pipeline)

        # Create a mock prediction result
        mock_predictions = pd.DataFrame(
            {
                "category_name": ["HVAC"],
                "uniformat_code": ["D3010"],
                "mcaa_system_category": ["Mechanical"],
                "Equipment_Type": ["Air Handling"],
                "System_Subtype": ["Cooling"],
            }
        )
        mock_orchestrator.predict.return_value = mock_predictions

        # Call make_sample_prediction_with_orchestrator
        result = make_sample_prediction_with_orchestrator(
            mock_orchestrator, mock_model, logger
        )

        # Check that orchestrator.predict was called with the correct arguments
        mock_orchestrator.predict.assert_called_once()
        args, kwargs = mock_orchestrator.predict.call_args
        assert kwargs["model"] is mock_model
        assert isinstance(kwargs["data"], pd.DataFrame)
        assert "equipment_tag" in kwargs["data"].columns
        assert "description" in kwargs["data"].columns
        assert "service_life" in kwargs["data"].columns

        # Check that the function returns the expected values
        assert isinstance(result, dict)
        assert "category_name" in result
        assert result["category_name"] == "HVAC"


class TestMain:
    """Tests for the main function."""

    @mock.patch("nexusml.train_model_pipeline_v2.parse_args")
    @mock.patch("nexusml.train_model_pipeline_v2.setup_logging")
    @mock.patch("nexusml.train_model_pipeline_v2.load_reference_data")
    @mock.patch("nexusml.train_model_pipeline_v2.validate_data")
    @mock.patch("nexusml.train_model_pipeline_v2.train_with_orchestrator")
    @mock.patch("nexusml.train_model_pipeline_v2.train_model")
    @mock.patch("nexusml.train_model_pipeline_v2.save_model")
    @mock.patch("nexusml.train_model_pipeline_v2.generate_visualizations")
    @mock.patch("nexusml.train_model_pipeline_v2.make_sample_prediction")
    def test_main_with_orchestrator(
        self,
        mock_make_sample_prediction,
        mock_generate_visualizations,
        mock_save_model,
        mock_train_model,
        mock_train_with_orchestrator,
        mock_validate_data,
        mock_load_reference_data,
        mock_setup_logging,
        mock_parse_args,
    ):
        """Test that main calls the correct functions when use_orchestrator is True."""
        # Create a mock logger
        mock_logger = mock.MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Create a mock args
        mock_args = mock.MagicMock()
        mock_args.use_orchestrator = True
        mock_args.visualize = True
        mock_args.to_dict.return_value = {"use_orchestrator": True, "visualize": True}
        mock_parse_args.return_value = mock_args

        # Create a mock validation result
        mock_validation_results = {"valid": True}
        mock_validate_data.return_value = mock_validation_results

        # Create a mock model, metrics, and viz_paths
        mock_model = mock.MagicMock(spec=Pipeline)
        mock_metrics = {"accuracy": 0.9, "f1": 0.85}
        mock_viz_paths = {
            "equipment_category_distribution": "path/to/equipment_category_distribution.png",
            "system_type_distribution": "path/to/system_type_distribution.png",
        }
        mock_train_with_orchestrator.return_value = (
            mock_model,
            mock_metrics,
            mock_viz_paths,
        )

        # Call main
        main()

        # Check that parse_args was called
        mock_parse_args.assert_called_once()

        # Check that setup_logging was called with the correct arguments
        mock_setup_logging.assert_called_once_with(mock_args.log_level)

        # Check that load_reference_data was called with the correct arguments
        mock_load_reference_data.assert_called_once_with(
            mock_args.reference_config_path, mock_logger
        )

        # Check that validate_data was called with the correct arguments
        mock_validate_data.assert_called_once_with(mock_args.data_path, mock_logger)

        # Check that train_with_orchestrator was called with the correct arguments
        mock_train_with_orchestrator.assert_called_once_with(mock_args, mock_logger)

        # Check that train_model was not called
        mock_train_model.assert_not_called()

        # Check that save_model was not called
        mock_save_model.assert_not_called()

        # Check that generate_visualizations was not called
        mock_generate_visualizations.assert_not_called()

        # Check that make_sample_prediction was not called
        mock_make_sample_prediction.assert_not_called()

    @mock.patch("nexusml.train_model_pipeline_v2.parse_args")
    @mock.patch("nexusml.train_model_pipeline_v2.setup_logging")
    @mock.patch("nexusml.train_model_pipeline_v2.load_reference_data")
    @mock.patch("nexusml.train_model_pipeline_v2.validate_data")
    @mock.patch("nexusml.train_model_pipeline_v2.train_with_orchestrator")
    @mock.patch("nexusml.train_model_pipeline_v2.train_model")
    @mock.patch("nexusml.train_model_pipeline_v2.save_model")
    @mock.patch("nexusml.train_model_pipeline_v2.generate_visualizations")
    @mock.patch("nexusml.train_model_pipeline_v2.make_sample_prediction")
    def test_main_with_legacy(
        self,
        mock_make_sample_prediction,
        mock_generate_visualizations,
        mock_save_model,
        mock_train_model,
        mock_train_with_orchestrator,
        mock_validate_data,
        mock_load_reference_data,
        mock_setup_logging,
        mock_parse_args,
    ):
        """Test that main calls the correct functions when use_orchestrator is False."""
        # Create a mock logger
        mock_logger = mock.MagicMock()
        mock_setup_logging.return_value = mock_logger

        # Create a mock args
        mock_args = mock.MagicMock()
        mock_args.use_orchestrator = False
        mock_args.visualize = True
        mock_args.to_dict.return_value = {"use_orchestrator": False, "visualize": True}
        mock_parse_args.return_value = mock_args

        # Create a mock validation result
        mock_validation_results = {"valid": True}
        mock_validate_data.return_value = mock_validation_results

        # Create a mock classifier, df, and metrics
        mock_classifier = mock.MagicMock()
        mock_classifier.model = mock.MagicMock(spec=Pipeline)
        mock_df = mock.MagicMock()
        mock_metrics = {"accuracy": 0.9, "f1": 0.85}
        mock_train_model.return_value = (mock_classifier, mock_df, mock_metrics)

        # Create a mock save_paths
        mock_save_paths = {
            "model_path": "path/to/model.pkl",
            "metadata_path": "path/to/metadata.json",
        }
        mock_save_model.return_value = mock_save_paths

        # Create a mock viz_paths
        mock_viz_paths = {
            "equipment_category_distribution": "path/to/equipment_category_distribution.png",
            "system_type_distribution": "path/to/system_type_distribution.png",
        }
        mock_generate_visualizations.return_value = mock_viz_paths

        # Call main
        main()

        # Check that parse_args was called
        mock_parse_args.assert_called_once()

        # Check that setup_logging was called with the correct arguments
        mock_setup_logging.assert_called_once_with(mock_args.log_level)

        # Check that load_reference_data was called with the correct arguments
        mock_load_reference_data.assert_called_once_with(
            mock_args.reference_config_path, mock_logger
        )

        # Check that validate_data was called with the correct arguments
        mock_validate_data.assert_called_once_with(mock_args.data_path, mock_logger)

        # Check that train_with_orchestrator was not called
        mock_train_with_orchestrator.assert_not_called()

        # Check that train_model was called with the correct arguments
        mock_train_model.assert_called_once_with(
            data_path=mock_args.data_path,
            feature_config_path=mock_args.feature_config_path,
            sampling_strategy=mock_args.sampling_strategy,
            test_size=mock_args.test_size,
            random_state=mock_args.random_state,
            optimize_params=mock_args.optimize_hyperparameters,
            logger=mock_logger,
        )

        # Check that save_model was called with the correct arguments
        mock_save_model.assert_called_once_with(
            mock_classifier,
            mock_args.output_dir,
            mock_args.model_name,
            mock_metrics,
            mock_logger,
        )

        # Check that generate_visualizations was called with the correct arguments
        mock_generate_visualizations.assert_called_once_with(
            mock_classifier,
            mock_df,
            mock_args.output_dir,
            mock_logger,
        )

        # Check that make_sample_prediction was called with the correct arguments
        mock_make_sample_prediction.assert_called_once_with(
            mock_classifier, logger=mock_logger
        )
