"""
Pipeline Orchestrator Module

This module provides the PipelineOrchestrator class, which is responsible for
coordinating the execution of pipeline components, handling errors consistently,
and providing comprehensive logging.
"""

import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.core.pipeline.registry import ComponentRegistry


class PipelineOrchestratorError(Exception):
    """Exception raised for errors in the PipelineOrchestrator."""

    pass


class PipelineOrchestrator:
    """
    Orchestrator for pipeline execution.

    The PipelineOrchestrator class coordinates the execution of pipeline components,
    handles errors consistently, and provides comprehensive logging. It uses the
    PipelineFactory to create components and the PipelineContext to manage state
    during execution.

    Attributes:
        factory: Factory for creating pipeline components.
        context: Context for managing state during pipeline execution.
        logger: Logger instance for logging messages.
    """

    def __init__(
        self,
        factory: PipelineFactory,
        context: Optional[PipelineContext] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize a new PipelineOrchestrator.

        Args:
            factory: Factory for creating pipeline components.
            context: Context for managing state during pipeline execution.
            logger: Logger instance for logging messages.
        """
        self.factory = factory
        self.context = context or PipelineContext()
        self.logger = logger or logging.getLogger(__name__)

    def train_model(
        self,
        data_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        test_size: float = 0.3,
        random_state: int = 42,
        optimize_hyperparameters: bool = False,
        output_dir: Optional[str] = None,
        model_name: str = "equipment_classifier",
        **kwargs,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Train a model using the pipeline components.

        This method orchestrates the execution of the pipeline components for training
        a model. It handles errors consistently and provides comprehensive logging.

        Args:
            data_path: Path to the training data.
            feature_config_path: Path to the feature configuration.
            test_size: Proportion of data to use for testing.
            random_state: Random state for reproducibility.
            optimize_hyperparameters: Whether to perform hyperparameter optimization.
            output_dir: Directory to save the trained model and results.
            model_name: Base name for the saved model.
            **kwargs: Additional arguments for pipeline components.

        Returns:
            Tuple containing the trained model and evaluation metrics.

        Raises:
            PipelineOrchestratorError: If an error occurs during pipeline execution.
        """
        try:
            # Initialize the context
            self.context.start()
            self.context.set("data_path", data_path)
            self.context.set("feature_config_path", feature_config_path)
            self.context.set("test_size", test_size)
            self.context.set("random_state", random_state)
            self.context.set("optimize_hyperparameters", optimize_hyperparameters)
            self.context.set("output_dir", output_dir)
            self.context.set("model_name", model_name)
            self.context.set("kwargs", kwargs)

            # Step 1: Load data
            self.context.start_component("data_loading")
            data_loader = self.factory.create_data_loader()
            data = data_loader.load_data(data_path, **kwargs)
            self.context.set("data", data)
            self.context.end_component()

            # Step 2: Preprocess data
            self.context.start_component("data_preprocessing")
            preprocessor = self.factory.create_data_preprocessor()
            preprocessed_data = preprocessor.preprocess(data, **kwargs)
            self.context.set("preprocessed_data", preprocessed_data)
            self.context.end_component()

            # Step 3: Engineer features
            self.context.start_component("feature_engineering")
            feature_engineer = self.factory.create_feature_engineer()
            feature_engineer.fit(preprocessed_data, **kwargs)
            engineered_data = feature_engineer.transform(preprocessed_data, **kwargs)
            self.context.set("engineered_data", engineered_data)
            self.context.set("feature_engineer", feature_engineer)
            self.context.end_component()

            # Step 4: Split data
            self.context.start_component("data_splitting")
            # Extract features and targets
            x = pd.DataFrame(
                {
                    "combined_text": engineered_data["combined_text"],
                    "service_life": engineered_data["service_life"],
                }
            )

            y = engineered_data[
                [
                    "category_name",
                    "uniformat_code",
                    "mcaa_system_category",
                    "Equipment_Type",
                    "System_Subtype",
                ]
            ]

            # Split data
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=test_size, random_state=random_state
            )

            self.context.set("x_train", x_train)
            self.context.set("x_test", x_test)
            self.context.set("y_train", y_train)
            self.context.set("y_test", y_test)
            self.context.end_component()

            # Step 5: Build model
            self.context.start_component("model_building")
            model_builder = self.factory.create_model_builder()
            model = model_builder.build_model(**kwargs)

            # Optimize hyperparameters if requested
            if optimize_hyperparameters:
                self.logger.info("Optimizing hyperparameters...")
                model = model_builder.optimize_hyperparameters(
                    model, x_train, y_train, **kwargs
                )

            self.context.set("model", model)
            self.context.end_component()

            # Step 6: Train model
            self.context.start_component("model_training")
            model_trainer = self.factory.create_model_trainer()
            trained_model = model_trainer.train(model, x_train, y_train, **kwargs)

            # Cross-validate the model
            cv_results = model_trainer.cross_validate(trained_model, x, y, **kwargs)
            self.context.set("cv_results", cv_results)
            self.context.set("trained_model", trained_model)
            self.context.end_component()

            # Step 7: Evaluate model
            self.context.start_component("model_evaluation")
            model_evaluator = self.factory.create_model_evaluator()
            metrics = model_evaluator.evaluate(trained_model, x_test, y_test, **kwargs)

            # Make predictions for detailed analysis
            # Use only the features that were used during training
            # In this case, we're only using service_life as that's what the model expects
            self.logger.info(f"x_test columns: {x_test.columns}")
            self.logger.info(f"y_test columns: {y_test.columns}")

            # Use both combined_text and service_life for prediction
            features_for_prediction = x_test[["combined_text", "service_life"]]
            self.logger.info(
                f"Making predictions with features shape: {features_for_prediction.shape}"
            )
            y_pred = trained_model.predict(features_for_prediction)
            # Handle case where y_pred might be a tuple or other structure
            if isinstance(y_pred, tuple):
                self.logger.info(f"Prediction is a tuple with {len(y_pred)} elements")
                if len(y_pred) > 0 and hasattr(y_pred[0], "shape"):
                    self.logger.info(f"First element shape: {y_pred[0].shape}")
            elif hasattr(y_pred, "shape"):
                self.logger.info(f"Prediction shape: {y_pred.shape}")
            else:
                self.logger.info(f"Prediction type: {type(y_pred)}")
            # Convert y_pred to the right format for DataFrame creation
            if isinstance(y_pred, tuple) and len(y_pred) > 0:
                # If y_pred is a tuple, use the first element
                self.logger.info(f"y_pred is a tuple with {len(y_pred)} elements")
                y_pred_array = y_pred[0]
            else:
                y_pred_array = y_pred

            # Add debug information about shapes
            self.logger.info(
                f"y_pred_array shape: {y_pred_array.shape if hasattr(y_pred_array, 'shape') else 'unknown'}"
            )
            self.logger.info(f"y_test shape: {y_test.shape}")

            # Handle shape mismatch between predictions and target columns
            if (
                hasattr(y_pred_array, "shape")
                and len(y_pred_array.shape) > 1
                and y_pred_array.shape[1] != len(y_test.columns)
            ):
                self.logger.warning(
                    f"Shape mismatch: predictions have {y_pred_array.shape[1]} columns, "
                    f"but target has {len(y_test.columns)} columns"
                )
                # Option 1: Try to use predict_proba if available and it's a classification task
                if hasattr(trained_model, "predict_proba"):
                    self.logger.info(
                        "Attempting to use predict_proba instead of predict"
                    )
                    try:
                        y_pred_proba = trained_model.predict_proba(
                            features_for_prediction
                        )
                        if hasattr(y_pred_proba, "shape") and y_pred_proba.shape[
                            1
                        ] == len(y_test.columns):
                            y_pred_array = y_pred_proba
                            self.logger.info(
                                f"Using predict_proba output with shape: {y_pred_array.shape}"
                            )
                    except Exception as e:
                        self.logger.warning(f"predict_proba failed: {str(e)}")

                # If still mismatched, create a DataFrame with appropriate columns
                if hasattr(y_pred_array, "shape") and y_pred_array.shape[1] != len(
                    y_test.columns
                ):
                    self.logger.info("Creating DataFrame with predicted_label column")
                    y_pred_df = pd.DataFrame(
                        y_pred_array,
                        columns=[
                            f"predicted_label_{i}" for i in range(y_pred_array.shape[1])
                        ],
                    )

            # If we get here, either shapes match or we're handling a single column prediction
            try:
                y_pred_df = pd.DataFrame(y_pred_array, columns=y_test.columns)
            except ValueError as e:
                self.logger.warning(f"DataFrame creation failed: {str(e)}")
                # Fallback: create DataFrame with generic column names
                if hasattr(y_pred_array, "shape") and len(y_pred_array.shape) > 1:
                    cols = [
                        f"predicted_label_{i}" for i in range(y_pred_array.shape[1])
                    ]
                else:
                    cols = ["predicted_label"]
                y_pred_df = pd.DataFrame(y_pred_array, columns=cols)
                self.logger.info(f"Created DataFrame with columns: {cols}")

            # Analyze predictions
            analysis = model_evaluator.analyze_predictions(
                trained_model, x_test, y_test, y_pred_df, **kwargs
            )

            self.context.set("metrics", metrics)
            self.context.set("analysis", analysis)
            self.context.set("y_pred", y_pred_df)
            self.context.end_component()

            # Step 8: Save model
            if output_dir:
                self.context.start_component("model_saving")
                model_serializer = self.factory.create_model_serializer()

                # Create output directory if it doesn't exist
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Save the model
                model_path = output_path / f"{model_name}.pkl"
                model_serializer.save_model(trained_model, model_path, **kwargs)

                # Save metadata
                metadata = {
                    "metrics": metrics,
                    "analysis": analysis,
                    "component_execution_times": self.context.get_component_execution_times(),
                }

                metadata_path = output_path / f"{model_name}_metadata.json"
                import json
                import numpy as np

                # Convert NumPy types to Python native types
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj

                # Convert NumPy types in metadata
                metadata = convert_numpy_types(metadata)

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                self.context.set("model_path", str(model_path))
                self.context.set("metadata_path", str(metadata_path))
                self.context.end_component()

            # Finalize context
            self.context.end("completed")

            return trained_model, metrics

        except Exception as e:
            self.logger.error(f"Error in pipeline execution: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.context.log("ERROR", f"Pipeline execution failed: {str(e)}")
            self.context.end("failed")
            raise PipelineOrchestratorError(
                f"Error in pipeline execution: {str(e)}"
            ) from e

    def predict(
        self,
        model: Optional[Pipeline] = None,
        model_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Make predictions using a trained model.

        This method orchestrates the execution of the pipeline components for making
        predictions. It handles errors consistently and provides comprehensive logging.

        Args:
            model: Trained model to use for predictions.
            model_path: Path to the trained model file.
            data: DataFrame containing the data to make predictions on.
            data_path: Path to the data file.
            output_path: Path to save the prediction results.
            **kwargs: Additional arguments for pipeline components.

        Returns:
            DataFrame containing the prediction results.

        Raises:
            PipelineOrchestratorError: If an error occurs during pipeline execution.
        """
        try:
            # Initialize the context
            self.context.start()
            self.context.set("model_path", model_path)
            self.context.set("data_path", data_path)
            self.context.set("output_path", output_path)
            self.context.set("kwargs", kwargs)

            # Step 1: Load model if not provided
            if model is None and model_path is not None:
                self.context.start_component("model_loading")
                model_serializer = self.factory.create_model_serializer()
                model = model_serializer.load_model(model_path, **kwargs)
                self.context.set("model", model)
                self.context.end_component()
            elif model is not None:
                self.context.set("model", model)
            else:
                raise PipelineOrchestratorError(
                    "Either model or model_path must be provided"
                )

            # Step 2: Load data if not provided
            if data is None and data_path is not None:
                self.context.start_component("data_loading")
                data_loader = self.factory.create_data_loader()
                data = data_loader.load_data(data_path, **kwargs)
                self.context.set("data", data)
                self.context.end_component()
            elif data is not None:
                self.context.set("data", data)
            else:
                raise PipelineOrchestratorError(
                    "Either data or data_path must be provided"
                )

            # Step 3: Preprocess data
            self.context.start_component("data_preprocessing")
            preprocessor = self.factory.create_data_preprocessor()
            preprocessed_data = preprocessor.preprocess(data, **kwargs)
            self.context.set("preprocessed_data", preprocessed_data)
            self.context.end_component()

            # Step 4: Engineer features
            self.context.start_component("feature_engineering")
            feature_engineer = self.factory.create_feature_engineer()
            engineered_data = feature_engineer.transform(preprocessed_data, **kwargs)
            self.context.set("engineered_data", engineered_data)
            self.context.end_component()

            # Step 5: Make predictions
            self.context.start_component("prediction")
            predictor = self.factory.create_predictor()
            predictions = predictor.predict(model, engineered_data, **kwargs)
            self.context.set("predictions", predictions)
            self.context.end_component()

            # Step 6: Save predictions if output path is provided
            if output_path:
                self.context.start_component("saving_predictions")
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                predictions.to_csv(output_path_obj, index=False)
                self.context.set("output_path", str(output_path_obj))
                self.context.end_component()

            # Finalize context
            self.context.end("completed")

            return predictions

        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.context.log("ERROR", f"Prediction pipeline failed: {str(e)}")
            self.context.end("failed")
            raise PipelineOrchestratorError(
                f"Error in prediction pipeline: {str(e)}"
            ) from e

    def evaluate(
        self,
        model: Optional[Pipeline] = None,
        model_path: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        data_path: Optional[str] = None,
        target_columns: Optional[List[str]] = None,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        This method orchestrates the execution of the pipeline components for evaluating
        a model. It handles errors consistently and provides comprehensive logging.

        Args:
            model: Trained model to evaluate.
            model_path: Path to the trained model file.
            data: DataFrame containing the test data.
            data_path: Path to the test data file.
            target_columns: List of target column names.
            output_path: Path to save the evaluation results.
            **kwargs: Additional arguments for pipeline components.

        Returns:
            Dictionary containing evaluation metrics.

        Raises:
            PipelineOrchestratorError: If an error occurs during pipeline execution.
        """
        try:
            # Initialize the context
            self.context.start()
            self.context.set("model_path", model_path)
            self.context.set("data_path", data_path)
            self.context.set("target_columns", target_columns)
            self.context.set("output_path", output_path)
            self.context.set("kwargs", kwargs)

            # Step 1: Load model if not provided
            if model is None and model_path is not None:
                self.context.start_component("model_loading")
                model_serializer = self.factory.create_model_serializer()
                model = model_serializer.load_model(model_path, **kwargs)
                self.context.set("model", model)
                self.context.end_component()
            elif model is not None:
                self.context.set("model", model)
            else:
                raise PipelineOrchestratorError(
                    "Either model or model_path must be provided"
                )

            # Step 2: Load data if not provided
            if data is None and data_path is not None:
                self.context.start_component("data_loading")
                data_loader = self.factory.create_data_loader()
                data = data_loader.load_data(data_path, **kwargs)
                self.context.set("data", data)
                self.context.end_component()
            elif data is not None:
                self.context.set("data", data)
            else:
                raise PipelineOrchestratorError(
                    "Either data or data_path must be provided"
                )

            # Step 3: Preprocess data
            self.context.start_component("data_preprocessing")
            preprocessor = self.factory.create_data_preprocessor()
            preprocessed_data = preprocessor.preprocess(data, **kwargs)
            self.context.set("preprocessed_data", preprocessed_data)
            self.context.end_component()

            # Step 4: Engineer features
            self.context.start_component("feature_engineering")
            feature_engineer = self.factory.create_feature_engineer()
            engineered_data = feature_engineer.transform(preprocessed_data, **kwargs)
            self.context.set("engineered_data", engineered_data)
            self.context.end_component()

            # Step 5: Prepare data for evaluation
            self.context.start_component("data_preparation")
            # Use default target columns if not provided
            if target_columns is None:
                target_columns = [
                    "category_name",
                    "uniformat_code",
                    "mcaa_system_category",
                    "Equipment_Type",
                    "System_Subtype",
                ]

            # Extract features and targets
            x = pd.DataFrame(
                {
                    "combined_text": engineered_data["combined_text"],
                    "service_life": engineered_data["service_life"],
                }
            )

            y = engineered_data[target_columns]

            self.context.set("x", x)
            self.context.set("y", y)
            self.context.end_component()

            # Step 6: Evaluate model
            self.context.start_component("model_evaluation")
            model_evaluator = self.factory.create_model_evaluator()
            metrics = model_evaluator.evaluate(model, x, y, **kwargs)

            # Make predictions for detailed analysis
            # Use only the features that were used during training
            # In this case, we're only using service_life as that's what the model expects
            self.logger.info(f"x columns: {x.columns}")
            self.logger.info(f"y columns: {y.columns}")

            # Use both combined_text and service_life for prediction
            features_for_prediction = x[["combined_text", "service_life"]]
            self.logger.info(
                f"Making predictions with features shape: {features_for_prediction.shape}"
            )
            y_pred = model.predict(features_for_prediction)

            # Handle case where y_pred might be a tuple or other structure
            if isinstance(y_pred, tuple):
                self.logger.info(f"Prediction is a tuple with {len(y_pred)} elements")
                if len(y_pred) > 0 and hasattr(y_pred[0], "shape"):
                    self.logger.info(f"First element shape: {y_pred[0].shape}")
            elif hasattr(y_pred, "shape"):
                self.logger.info(f"Prediction shape: {y_pred.shape}")
            else:
                self.logger.info(f"Prediction type: {type(y_pred)}")

            # Convert y_pred to the right format for DataFrame creation
            if isinstance(y_pred, tuple) and len(y_pred) > 0:
                # If y_pred is a tuple, use the first element
                self.logger.info(f"y_pred is a tuple with {len(y_pred)} elements")
                y_pred_array = y_pred[0]
            else:
                y_pred_array = y_pred

            # Add debug information about shapes
            if hasattr(y_pred_array, "shape"):
                self.logger.info(f"y_pred_array shape: {y_pred_array.shape}")
            else:
                self.logger.info("y_pred_array shape: unknown (no shape attribute)")
            self.logger.info(f"y shape: {y.shape}")

            # Handle shape mismatch between predictions and target columns
            if hasattr(y_pred_array, "shape") and len(y_pred_array.shape) > 1:
                pred_cols = y_pred_array.shape[1]
                target_cols = len(y.columns)
                if pred_cols != target_cols:
                    self.logger.warning(
                        f"Shape mismatch: predictions have {pred_cols} columns, "
                        f"but target has {target_cols} columns"
                    )
                    # Option 1: Try to use predict_proba if available and it's a classification task
                    if hasattr(model, "predict_proba"):
                        self.logger.info(
                            "Attempting to use predict_proba instead of predict"
                        )
                        try:
                            y_pred_proba = model.predict_proba(features_for_prediction)
                            if (
                                hasattr(y_pred_proba, "shape")
                                and y_pred_proba.shape[1] == target_cols
                            ):
                                y_pred_array = y_pred_proba
                                self.logger.info(
                                    f"Using predict_proba output with shape: {y_pred_array.shape}"
                                )
                        except Exception as e:
                            self.logger.warning(f"predict_proba failed: {str(e)}")

                    # If still mismatched, create a DataFrame with appropriate columns
                    if (
                        hasattr(y_pred_array, "shape")
                        and y_pred_array.shape[1] != target_cols
                    ):
                        self.logger.info(
                            "Creating DataFrame with predicted_label column"
                        )
                        y_pred_df = pd.DataFrame(
                            y_pred_array,
                            columns=[
                                f"predicted_label_{i}"
                                for i in range(y_pred_array.shape[1])
                            ],
                        )
                        # Continue with analysis using the custom columns
                        analysis = model_evaluator.analyze_predictions(
                            model, x, y, y_pred_df, **kwargs
                        )

                        self.context.set("metrics", metrics)
                        self.context.set("analysis", analysis)
                        self.context.set("y_pred", y_pred_df)
                        self.context.end_component()

                        # Step 7: Save evaluation results if output path is provided
                        if output_path:
                            self.context.start_component("saving_evaluation")
                            output_path_obj = Path(output_path)
                            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                            # Combine metrics and analysis
                            evaluation_results = {
                                "metrics": metrics,
                                "analysis": analysis,
                                "component_execution_times": self.context.get_component_execution_times(),
                            }

                            # Save as JSON
                            import json
                            import numpy as np

                            # Convert NumPy types to Python native types
                            def convert_numpy_types(obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                elif isinstance(obj, pd.DataFrame):
                                    return obj.to_dict(orient='records')
                                elif isinstance(obj, pd.Series):
                                    return obj.to_dict()
                                elif isinstance(obj, dict):
                                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                                elif isinstance(obj, list):
                                    return [convert_numpy_types(item) for item in obj]
                                else:
                                    return obj

                            # Convert NumPy types in evaluation_results
                            evaluation_results = convert_numpy_types(evaluation_results)

                            with open(output_path_obj, "w") as f:
                                json.dump(evaluation_results, f, indent=2)

                            self.context.set("output_path", str(output_path_obj))
                            self.context.end_component()

                        # Finalize context
                        self.context.end("completed")

                        return {
                            "metrics": metrics,
                            "analysis": analysis,
                        }

            # If we get here, either shapes match or we're handling a single column prediction
            try:
                y_pred_df = pd.DataFrame(y_pred_array, columns=y.columns)
            except ValueError as e:
                self.logger.warning(f"DataFrame creation failed: {str(e)}")
                # Fallback: create DataFrame with generic column names
                if hasattr(y_pred_array, "shape") and len(y_pred_array.shape) > 1:
                    cols = [
                        f"predicted_label_{i}" for i in range(y_pred_array.shape[1])
                    ]
                else:
                    cols = ["predicted_label"]
                y_pred_df = pd.DataFrame(y_pred_array, columns=cols)
                self.logger.info(f"Created DataFrame with columns: {cols}")

            # Analyze predictions
            analysis = model_evaluator.analyze_predictions(
                model, x, y, y_pred_df, **kwargs
            )

            self.context.set("metrics", metrics)
            self.context.set("analysis", analysis)
            self.context.set("y_pred", y_pred_df)
            self.context.end_component()

            # Step 7: Save evaluation results if output path is provided
            if output_path:
                self.context.start_component("saving_evaluation")
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)

                # Combine metrics and analysis
                evaluation_results = {
                    "metrics": metrics,
                    "analysis": analysis,
                    "component_execution_times": self.context.get_component_execution_times(),
                }

                # Save as JSON
                import json
                import numpy as np

                # Convert NumPy types to Python native types
                def convert_numpy_types(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict(orient='records')
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy_types(item) for item in obj]
                    else:
                        return obj

                # Convert NumPy types in evaluation_results
                evaluation_results = convert_numpy_types(evaluation_results)

                with open(output_path_obj, "w") as f:
                    json.dump(evaluation_results, f, indent=2)

                self.context.set("output_path", str(output_path_obj))
                self.context.end_component()

            # Finalize context
            self.context.end("completed")

            return {
                "metrics": metrics,
                "analysis": analysis,
            }

        except Exception as e:
            self.logger.error(f"Error in evaluation pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.context.log("ERROR", f"Evaluation pipeline failed: {str(e)}")
            self.context.end("failed")
            raise PipelineOrchestratorError(
                f"Error in evaluation pipeline: {str(e)}"
            ) from e

    def save_model(
        self,
        model: Pipeline,
        path: Union[str, Path],
        **kwargs,
    ) -> str:
        """
        Save a trained model to disk.

        Args:
            model: Trained model to save.
            path: Path where the model should be saved.
            **kwargs: Additional arguments for the model serializer.

        Returns:
            Path to the saved model.

        Raises:
            PipelineOrchestratorError: If an error occurs during model saving.
        """
        try:
            # Initialize the context
            self.context.start()
            self.context.set("model", model)
            self.context.set("path", str(path))
            self.context.set("kwargs", kwargs)

            # Save the model
            self.context.start_component("model_saving")
            model_serializer = self.factory.create_model_serializer()
            model_serializer.save_model(model, path, **kwargs)
            self.context.end_component()

            # Finalize context
            self.context.end("completed")

            return str(path)

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.context.log("ERROR", f"Model saving failed: {str(e)}")
            self.context.end("failed")
            raise PipelineOrchestratorError(f"Error saving model: {str(e)}") from e

    def load_model(
        self,
        path: Union[str, Path],
        **kwargs,
    ) -> Pipeline:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model.
            **kwargs: Additional arguments for the model serializer.

        Returns:
            Loaded model.

        Raises:
            PipelineOrchestratorError: If an error occurs during model loading.
        """
        try:
            # Initialize the context
            self.context.start()
            self.context.set("path", str(path))
            self.context.set("kwargs", kwargs)

            # Load the model
            self.context.start_component("model_loading")
            model_serializer = self.factory.create_model_serializer()
            model = model_serializer.load_model(path, **kwargs)
            self.context.set("model", model)
            self.context.end_component()

            # Finalize context
            self.context.end("completed")

            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.context.log("ERROR", f"Model loading failed: {str(e)}")
            self.context.end("failed")
            raise PipelineOrchestratorError(f"Error loading model: {str(e)}") from e

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution.

        Returns:
            Dictionary containing execution summary information.
        """
        return self.context.get_execution_summary()
