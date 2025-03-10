"""
Pipeline Stage Base Implementations Module

This module provides base implementations for the pipeline stage interfaces.
These base classes implement common functionality and provide default behavior
where appropriate, following the Template Method pattern.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.interfaces import (
    DataLoadingStage,
    DataSplittingStage,
    FeatureEngineeringStage,
    ModelBuildingStage,
    ModelEvaluationStage,
    ModelSavingStage,
    ModelTrainingStage,
    PipelineStage,
    PredictionStage,
    ValidationStage,
)


class BasePipelineStage(PipelineStage):
    """
    Base implementation of the PipelineStage interface.

    Provides common functionality for all pipeline stages.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the stage with a name and description.

        Args:
            name: Stage name.
            description: Stage description.
        """
        self._name = name
        self._description = description

    def get_name(self) -> str:
        """
        Get the name of the stage.

        Returns:
            Stage name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the stage.

        Returns:
            Stage description.
        """
        return self._description

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        This base implementation always returns True.
        Subclasses should override this method to provide specific validation.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return True

    def execute(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the stage.

        This base implementation logs the stage execution and delegates to
        the stage-specific implementation method.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the stage cannot be executed.
        """
        try:
            # Start timing the stage execution
            context.start_component(self.get_name())

            # Validate the context
            if not self.validate_context(context):
                raise ValueError(f"Invalid context for stage {self.get_name()}")

            # Execute the stage-specific implementation
            self._execute_impl(context, **kwargs)

            # End timing the stage execution
            context.end_component()
        except Exception as e:
            # Log the error and re-raise
            context.log("ERROR", f"Error in stage {self.get_name()}: {str(e)}")
            raise

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Stage-specific implementation of the execute method.

        This method should be overridden by subclasses to provide the actual
        implementation of the stage.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _execute_impl()")


class BaseDataLoadingStage(BasePipelineStage, DataLoadingStage):
    """
    Base implementation of the DataLoadingStage interface.

    Provides common functionality for data loading stages.
    """

    def __init__(
        self,
        name: str = "DataLoading",
        description: str = "Loads data from a source",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data loading stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the data loading stage.

        This implementation loads data using the load_data method and stores
        it in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the data path from the context or kwargs
        data_path = kwargs.get("data_path", context.get("data_path"))

        # Create a copy of kwargs without data_path to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if "data_path" in kwargs_copy:
            del kwargs_copy["data_path"]

        # Load the data
        data = self.load_data(data_path, **kwargs_copy)

        # Store the data in the context
        context.set("data", data)
        context.log("INFO", f"Loaded data with shape {data.shape}")

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from the specified path.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific data loading logic.

        Args:
            data_path: Path to the data file. If None, uses a default path.
            **kwargs: Additional arguments for data loading.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_data()")


class BaseValidationStage(BasePipelineStage, ValidationStage):
    """
    Base implementation of the ValidationStage interface.

    Provides common functionality for data validation stages.
    """

    def __init__(
        self,
        name: str = "Validation",
        description: str = "Validates data against requirements",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the validation stage.

        This implementation validates the data using the validate_data method
        and stores the validation results in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the data is not valid and strict validation is enabled.
        """
        # Get the data from the context
        data = context.get("data")
        if data is None:
            raise ValueError("No data found in context")

        # Validate the data
        validation_results = self.validate_data(data, **kwargs)

        # Store the validation results in the context
        context.set("validation_results", validation_results)

        # Log validation results
        if validation_results.get("valid", False):
            context.log("INFO", "Data validation passed")
        else:
            issues = validation_results.get("issues", [])
            context.log("WARNING", f"Data validation failed: {issues}")

            # If strict validation is enabled, raise an error
            if kwargs.get("strict", self.config.get("strict", False)):
                raise ValueError(f"Data validation failed: {issues}")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the input data.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific validation logic.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement validate_data()")


class BaseFeatureEngineeringStage(BasePipelineStage, FeatureEngineeringStage):
    """
    Base implementation of the FeatureEngineeringStage interface.

    Provides common functionality for feature engineering stages.
    """

    def __init__(
        self,
        name: str = "FeatureEngineering",
        description: str = "Engineers features from raw data",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the feature engineering stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the feature engineering stage.

        This implementation engineers features using the engineer_features method
        and stores the engineered data in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the data from the context
        data = context.get("data")
        if data is None:
            raise ValueError("No data found in context")

        # Engineer features
        engineered_data = self.engineer_features(data, **kwargs)

        # Store the engineered data in the context
        context.set("engineered_data", engineered_data)
        context.log("INFO", f"Engineered features with shape {engineered_data.shape}")

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific feature engineering logic.

        Args:
            data: Input DataFrame with raw features.
            **kwargs: Additional arguments for feature engineering.

        Returns:
            DataFrame with engineered features.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement engineer_features()")


class BaseDataSplittingStage(BasePipelineStage, DataSplittingStage):
    """
    Base implementation of the DataSplittingStage interface.

    Provides common functionality for data splitting stages.
    """

    def __init__(
        self,
        name: str = "DataSplitting",
        description: str = "Splits data into training and testing sets",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the data splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the data splitting stage.

        This implementation splits the data using the split_data method
        and stores the split data in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the data from the context
        data = context.get("engineered_data", context.get("data"))
        if data is None:
            raise ValueError("No data found in context")

        # Get target columns from kwargs or config
        target_columns = kwargs.get(
            "target_columns", self.config.get("target_columns", [])
        )
        if not target_columns:
            raise ValueError("No target columns specified")

        # Create a copy of kwargs without target_columns to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if "target_columns" in kwargs_copy:
            del kwargs_copy["target_columns"]

        # Split the data
        x_train, x_test, y_train, y_test = self.split_data(
            data, target_columns, **kwargs_copy
        )

        # Store the split data in the context
        context.set("x_train", x_train)
        context.set("x_test", x_test)
        context.set("y_train", y_train)
        context.set("y_test", y_test)
        context.log(
            "INFO",
            f"Split data into training ({x_train.shape[0]} samples) and testing ({x_test.shape[0]} samples) sets",
        )

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        This base implementation uses scikit-learn's train_test_split function.
        Subclasses can override this method to provide custom splitting logic.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).
        """
        # Get feature columns (all columns except target columns)
        feature_columns = [col for col in data.columns if col not in target_columns]

        # Extract features and targets
        x = data[feature_columns]
        y = data[target_columns]

        # Get split parameters
        test_size = kwargs.get("test_size", self.config.get("test_size", 0.3))
        random_state = kwargs.get("random_state", self.config.get("random_state", 42))
        stratify = kwargs.get("stratify", self.config.get("stratify", None))

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        return x_train, x_test, y_train, y_test


class BaseModelBuildingStage(BasePipelineStage, ModelBuildingStage):
    """
    Base implementation of the ModelBuildingStage interface.

    Provides common functionality for model building stages.
    """

    def __init__(
        self,
        name: str = "ModelBuilding",
        description: str = "Builds a machine learning model",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model building stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the model building stage.

        This implementation builds a model using the build_model method
        and stores it in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Build the model
        model = self.build_model(**kwargs)

        # Store the model in the context
        context.set("model", model)
        context.log("INFO", f"Built model: {type(model).__name__}")

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific model building logic.

        Args:
            **kwargs: Configuration parameters for the model.

        Returns:
            Configured model pipeline.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build_model()")


class BaseModelTrainingStage(BasePipelineStage, ModelTrainingStage):
    """
    Base implementation of the ModelTrainingStage interface.

    Provides common functionality for model training stages.
    """

    def __init__(
        self,
        name: str = "ModelTraining",
        description: str = "Trains a machine learning model",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model training stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the model training stage.

        This implementation trains a model using the train_model method
        and stores it in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the model, training features, and targets from the context
        model = context.get("model")
        if model is None:
            raise ValueError("No model found in context")

        x_train = context.get("x_train")
        if x_train is None:
            raise ValueError("No training features found in context")

        y_train = context.get("y_train")
        if y_train is None:
            raise ValueError("No training targets found in context")

        # Train the model
        trained_model = self.train_model(model, x_train, y_train, **kwargs)

        # Store the trained model in the context
        context.set("trained_model", trained_model)
        context.log("INFO", "Model training completed")

    def train_model(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model on the provided data.

        This base implementation simply calls the model's fit method.
        Subclasses can override this method to provide custom training logic.

        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.

        Returns:
            Trained model pipeline.
        """
        model.fit(x_train, y_train)
        return model


class BaseModelEvaluationStage(BasePipelineStage, ModelEvaluationStage):
    """
    Base implementation of the ModelEvaluationStage interface.

    Provides common functionality for model evaluation stages.
    """

    def __init__(
        self,
        name: str = "ModelEvaluation",
        description: str = "Evaluates a trained model",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model evaluation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the model evaluation stage.

        This implementation evaluates a model using the evaluate_model method
        and stores the evaluation results in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the trained model, test features, and targets from the context
        model = context.get("trained_model", context.get("model"))
        if model is None:
            raise ValueError("No trained model found in context")

        x_test = context.get("x_test")
        if x_test is None:
            raise ValueError("No test features found in context")

        y_test = context.get("y_test")
        if y_test is None:
            raise ValueError("No test targets found in context")

        # Evaluate the model
        evaluation_results = self.evaluate_model(model, x_test, y_test, **kwargs)

        # Store the evaluation results in the context
        context.set("evaluation_results", evaluation_results)
        context.log("INFO", f"Model evaluation completed: {evaluation_results}")

    def evaluate_model(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific evaluation logic.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement evaluate_model()")


class BaseModelSavingStage(BasePipelineStage, ModelSavingStage):
    """
    Base implementation of the ModelSavingStage interface.

    Provides common functionality for model saving stages.
    """

    def __init__(
        self,
        name: str = "ModelSaving",
        description: str = "Saves a trained model",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model saving stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the model saving stage.

        This implementation saves a model using the save_model method.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the trained model from the context
        model = context.get("trained_model", context.get("model"))
        if model is None:
            raise ValueError("No trained model found in context")

        # Get the output path from kwargs or context
        output_path = kwargs.get("output_path", context.get("output_path"))
        if output_path is None:
            raise ValueError("No output path specified")

        # Get metadata from the context
        metadata = {
            "evaluation_results": context.get("evaluation_results", {}),
            "component_execution_times": context.get_component_execution_times(),
        }

        # Save the model
        self.save_model(model, output_path, metadata, **kwargs)
        context.log("INFO", f"Model saved to {output_path}")

    def save_model(
        self,
        model: Pipeline,
        path: Union[str, Path],
        metadata: Dict[str, Any],
        **kwargs,
    ) -> None:
        """
        Save a trained model and its metadata to disk.

        This base implementation raises NotImplementedError.
        Subclasses must override this method to provide specific saving logic.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            metadata: Model metadata to save.
            **kwargs: Additional arguments for saving.

        Raises:
            NotImplementedError: This base method must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement save_model()")


class BasePredictionStage(BasePipelineStage, PredictionStage):
    """
    Base implementation of the PredictionStage interface.

    Provides common functionality for prediction stages.
    """

    def __init__(
        self,
        name: str = "Prediction",
        description: str = "Makes predictions using a trained model",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the prediction stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
        """
        super().__init__(name, description)
        self.config = config or {}

    def _execute_impl(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the prediction stage.

        This implementation makes predictions using the predict method
        and stores the predictions in the context.

        Args:
            context: The pipeline context for sharing data between stages.
            **kwargs: Additional arguments for stage execution.
        """
        # Get the model and data from the context
        model = context.get("trained_model", context.get("model"))
        if model is None:
            raise ValueError("No trained model found in context")

        data = context.get("engineered_data", context.get("data"))
        if data is None:
            raise ValueError("No data found in context")

        # Make predictions
        predictions = self.predict(model, data, **kwargs)

        # Store the predictions in the context
        context.set("predictions", predictions)
        context.log("INFO", f"Made predictions with shape {predictions.shape}")

    def predict(self, model: Pipeline, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Make predictions using a trained model.

        This base implementation simply calls the model's predict method.
        Subclasses can override this method to provide custom prediction logic.

        Args:
            model: Trained model pipeline.
            data: Input data for prediction.
            **kwargs: Additional arguments for prediction.

        Returns:
            DataFrame containing predictions.
        """
        # Make predictions
        predictions = model.predict(data)

        # Convert to DataFrame if it's not already
        if not isinstance(predictions, pd.DataFrame):
            # Try to get column names from the model
            try:
                column_names = model.classes_
            except AttributeError:
                # If that fails, use generic column names
                column_names = [f"prediction_{i}" for i in range(predictions.shape[1])]

            predictions = pd.DataFrame(predictions, columns=column_names)

        return predictions
