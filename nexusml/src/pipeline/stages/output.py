"""
Output Saving Stage Module

This module provides the OutputSavingStage and EvaluationOutputSavingStage classes,
which are responsible for saving pipeline outputs to disk.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import BasePipelineStage

# Set up logging
logger = logging.getLogger(__name__)


class OutputSavingStage(BasePipelineStage):
    """
    Stage for saving pipeline outputs to disk.

    The OutputSavingStage class is responsible for saving prediction results
    to disk in various formats.

    Attributes:
        name: Name of the stage.
        description: Description of the stage.
    """

    def __init__(
        self,
        name: str = "OutputSavingStage",
        description: str = "Stage for saving pipeline outputs to disk",
    ):
        """
        Initialize the output saving stage.

        Args:
            name: Name of the stage.
            description: Description of the stage.
        """
        self._name = name
        self._description = description

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # Check if predictions exist in the context
        predictions = context.get("predictions")
        return predictions is not None

    def execute(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the output saving stage.

        This method saves prediction results to disk in the specified format.

        Args:
            context: Pipeline context.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the output path is not provided or the predictions cannot be saved.
        """
        logger.info("Executing output saving stage")

        # Get the output path from the context or kwargs
        output_path = kwargs.get("output_path") or context.get("output_path")
        if not output_path:
            raise ValueError("Output path not provided")

        # Convert to Path object if it's a string
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the predictions from the context
        predictions = context.get("predictions")
        if predictions is None:
            raise ValueError("No predictions found in context")

        try:
            # Save the predictions
            logger.info(f"Saving predictions to {output_path}")

            # Determine the file format based on the extension
            extension = output_path.suffix.lower()
            if extension == ".csv":
                predictions.to_csv(output_path, index=False)
            elif extension in (".xls", ".xlsx"):
                predictions.to_excel(output_path, index=False)
            elif extension == ".json":
                predictions.to_json(output_path, orient="records", indent=2)
            else:
                # Default to CSV
                predictions.to_csv(output_path, index=False)

            # Add the output path to the context
            context.set("output_path", str(output_path))
            logger.info("Predictions saved successfully")

        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
            raise ValueError(f"Error saving predictions: {str(e)}") from e


class EvaluationOutputSavingStage(BasePipelineStage):
    """
    Stage for saving evaluation results to disk.

    The EvaluationOutputSavingStage class is responsible for saving evaluation
    results to disk in JSON format.

    Attributes:
        name: Name of the stage.
        description: Description of the stage.
    """

    def __init__(
        self,
        name: str = "EvaluationOutputSavingStage",
        description: str = "Stage for saving evaluation results to disk",
    ):
        """
        Initialize the evaluation output saving stage.

        Args:
            name: Name of the stage.
            description: Description of the stage.
        """
        self._name = name
        self._description = description

    def get_name(self) -> str:
        """
        Get the name of the component.

        Returns:
            Component name.
        """
        return self._name

    def get_description(self) -> str:
        """
        Get a description of the component.

        Returns:
            Component description.
        """
        return self._description

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the component configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        # Check if metrics exist in the context
        metrics = context.get("metrics")
        return metrics is not None

    def execute(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the evaluation output saving stage.

        This method saves evaluation results to disk in JSON format.

        Args:
            context: Pipeline context.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the output path is not provided or the evaluation results cannot be saved.
        """
        logger.info("Executing evaluation output saving stage")

        # Get the output path from the context or kwargs
        output_path = kwargs.get("output_path") or context.get("output_path")
        if not output_path:
            raise ValueError("Output path not provided")

        # Convert to Path object if it's a string
        if isinstance(output_path, str):
            output_path = Path(output_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get the evaluation metrics and analysis from the context
        metrics = context.get("metrics")
        analysis = context.get("analysis")

        if metrics is None:
            raise ValueError("No evaluation metrics found in context")

        try:
            # Combine metrics and analysis into a single dictionary
            evaluation_results = {
                "metrics": metrics,
                "analysis": analysis,
                "component_execution_times": context.get_component_execution_times(),
            }

            # Save the evaluation results
            logger.info(f"Saving evaluation results to {output_path}")

            # Save as JSON
            with open(output_path, "w") as f:
                json.dump(evaluation_results, f, indent=2)

            # Add the output path to the context
            context.set("output_path", str(output_path))
            logger.info("Evaluation results saved successfully")

        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
            raise ValueError(f"Error saving evaluation results: {str(e)}") from e
