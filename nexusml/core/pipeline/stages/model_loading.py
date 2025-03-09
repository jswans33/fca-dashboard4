"""
Model Loading Stage Module

This module provides the ModelLoadingStage class, which is responsible for
loading a trained model from disk.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pickle

from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import PipelineStage

# Set up logging
logger = logging.getLogger(__name__)


class ModelLoadingStage(PipelineStage):
    """
    Stage for loading a trained model from disk.

    The ModelLoadingStage class is responsible for loading a trained model
    from disk and adding it to the pipeline context.

    Attributes:
        name: Name of the stage.
        description: Description of the stage.
    """

    def __init__(
        self,
        name: str = "ModelLoadingStage",
        description: str = "Stage for loading a trained model from disk",
    ):
        """
        Initialize the model loading stage.

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
        # This stage requires a model path in the context or kwargs
        # We'll check for the model path in the execute method
        return True

    def execute(self, context: PipelineContext, **kwargs) -> None:
        """
        Execute the model loading stage.

        This method loads a trained model from disk and adds it to the pipeline context.

        Args:
            context: Pipeline context.
            **kwargs: Additional arguments for stage execution.

        Raises:
            ValueError: If the model path is not provided or the model cannot be loaded.
        """
        logger.info("Executing model loading stage")

        # Get the model path from the context or kwargs
        model_path = kwargs.get("model_path") or context.get("model_path")
        if not model_path:
            raise ValueError("Model path not provided")

        # Convert to Path object if it's a string
        if isinstance(model_path, str):
            model_path = Path(model_path)

        # Check if the model file exists
        if not model_path.exists():
            raise ValueError(f"Model file not found: {model_path}")

        try:
            # Load the model
            logger.info(f"Loading model from {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Add the model to the context
            context.set("model", model)
            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise ValueError(f"Error loading model: {str(e)}") from e