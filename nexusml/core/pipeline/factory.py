"""
Pipeline Factory Module

This module provides the PipelineFactory class, which is responsible for
creating pipeline instances with proper dependencies.
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from nexusml.config.manager import ConfigurationManager
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.pipelines.base import BasePipeline
from nexusml.core.pipeline.pipelines.training import TrainingPipeline
from nexusml.core.pipeline.pipelines.prediction import PredictionPipeline
from nexusml.core.pipeline.pipelines.evaluation import EvaluationPipeline
from nexusml.core.pipeline.registry import ComponentRegistry

# Set up logging
logger = logging.getLogger(__name__)


class PipelineFactoryError(Exception):
    """Exception raised for errors in the PipelineFactory."""
    pass


class PipelineFactory:
    """
    Factory for creating pipeline instances.

    The PipelineFactory class is responsible for creating pipeline instances
    with proper dependencies. It uses the ComponentRegistry to look up pipeline
    implementations and the DIContainer to resolve dependencies.

    Attributes:
        registry: Component registry for looking up pipeline implementations.
        container: DI container for resolving dependencies.
        config_manager: Configuration manager for loading pipeline configurations.
    """

    def __init__(
        self, 
        registry: ComponentRegistry, 
        container: DIContainer,
        config_manager: Optional[ConfigurationManager] = None
    ):
        """
        Initialize a new PipelineFactory.

        Args:
            registry: Component registry for looking up pipeline implementations.
            container: DI container for resolving dependencies.
            config_manager: Configuration manager for loading pipeline configurations.
        """
        self.registry = registry
        self.container = container
        self.config_manager = config_manager or ConfigurationManager()
        
        # Register built-in pipeline types
        self._pipeline_types = {
            "training": TrainingPipeline,
            "prediction": PredictionPipeline,
            "evaluation": EvaluationPipeline,
        }
        
        logger.info("PipelineFactory initialized with built-in pipeline types")

    def create_pipeline(
        self, 
        pipeline_type: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> BasePipeline:
        """
        Create a pipeline of the specified type.

        Args:
            pipeline_type: Type of pipeline to create.
            config: Configuration for the pipeline.

        Returns:
            Created pipeline instance.

        Raises:
            PipelineFactoryError: If the pipeline type is not supported.
        """
        try:
            # Check if the pipeline type is registered
            if pipeline_type not in self._pipeline_types:
                # Try to get it from the registry
                pipeline_class = self.registry.get("pipeline", pipeline_type)
                if pipeline_class is None:
                    raise PipelineFactoryError(f"Unsupported pipeline type: {pipeline_type}")
            else:
                pipeline_class = self._pipeline_types[pipeline_type]
            
            # Create the pipeline instance
            pipeline = pipeline_class(config=config or {}, container=self.container)
            logger.info(f"Created pipeline of type {pipeline_type}")
            
            return pipeline
        except Exception as e:
            if isinstance(e, PipelineFactoryError):
                raise
            raise PipelineFactoryError(f"Error creating pipeline: {str(e)}") from e

    def register_pipeline_type(self, name: str, pipeline_class: Type[BasePipeline]) -> None:
        """
        Register a new pipeline type.

        Args:
            name: Name of the pipeline type.
            pipeline_class: Pipeline class to register.
        """
        self._pipeline_types[name] = pipeline_class
        logger.info(f"Registered pipeline type: {name}")
        
        # Also register with the component registry
        self.registry.register("pipeline", name, pipeline_class)

    def create_training_pipeline(self, config: Optional[Dict[str, Any]] = None) -> TrainingPipeline:
        """
        Create a training pipeline.

        Args:
            config: Configuration for the pipeline.

        Returns:
            Training pipeline instance.
        """
        return self.create_pipeline("training", config)

    def create_prediction_pipeline(self, config: Optional[Dict[str, Any]] = None) -> PredictionPipeline:
        """
        Create a prediction pipeline.

        Args:
            config: Configuration for the pipeline.

        Returns:
            Prediction pipeline instance.
        """
        return self.create_pipeline("prediction", config)

    def create_evaluation_pipeline(self, config: Optional[Dict[str, Any]] = None) -> EvaluationPipeline:
        """
        Create an evaluation pipeline.

        Args:
            config: Configuration for the pipeline.

        Returns:
            Evaluation pipeline instance.
        """
        return self.create_pipeline("evaluation", config)

    def create_pipeline_from_config(self, config_path: str) -> BasePipeline:
        """
        Create a pipeline from a configuration file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Created pipeline instance.

        Raises:
            PipelineFactoryError: If the configuration is invalid or the pipeline type is not supported.
        """
        try:
            # Load the configuration
            config = self.config_manager.load_config(config_path)
            
            # Get the pipeline type
            pipeline_type = config.get("pipeline_type")
            if not pipeline_type:
                raise PipelineFactoryError("Pipeline type not specified in configuration")
            
            # Create the pipeline
            return self.create_pipeline(pipeline_type, config)
        except Exception as e:
            if isinstance(e, PipelineFactoryError):
                raise
            raise PipelineFactoryError(f"Error creating pipeline from config: {str(e)}") from e
