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
        
    def create_data_loader(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a data loader instance.

        Args:
            config: Configuration for the data loader.

        Returns:
            Data loader instance.
            
        Raises:
            PipelineFactoryError: If the data loader cannot be created.
        """
        try:
            # Try to get the data loader from the registry
            data_loader_class = self.registry.get("data_loader", "standard")
            if data_loader_class is None:
                # Try to resolve it from the container
                from nexusml.core.pipeline.interfaces import DataLoader
                data_loader = self.container.resolve(DataLoader)
                return data_loader
            
            # Create the data loader instance
            data_loader = data_loader_class(config=config or {})
            logger.info("Created data loader")
            
            return data_loader
        except Exception as e:
            raise PipelineFactoryError(f"Error creating data loader: {str(e)}") from e
            
    def create_model_serializer(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a model serializer instance.

        Args:
            config: Configuration for the model serializer.

        Returns:
            Model serializer instance.
            
        Raises:
            PipelineFactoryError: If the model serializer cannot be created.
        """
        try:
            # Try to get the model serializer from the registry
            model_serializer_class = self.registry.get("model_serializer", "simple")
            if model_serializer_class is None:
                # Try to resolve it from the container
                from nexusml.core.pipeline.interfaces import ModelSerializer
                model_serializer = self.container.resolve(ModelSerializer)
                return model_serializer
            
            # Create the model serializer instance
            model_serializer = model_serializer_class(config=config or {})
            logger.info("Created model serializer")
            
            return model_serializer
        except Exception as e:
            raise PipelineFactoryError(f"Error creating model serializer: {str(e)}") from e

    def create_data_preprocessor(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a data preprocessor instance.

        Args:
            config: Configuration for the data preprocessor.

        Returns:
            Data preprocessor instance.
            
        Raises:
            PipelineFactoryError: If the data preprocessor cannot be created.
        """
        try:
            # Try to get the data preprocessor from the registry
            data_preprocessor_class = self.registry.get("data_preprocessor", "standard")
            if data_preprocessor_class is None:
                # Try to resolve it from the container
                from nexusml.core.pipeline.interfaces import DataPreprocessor
                data_preprocessor = self.container.resolve(DataPreprocessor)
                return data_preprocessor
            
            # Create the data preprocessor instance
            data_preprocessor = data_preprocessor_class(config=config or {})
            logger.info("Created data preprocessor")
            
            return data_preprocessor
        except Exception as e:
            raise PipelineFactoryError(f"Error creating data preprocessor: {str(e)}") from e
    
    def create_feature_engineer(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a feature engineer instance.

        Args:
            config: Configuration for the feature engineer.

        Returns:
            Feature engineer instance.
            
        Raises:
            PipelineFactoryError: If the feature engineer cannot be created.
        """
        try:
            # Try to get the feature engineer from the registry
            feature_engineer_class = self.registry.get("feature_engineer", "simple")
            if feature_engineer_class is None:
                # Try to resolve it from the container
                from nexusml.core.pipeline.interfaces import FeatureEngineer
                feature_engineer = self.container.resolve(FeatureEngineer)
                return feature_engineer
            
            # Create the feature engineer instance
            feature_engineer = feature_engineer_class(config=config or {})
            logger.info("Created feature engineer")
            
            return feature_engineer
        except Exception as e:
            raise PipelineFactoryError(f"Error creating feature engineer: {str(e)}") from e
    
    def create_model_builder(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a model builder instance.

        Args:
            config: Configuration for the model builder.

        Returns:
            Model builder instance.
            
        Raises:
            PipelineFactoryError: If the model builder cannot be created.
        """
        try:
            # Try to get the model builder from the registry
            model_builder_class = self.registry.get("model_builder", "random_forest")
            if model_builder_class is None:
                # Try to resolve it from the container
                from nexusml.core.model_building.base import ModelBuilder
                model_builder = self.container.resolve(ModelBuilder)
                return model_builder
            
            # Create the model builder instance
            model_builder = model_builder_class(config=config or {})
            logger.info("Created model builder")
            
            return model_builder
        except Exception as e:
            raise PipelineFactoryError(f"Error creating model builder: {str(e)}") from e
    
    def create_model_trainer(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a model trainer instance.

        Args:
            config: Configuration for the model trainer.

        Returns:
            Model trainer instance.
            
        Raises:
            PipelineFactoryError: If the model trainer cannot be created.
        """
        try:
            # Try to get the model trainer from the registry
            model_trainer_class = self.registry.get("model_trainer", "standard")
            if model_trainer_class is None:
                # Try to resolve it from the container
                from nexusml.core.model_training.base import ModelTrainer
                model_trainer = self.container.resolve(ModelTrainer)
                return model_trainer
            
            # Create the model trainer instance
            model_trainer = model_trainer_class(config=config or {})
            logger.info("Created model trainer")
            
            return model_trainer
        except Exception as e:
            raise PipelineFactoryError(f"Error creating model trainer: {str(e)}") from e
    
    def create_model_evaluator(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a model evaluator instance.

        Args:
            config: Configuration for the model evaluator.

        Returns:
            Model evaluator instance.
            
        Raises:
            PipelineFactoryError: If the model evaluator cannot be created.
        """
        try:
            # Try to get the model evaluator from the registry
            model_evaluator_class = self.registry.get("model_evaluator", "classification")
            if model_evaluator_class is None:
                # Try to resolve it from the container
                from nexusml.core.model_building.base import ModelEvaluator
                model_evaluator = self.container.resolve(ModelEvaluator)
                return model_evaluator
            
            # Create the model evaluator instance
            model_evaluator = model_evaluator_class(config=config or {})
            logger.info("Created model evaluator")
            
            return model_evaluator
        except Exception as e:
            raise PipelineFactoryError(f"Error creating model evaluator: {str(e)}") from e
    
    def create_predictor(self, config: Optional[Dict[str, Any]] = None):
        """
        Create a predictor instance.

        Args:
            config: Configuration for the predictor.

        Returns:
            Predictor instance.
            
        Raises:
            PipelineFactoryError: If the predictor cannot be created.
        """
        try:
            # Try to get the predictor from the registry
            predictor_class = self.registry.get("predictor", "standard")
            if predictor_class is None:
                # Try to resolve it from the container
                from nexusml.core.pipeline.interfaces import Predictor
                from nexusml.core.pipeline.stages.prediction import StandardPredictionStage
                
                # First try to resolve a Predictor interface
                try:
                    predictor = self.container.resolve(Predictor)
                    logger.info("Resolved Predictor from container")
                    return predictor
                except Exception:
                    # If that fails, create a StandardPredictionStage
                    logger.info("Creating StandardPredictionStage as fallback")
                    return StandardPredictionStage(config=config or {})
            
            # Create the predictor instance
            predictor = predictor_class(config=config or {})
            logger.info("Created predictor")
            
            return predictor
        except Exception as e:
            raise PipelineFactoryError(f"Error creating predictor: {str(e)}") from e
            
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
