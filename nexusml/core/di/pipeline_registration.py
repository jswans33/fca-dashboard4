"""
Pipeline Component Registration Module

This module provides functions for registering pipeline components with the DI container.
It serves as a central place for configuring the dependency injection container
with all the pipeline components needed by the NexusML suite.
"""

import logging
from typing import Optional

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.pipeline.stages.data_loading import ConfigurableDataLoadingStage
from nexusml.core.pipeline.stages.validation import ConfigDrivenValidationStage
from nexusml.core.pipeline.stages.feature_engineering import (
    SimpleFeatureEngineeringStage,
    TextFeatureEngineeringStage,
    NumericFeatureEngineeringStage,
)
from nexusml.core.pipeline.stages.data_splitting import RandomSplittingStage
from nexusml.core.pipeline.stages.model_building import (
    ConfigDrivenModelBuildingStage,
    RandomForestModelBuildingStage,
    GradientBoostingModelBuildingStage,
)
from nexusml.core.pipeline.stages.model_training import StandardModelTrainingStage
from nexusml.core.pipeline.stages.model_evaluation import ClassificationEvaluationStage
from nexusml.core.pipeline.stages.model_saving import ModelCardSavingStage
from nexusml.core.pipeline.stages.model_loading import ModelLoadingStage
from nexusml.core.pipeline.stages.output import (
    OutputSavingStage,
    EvaluationOutputSavingStage,
)
from nexusml.core.pipeline.stages.prediction import (
    StandardPredictionStage,
    ProbabilityPredictionStage,
)
from nexusml.config.manager import ConfigurationManager

# Set up logging
logger = logging.getLogger(__name__)


def register_pipeline_stages(
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register pipeline stage components with the DI container.

    This function registers all the pipeline stage components needed by the NexusML suite,
    including data loading, validation, feature engineering, model building, and other stages.

    Args:
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()
    container = provider.container

    # Register data loading stages
    provider.register_implementation(
        ConfigurableDataLoadingStage, ConfigurableDataLoadingStage, singleton=False
    )
    logger.info("Registered ConfigurableDataLoadingStage with DI container")

    # Register validation stages
    provider.register_implementation(
        ConfigDrivenValidationStage, ConfigDrivenValidationStage, singleton=False
    )
    logger.info("Registered ConfigDrivenValidationStage with DI container")

    # Register feature engineering stages
    provider.register_implementation(
        SimpleFeatureEngineeringStage, SimpleFeatureEngineeringStage, singleton=False
    )
    provider.register_implementation(
        TextFeatureEngineeringStage, TextFeatureEngineeringStage, singleton=False
    )
    provider.register_implementation(
        NumericFeatureEngineeringStage, NumericFeatureEngineeringStage, singleton=False
    )
    logger.info("Registered feature engineering stages with DI container")

    # Register data splitting stages
    provider.register_implementation(
        RandomSplittingStage, RandomSplittingStage, singleton=False
    )
    logger.info("Registered RandomSplittingStage with DI container")

    # Register model building stages
    provider.register_implementation(
        ConfigDrivenModelBuildingStage, ConfigDrivenModelBuildingStage, singleton=False
    )
    provider.register_implementation(
        RandomForestModelBuildingStage, RandomForestModelBuildingStage, singleton=False
    )
    provider.register_implementation(
        GradientBoostingModelBuildingStage,
        GradientBoostingModelBuildingStage,
        singleton=False,
    )
    logger.info("Registered model building stages with DI container")

    # Register model training stages
    provider.register_implementation(
        StandardModelTrainingStage, StandardModelTrainingStage, singleton=False
    )
    logger.info("Registered StandardModelTrainingStage with DI container")

    # Register model evaluation stages
    provider.register_implementation(
        ClassificationEvaluationStage, ClassificationEvaluationStage, singleton=False
    )
    logger.info("Registered ClassificationEvaluationStage with DI container")

    # Register model saving stages
    provider.register_implementation(
        ModelCardSavingStage, ModelCardSavingStage, singleton=False
    )
    logger.info("Registered ModelCardSavingStage with DI container")

    # Register model loading stages
    provider.register_implementation(
        ModelLoadingStage, ModelLoadingStage, singleton=False
    )
    logger.info("Registered ModelLoadingStage with DI container")

    # Register output saving stages
    provider.register_implementation(
        OutputSavingStage, OutputSavingStage, singleton=False
    )
    provider.register_implementation(
        EvaluationOutputSavingStage, EvaluationOutputSavingStage, singleton=False
    )
    logger.info("Registered output saving stages with DI container")

    # Register prediction stages
    provider.register_implementation(
        StandardPredictionStage, StandardPredictionStage, singleton=False
    )
    provider.register_implementation(
        ProbabilityPredictionStage, ProbabilityPredictionStage, singleton=False
    )
    logger.info("Registered prediction stages with DI container")


def register_pipeline_factories(
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register pipeline factory components with the DI container.

    This function registers all the pipeline factory components needed by the NexusML suite,
    including the PipelineFactory and ComponentRegistry.

    Args:
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()

    # Import here to avoid circular imports
    from nexusml.core.pipeline.factory import PipelineFactory
    from nexusml.core.pipeline.registry import ComponentRegistry

    # Register ConfigurationManager as a singleton
    provider.register_implementation(ConfigurationManager, ConfigurationManager, singleton=True)
    logger.info("Registered ConfigurationManager with DI container")

    # Register ComponentRegistry as a singleton
    provider.register_implementation(ComponentRegistry, ComponentRegistry, singleton=True)
    logger.info("Registered ComponentRegistry with DI container")

    # Register PipelineFactory
    def pipeline_factory_provider(container: DIContainer) -> PipelineFactory:
        registry = container.resolve(ComponentRegistry)
        config_manager = container.resolve(ConfigurationManager)
        return PipelineFactory(registry, container, config_manager)

    provider.register_factory(PipelineFactory, pipeline_factory_provider, singleton=True)
    logger.info("Registered PipelineFactory with DI container")


def register_pipeline_components(
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register all pipeline components with the DI container.

    This function is the main entry point for registering all pipeline components
    with the DI container. It calls the other registration functions to register
    specific types of components.

    Args:
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()

    # Register pipeline stages
    register_pipeline_stages(provider)

    # Register pipeline factories
    register_pipeline_factories(provider)

    logger.info("All pipeline components registered successfully")