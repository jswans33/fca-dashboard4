"""
Pipeline Component Resolution Test

This script tests the resolution of pipeline components from the dependency injection container.
It verifies that all registered components can be properly resolved and are of the expected type.
"""

import logging
import sys
from typing import Any, Dict, List, Type

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.di.registration import register_core_components
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
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
from nexusml.core.pipeline.stages.prediction import (
    StandardPredictionStage,
    ProbabilityPredictionStage,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_component_resolution() -> Dict[str, bool]:
    """
    Test the resolution of pipeline components from the DI container.

    Returns:
        Dictionary mapping component names to resolution success status.
    """
    # Initialize the container provider and register components
    provider = ContainerProvider()
    register_core_components(provider)
    container = provider.container

    # Define components to test
    components_to_test: List[Type[Any]] = [
        # Pipeline factories
        PipelineFactory,
        ComponentRegistry,
        
        # Pipeline stages
        ConfigurableDataLoadingStage,
        ConfigDrivenValidationStage,
        SimpleFeatureEngineeringStage,
        TextFeatureEngineeringStage,
        NumericFeatureEngineeringStage,
        RandomSplittingStage,
        ConfigDrivenModelBuildingStage,
        RandomForestModelBuildingStage,
        GradientBoostingModelBuildingStage,
        StandardModelTrainingStage,
        ClassificationEvaluationStage,
        ModelCardSavingStage,
        StandardPredictionStage,
        ProbabilityPredictionStage,
    ]

    # Test resolution of each component
    results = {}
    for component_type in components_to_test:
        component_name = component_type.__name__
        try:
            resolved = container.resolve(component_type)
            if isinstance(resolved, component_type):
                logger.info(f"✅ Successfully resolved {component_name}")
                results[component_name] = True
            else:
                logger.error(
                    f"❌ Resolved component {component_name} is of unexpected type: {type(resolved).__name__}"
                )
                results[component_name] = False
        except Exception as e:
            logger.error(f"❌ Failed to resolve {component_name}: {str(e)}")
            results[component_name] = False

    return results


def print_summary(results: Dict[str, bool]) -> None:
    """
    Print a summary of the test results.

    Args:
        results: Dictionary mapping component names to resolution success status.
    """
    total = len(results)
    successful = sum(1 for success in results.values() if success)
    failed = total - successful

    logger.info(f"\n{'=' * 50}")
    logger.info(f"COMPONENT RESOLUTION TEST SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"Total components tested: {total}")
    logger.info(f"Successfully resolved:   {successful}")
    logger.info(f"Failed to resolve:       {failed}")
    logger.info(f"{'=' * 50}")

    if failed > 0:
        logger.info("Failed components:")
        for component_name, success in results.items():
            if not success:
                logger.info(f"  - {component_name}")
        logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    logger.info("Starting pipeline component resolution test...")
    results = test_component_resolution()
    print_summary(results)

    # Exit with appropriate status code
    if all(results.values()):
        logger.info("All components resolved successfully!")
        sys.exit(0)
    else:
        logger.error("Some components failed to resolve. See summary above.")
        sys.exit(1)