"""
Tests for Pipeline Registration Module

This module contains tests for the pipeline registration functionality,
ensuring that pipeline components are properly registered with the DI container.
"""

import unittest
from unittest.mock import MagicMock, patch

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.di.pipeline_registration import (
    register_pipeline_components,
    register_pipeline_stages,
    register_pipeline_factories,
)
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
from nexusml.core.pipeline.stages.model_loading import ModelLoadingStage
from nexusml.core.pipeline.stages.output import (
    OutputSavingStage,
    EvaluationOutputSavingStage,
)
from nexusml.core.pipeline.stages.prediction import (
    StandardPredictionStage,
    ProbabilityPredictionStage,
)


class TestPipelineRegistration(unittest.TestCase):
    """
    Test case for pipeline registration functionality.
    """

    def setUp(self):
        """
        Set up the test case.
        """
        # Create a fresh container provider for each test
        self.provider = ContainerProvider()
        self.provider.reset()
        self.container = self.provider.container

    def test_register_pipeline_stages(self):
        """
        Test that pipeline stages are properly registered.
        """
        # Register pipeline stages
        register_pipeline_stages(self.provider)

        # Verify that stages can be resolved
        stages_to_test = [
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
            ModelLoadingStage,
            OutputSavingStage,
            EvaluationOutputSavingStage,
            StandardPredictionStage,
            ProbabilityPredictionStage,
        ]

        for stage_type in stages_to_test:
            with self.subTest(stage_type=stage_type.__name__):
                # Verify that the stage can be resolved
                stage = self.container.resolve(stage_type)
                self.assertIsNotNone(stage)
                self.assertIsInstance(stage, stage_type)

    def test_register_pipeline_factories(self):
        """
        Test that pipeline factories are properly registered.
        """
        # Register ComponentRegistry first (required by PipelineFactory)
        self.provider.register_implementation(
            ComponentRegistry, ComponentRegistry, singleton=True
        )

        # Register pipeline factories
        register_pipeline_factories(self.provider)

        # Verify that factories can be resolved
        factories_to_test = [
            PipelineFactory,
        ]

        for factory_type in factories_to_test:
            with self.subTest(factory_type=factory_type.__name__):
                # Verify that the factory can be resolved
                factory = self.container.resolve(factory_type)
                self.assertIsNotNone(factory)
                self.assertIsInstance(factory, factory_type)

    def test_register_pipeline_components(self):
        """
        Test that all pipeline components are properly registered.
        """
        # Register all pipeline components
        register_pipeline_components(self.provider)

        # Verify that stages can be resolved
        stages_to_test = [
            ConfigurableDataLoadingStage,
            ConfigDrivenValidationStage,
            SimpleFeatureEngineeringStage,
            RandomSplittingStage,
            ConfigDrivenModelBuildingStage,
            StandardModelTrainingStage,
            ClassificationEvaluationStage,
            ModelCardSavingStage,
            ModelLoadingStage,
            OutputSavingStage,
            EvaluationOutputSavingStage,
            StandardPredictionStage,
        ]

        for stage_type in stages_to_test:
            with self.subTest(stage_type=stage_type.__name__):
                # Verify that the stage can be resolved
                stage = self.container.resolve(stage_type)
                self.assertIsNotNone(stage)
                self.assertIsInstance(stage, stage_type)

        # Verify that factories can be resolved
        factories_to_test = [
            PipelineFactory,
        ]

        for factory_type in factories_to_test:
            with self.subTest(factory_type=factory_type.__name__):
                # Verify that the factory can be resolved
                factory = self.container.resolve(factory_type)
                self.assertIsNotNone(factory)
                self.assertIsInstance(factory, factory_type)

    @patch('nexusml.core.di.pipeline_registration.register_pipeline_stages')
    @patch('nexusml.core.di.pipeline_registration.register_pipeline_factories')
    def test_register_pipeline_components_calls_sub_functions(
        self, mock_register_factories, mock_register_stages
    ):
        """
        Test that register_pipeline_components calls the sub-functions.
        """
        # Register all pipeline components
        register_pipeline_components(self.provider)

        # Verify that the sub-functions were called
        mock_register_stages.assert_called_once_with(self.provider)
        mock_register_factories.assert_called_once_with(self.provider)


if __name__ == '__main__':
    unittest.main()