"""
Training Pipeline Module

This module provides the TrainingPipeline class, which is responsible for
orchestrating the execution of pipeline stages for training machine learning models.
"""

import logging
from typing import Any, Dict, Optional

from nexusml.src.pipeline.pipelines.base import BasePipeline
from nexusml.src.pipeline.stages.data_loading import ConfigurableDataLoadingStage
from nexusml.src.pipeline.stages.data_splitting import RandomSplittingStage
from nexusml.src.pipeline.stages.feature_engineering import (
    SimpleFeatureEngineeringStage,
)
from nexusml.src.pipeline.stages.model_building import ConfigDrivenModelBuildingStage
from nexusml.src.pipeline.stages.model_evaluation import ClassificationEvaluationStage
from nexusml.src.pipeline.stages.model_saving import ModelCardSavingStage
from nexusml.src.pipeline.stages.model_training import StandardModelTrainingStage
from nexusml.src.pipeline.stages.validation import ConfigDrivenValidationStage
from nexusml.src.utils.di.container import DIContainer

# Set up logging
logger = logging.getLogger(__name__)


class TrainingPipeline(BasePipeline):
    """
    Pipeline for training machine learning models.

    The TrainingPipeline class orchestrates the execution of pipeline stages
    for training machine learning models. It includes stages for data loading,
    validation, feature engineering, model building, training, evaluation, and saving.

    Attributes:
        config: Configuration for the pipeline.
        container: DI container for resolving dependencies.
        stages: List of pipeline stages to execute.
    """

    def _initialize_stages(self) -> None:
        """
        Initialize the training pipeline stages.

        This method adds the standard stages for a training pipeline:
        1. Data loading
        2. Data validation
        3. Feature engineering
        4. Data splitting
        5. Model building
        6. Model training
        7. Model evaluation
        8. Model saving
        """
        logger.info("Initializing training pipeline stages")

        # Add data loading stage
        self.add_stage(self.container.resolve(ConfigurableDataLoadingStage))

        # Add validation stage
        self.add_stage(self.container.resolve(ConfigDrivenValidationStage))

        # Add feature engineering stage
        self.add_stage(self.container.resolve(SimpleFeatureEngineeringStage))

        # Add data splitting stage
        self.add_stage(self.container.resolve(RandomSplittingStage))

        # Add model building stage
        self.add_stage(self.container.resolve(ConfigDrivenModelBuildingStage))

        # Add model training stage
        self.add_stage(self.container.resolve(StandardModelTrainingStage))

        # Add model evaluation stage
        self.add_stage(self.container.resolve(ClassificationEvaluationStage))

        # Add model saving stage if output_dir is provided in config
        if self.config.get("output_dir"):
            self.add_stage(self.container.resolve(ModelCardSavingStage))
        else:
            logger.warning(
                "No output_dir specified in config, skipping model saving stage"
            )

        logger.info(f"Training pipeline initialized with {len(self.stages)} stages")
