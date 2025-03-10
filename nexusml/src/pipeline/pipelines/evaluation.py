"""
Evaluation Pipeline Module

This module provides the EvaluationPipeline class, which is responsible for
orchestrating the execution of pipeline stages for evaluating trained machine
learning models on test data.
"""

import logging
from typing import Any, Dict, Optional

from nexusml.src.pipeline.pipelines.base import BasePipeline
from nexusml.src.pipeline.stages.data_loading import ConfigurableDataLoadingStage
from nexusml.src.pipeline.stages.feature_engineering import (
    SimpleFeatureEngineeringStage,
)
from nexusml.src.pipeline.stages.model_evaluation import ClassificationEvaluationStage
from nexusml.src.pipeline.stages.model_loading import ModelLoadingStage
from nexusml.src.pipeline.stages.validation import ConfigDrivenValidationStage
from nexusml.src.utils.di.container import DIContainer

# Set up logging
logger = logging.getLogger(__name__)


class EvaluationPipeline(BasePipeline):
    """
    Pipeline for evaluating trained machine learning models on test data.

    The EvaluationPipeline class orchestrates the execution of pipeline stages
    for evaluating trained machine learning models on test data. It includes
    stages for data loading, validation, feature engineering, model loading,
    and evaluation.

    Attributes:
        config: Configuration for the pipeline.
        container: DI container for resolving dependencies.
        stages: List of pipeline stages to execute.
    """

    def _initialize_stages(self) -> None:
        """
        Initialize the evaluation pipeline stages.

        This method adds the standard stages for an evaluation pipeline:
        1. Data loading
        2. Data validation
        3. Feature engineering
        4. Model loading
        5. Model evaluation
        """
        logger.info("Initializing evaluation pipeline stages")

        # Add data loading stage
        self.add_stage(self.container.resolve(ConfigurableDataLoadingStage))

        # Add validation stage
        self.add_stage(self.container.resolve(ConfigDrivenValidationStage))

        # Add feature engineering stage
        self.add_stage(self.container.resolve(SimpleFeatureEngineeringStage))

        # Add model loading stage
        self.add_stage(self.container.resolve(ModelLoadingStage))

        # Add model evaluation stage
        self.add_stage(self.container.resolve(ClassificationEvaluationStage))

        # Add output saving stage if output_path is provided in config
        if self.config.get("output_path"):
            from nexusml.src.pipeline.stages.output import EvaluationOutputSavingStage

            self.add_stage(self.container.resolve(EvaluationOutputSavingStage))
        else:
            logger.warning(
                "No output_path specified in config, evaluation results will not be saved"
            )

        logger.info(f"Evaluation pipeline initialized with {len(self.stages)} stages")
