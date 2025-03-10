"""
Prediction Pipeline Module

This module provides the PredictionPipeline class, which is responsible for
orchestrating the execution of pipeline stages for making predictions with
trained machine learning models.
"""

import logging
from typing import Any, Dict, Optional

from nexusml.src.pipeline.pipelines.base import BasePipeline
from nexusml.src.pipeline.stages.data_loading import ConfigurableDataLoadingStage
from nexusml.src.pipeline.stages.feature_engineering import (
    SimpleFeatureEngineeringStage,
)
from nexusml.src.pipeline.stages.model_loading import ModelLoadingStage
from nexusml.src.pipeline.stages.prediction import StandardPredictionStage
from nexusml.src.pipeline.stages.validation import ConfigDrivenValidationStage
from nexusml.src.utils.di.container import DIContainer

# Set up logging
logger = logging.getLogger(__name__)


class PredictionPipeline(BasePipeline):
    """
    Pipeline for making predictions with trained machine learning models.

    The PredictionPipeline class orchestrates the execution of pipeline stages
    for making predictions with trained machine learning models. It includes
    stages for data loading, validation, feature engineering, model loading,
    and prediction.

    Attributes:
        config: Configuration for the pipeline.
        container: DI container for resolving dependencies.
        stages: List of pipeline stages to execute.
    """

    def _initialize_stages(self) -> None:
        """
        Initialize the prediction pipeline stages.

        This method adds the standard stages for a prediction pipeline:
        1. Data loading
        2. Data validation
        3. Feature engineering
        4. Model loading
        5. Prediction
        """
        logger.info("Initializing prediction pipeline stages")

        # Add data loading stage
        self.add_stage(self.container.resolve(ConfigurableDataLoadingStage))

        # Add validation stage
        self.add_stage(self.container.resolve(ConfigDrivenValidationStage))

        # Add feature engineering stage
        self.add_stage(self.container.resolve(SimpleFeatureEngineeringStage))

        # Add model loading stage
        self.add_stage(self.container.resolve(ModelLoadingStage))

        # Add prediction stage
        # Use probability prediction if specified in config
        if self.config.get("return_probabilities", False):
            from nexusml.src.pipeline.stages.prediction import (
                ProbabilityPredictionStage,
            )

            self.add_stage(self.container.resolve(ProbabilityPredictionStage))
        else:
            self.add_stage(self.container.resolve(StandardPredictionStage))

        # Add output saving stage if output_path is provided in config
        if self.config.get("output_path"):
            from nexusml.src.pipeline.stages.output import OutputSavingStage

            self.add_stage(self.container.resolve(OutputSavingStage))
        else:
            logger.warning(
                "No output_path specified in config, results will not be saved"
            )

        logger.info(f"Prediction pipeline initialized with {len(self.stages)} stages")
