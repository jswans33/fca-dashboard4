"""
Compatibility Module for Model Building

This module provides compatibility functions for the existing code that depends
on the old model building API. It bridges the gap between the old and new APIs.
"""

import logging
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline

from nexusml.core.model_building.builders.random_forest import RandomForestBuilder

# Set up logging
logger = logging.getLogger(__name__)


def build_enhanced_model(
    sampling_strategy: str = "direct",
    feature_config_path: Optional[str] = None,
    **kwargs,
) -> Pipeline:
    """
    Build an enhanced model with configurable sampling strategy.

    This function is provided for backward compatibility with the existing code.
    It delegates to the RandomForestBuilder to build the model.

    Args:
        sampling_strategy: Sampling strategy to use ("direct" is the only supported option for now)
        feature_config_path: Path to the feature configuration file. If None, uses the default path.
        **kwargs: Additional parameters for the model

    Returns:
        Pipeline: Scikit-learn pipeline with feature engineering, preprocessor and classifier
    """
    logger.info("Building enhanced model with RandomForestBuilder")
    
    # Create a RandomForestBuilder
    builder = RandomForestBuilder()
    
    # Update the builder's configuration with the provided kwargs
    config = builder.get_config()
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
    
    # Set the updated configuration
    builder.set_config(config)
    
    # Build the model
    model = builder.build_model()
    
    logger.info("Enhanced model built successfully")
    return model


def optimize_hyperparameters(model: Pipeline, x_train, y_train) -> Pipeline:
    """
    Optimize hyperparameters for better handling of all classes including "Other".

    This function is provided for backward compatibility with the existing code.
    It delegates to the RandomForestBuilder to optimize the hyperparameters.

    Args:
        model (Pipeline): Model pipeline to optimize
        x_train: Training features
        y_train: Training targets

    Returns:
        Pipeline: Optimized pipeline
    """
    logger.info("Optimizing hyperparameters with RandomForestBuilder")
    
    # Create a RandomForestBuilder
    builder = RandomForestBuilder()
    
    # Get the parameter grid
    param_grid = builder.get_param_grid()
    
    # Optimize hyperparameters
    optimized_model = builder.optimize_hyperparameters(
        model, x_train, y_train, param_grid=param_grid
    )
    
    logger.info("Hyperparameters optimized successfully")
    return optimized_model