"""
Standard Model Trainer Module

This module provides a StandardModelTrainer implementation that trains
machine learning models using standard training procedures.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.model_building.base import BaseConfigurableModelTrainer

# Set up logging
logger = logging.getLogger(__name__)


@injectable
class StandardModelTrainer(BaseConfigurableModelTrainer):
    """
    Implementation of the ModelTrainer interface for standard model training.
    
    This class trains models using standard training procedures based on
    configuration provided by the ConfigurationProvider.
    """
    
    def __init__(
        self,
        name: str = "StandardModelTrainer",
        description: str = "Standard model trainer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the StandardModelTrainer.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        logger.info(f"Initialized {name}")
    
    def train(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model on the provided data.
        
        Args:
            model: Model pipeline to train.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for training.
            
        Returns:
            Trained model pipeline.
            
        Raises:
            ValueError: If the model cannot be trained.
        """
        try:
            logger.info(f"Training model with {len(x_train)} samples")
            
            # Extract training parameters from config and kwargs
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))
            
            # Log training information
            logger.info(f"X_train shape: {x_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_train columns: {x_train.columns.tolist()}")
            logger.info(f"y_train columns: {y_train.columns.tolist()}")
            
            # Train the model
            if verbose:
                print(f"Training model with {len(x_train)} samples...")
                print(f"X_train shape: {x_train.shape}")
                print(f"y_train shape: {y_train.shape}")
            
            # Call the parent class's train method to fit the model
            trained_model = super().train(model, x_train, y_train, **kwargs)
            
            if verbose:
                print("Model training completed")
            
            logger.info("Model training completed successfully")
            return trained_model
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise ValueError(f"Error training model: {str(e)}") from e
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model trainer configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Check if the required parameters exist
        required_params = ["random_state"]
        for param in required_params:
            if param not in config:
                logger.warning(f"Missing '{param}' parameter in configuration")
                return False
        
        return True