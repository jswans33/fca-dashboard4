"""
Cross-Validation Model Trainer Module

This module provides a CrossValidationTrainer implementation that trains
machine learning models using cross-validation procedures.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.model_building.base import BaseConfigurableModelTrainer

# Set up logging
logger = logging.getLogger(__name__)


@injectable
class CrossValidationTrainer(BaseConfigurableModelTrainer):
    """
    Implementation of the ModelTrainer interface for cross-validation training.
    
    This class trains models using cross-validation procedures based on
    configuration provided by the ConfigurationProvider.
    """
    
    def __init__(
        self,
        name: str = "CrossValidationTrainer",
        description: str = "Cross-validation model trainer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the CrossValidationTrainer.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        self._cv_results: Dict[str, np.ndarray] = {}
        self._cv_predictions: Optional[np.ndarray] = None
        logger.info(f"Initialized {name}")
    
    def train(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Train a model on the provided data using cross-validation.
        
        This method performs cross-validation on the model and then trains
        the model on the full training set.
        
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
            logger.info(f"Training model with cross-validation on {len(x_train)} samples")
            
            # Extract cross-validation parameters from config and kwargs
            cv = kwargs.get("cv", self.config.get("cv", 5))
            scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))
            return_train_score = kwargs.get("return_train_score", self.config.get("return_train_score", True))
            
            # Log training information
            logger.info(f"X_train shape: {x_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_train columns: {x_train.columns.tolist()}")
            logger.info(f"y_train columns: {y_train.columns.tolist()}")
            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            
            # Perform cross-validation
            if verbose:
                print(f"Performing {cv}-fold cross-validation...")
                print(f"X_train shape: {x_train.shape}")
                print(f"y_train shape: {y_train.shape}")
            
            # Perform cross-validation and store results
            self._cv_results = cross_validate(
                model, x_train, y_train, cv=cv, scoring=scoring, 
                return_train_score=return_train_score, verbose=verbose
            )
            
            # Get cross-validation predictions
            self._cv_predictions = cross_val_predict(
                model, x_train, y_train, cv=cv, verbose=verbose
            )
            
            # Print cross-validation results
            if verbose:
                print("\nCross-validation results:")
                print(f"Test score: {np.mean(self._cv_results['test_score']):.4f} ± {np.std(self._cv_results['test_score']):.4f}")
                if return_train_score:
                    print(f"Train score: {np.mean(self._cv_results['train_score']):.4f} ± {np.std(self._cv_results['train_score']):.4f}")
                print(f"Fit time: {np.mean(self._cv_results['fit_time']):.4f} seconds")
                print(f"Score time: {np.mean(self._cv_results['score_time']):.4f} seconds")
            
            # Log cross-validation results
            logger.info(f"Cross-validation test score: {np.mean(self._cv_results['test_score']):.4f} ± {np.std(self._cv_results['test_score']):.4f}")
            if return_train_score:
                logger.info(f"Cross-validation train score: {np.mean(self._cv_results['train_score']):.4f} ± {np.std(self._cv_results['train_score']):.4f}")
            
            # Train the model on the full training set
            if verbose:
                print("\nTraining model on full training set...")
            
            # Call the parent class's train method to fit the model on the full training set
            trained_model = super().train(model, x_train, y_train, **kwargs)
            
            if verbose:
                print("Model training completed")
            
            logger.info("Model training with cross-validation completed successfully")
            return trained_model
        
        except Exception as e:
            logger.error(f"Error training model with cross-validation: {str(e)}")
            raise ValueError(f"Error training model with cross-validation: {str(e)}") from e
    
    def cross_validate(
        self, model: Pipeline, x: pd.DataFrame, y: pd.DataFrame, **kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model pipeline to validate.
            x: Feature data.
            y: Target data.
            **kwargs: Additional arguments for cross-validation.
            
        Returns:
            Dictionary of validation metrics.
        """
        try:
            logger.info(f"Performing cross-validation on {len(x)} samples")
            
            # Extract cross-validation parameters from config and kwargs
            cv = kwargs.get("cv", self.config.get("cv", 5))
            scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))
            return_train_score = kwargs.get("return_train_score", self.config.get("return_train_score", True))
            
            # Log cross-validation information
            logger.info(f"X shape: {x.shape}")
            logger.info(f"y shape: {y.shape}")
            logger.info(f"X columns: {x.columns.tolist()}")
            logger.info(f"y columns: {y.columns.tolist()}")
            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            
            # Perform cross-validation
            if verbose:
                print(f"Performing {cv}-fold cross-validation...")
                print(f"X shape: {x.shape}")
                print(f"y shape: {y.shape}")
            
            # Perform cross-validation
            cv_results = cross_validate(
                model, x, y, cv=cv, scoring=scoring, 
                return_train_score=return_train_score, verbose=verbose
            )
            
            # Convert numpy arrays to lists for serialization
            result = {
                "test_score": cv_results["test_score"].tolist(),
                "fit_time": cv_results["fit_time"].tolist(),
                "score_time": cv_results["score_time"].tolist(),
            }
            
            if return_train_score:
                result["train_score"] = cv_results["train_score"].tolist()
            
            # Print cross-validation results
            if verbose:
                print("\nCross-validation results:")
                print(f"Test score: {np.mean(result['test_score']):.4f} ± {np.std(result['test_score']):.4f}")
                if return_train_score:
                    print(f"Train score: {np.mean(result['train_score']):.4f} ± {np.std(result['train_score']):.4f}")
                print(f"Fit time: {np.mean(result['fit_time']):.4f} seconds")
                print(f"Score time: {np.mean(result['score_time']):.4f} seconds")
            
            # Log cross-validation results
            logger.info(f"Cross-validation test score: {np.mean(result['test_score']):.4f} ± {np.std(result['test_score']):.4f}")
            if return_train_score:
                logger.info(f"Cross-validation train score: {np.mean(result['train_score']):.4f} ± {np.std(result['train_score']):.4f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error performing cross-validation: {str(e)}")
            raise ValueError(f"Error performing cross-validation: {str(e)}") from e
    
    def get_cv_results(self) -> Dict[str, List[float]]:
        """
        Get the cross-validation results from the last training run.
        
        Returns:
            Dictionary of cross-validation metrics.
            
        Raises:
            ValueError: If cross-validation has not been performed.
        """
        if not self._cv_results:
            raise ValueError("Cross-validation has not been performed")
        
        # Convert numpy arrays to lists for serialization
        result: Dict[str, List[float]] = {
            "test_score": self._cv_results["test_score"].tolist(),
            "fit_time": self._cv_results["fit_time"].tolist(),
            "score_time": self._cv_results["score_time"].tolist(),
        }
        
        if "train_score" in self._cv_results:
            result["train_score"] = self._cv_results["train_score"].tolist()
        
        return result
    
    def get_cv_predictions(self) -> np.ndarray:
        """
        Get the cross-validation predictions from the last training run.
        
        Returns:
            Array of cross-validation predictions.
            
        Raises:
            ValueError: If cross-validation has not been performed.
        """
        if self._cv_predictions is None:
            raise ValueError("Cross-validation has not been performed")
        
        return self._cv_predictions
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model trainer configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Check if the required parameters exist
        required_params = ["cv", "scoring", "random_state"]
        for param in required_params:
            if param not in config:
                logger.warning(f"Missing '{param}' parameter in configuration")
                return False
        
        return True