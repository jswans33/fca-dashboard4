"""
Model Building Interfaces Module

This module defines the interfaces for model building components in the NexusML suite.
Each interface follows the Interface Segregation Principle (ISP) from SOLID,
defining a minimal set of methods that components must implement.
"""

import abc
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline


class ModelBuilder(abc.ABC):
    """
    Interface for model building components.
    
    Responsible for creating and configuring machine learning models.
    """
    
    @abc.abstractmethod
    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.
        
        Args:
            **kwargs: Configuration parameters for the model.
            
        Returns:
            Configured model pipeline.
            
        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        pass
    
    @abc.abstractmethod
    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.
        
        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for hyperparameter optimization.
            
        Returns:
            Optimized model pipeline.
            
        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        pass
    
    @abc.abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the model.
        
        Returns:
            Dictionary of default parameters.
        """
        pass
    
    @abc.abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get the parameter grid for hyperparameter optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of values to try.
        """
        pass


class ConfigurableModelBuilder(ModelBuilder):
    """
    Interface for configurable model builders.
    
    Extends the ModelBuilder interface with methods for configuration.
    """
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the model builder.
        
        Returns:
            Dictionary containing the configuration.
        """
        pass
    
    @abc.abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the model builder.
        
        Args:
            config: Configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        pass
    
    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model builder configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if the configuration is valid, False otherwise.
        """
        pass


class ModelTrainer(abc.ABC):
    """
    Interface for model training components.
    
    Responsible for training machine learning models on prepared data.
    """
    
    @abc.abstractmethod
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
        pass
    
    @abc.abstractmethod
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
            
        Raises:
            ValueError: If cross-validation cannot be performed.
        """
        pass


class ConfigurableModelTrainer(ModelTrainer):
    """
    Interface for configurable model trainers.
    
    Extends the ModelTrainer interface with methods for configuration.
    """
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the model trainer.
        
        Returns:
            Dictionary containing the configuration.
        """
        pass
    
    @abc.abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration for the model trainer.
        
        Args:
            config: Configuration dictionary.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        pass
    
    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model trainer configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if the configuration is valid, False otherwise.
        """
        pass


class HyperparameterOptimizer(abc.ABC):
    """
    Interface for hyperparameter optimization components.
    
    Responsible for optimizing model hyperparameters.
    """
    
    @abc.abstractmethod
    def optimize(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.
        
        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for optimization.
            
        Returns:
            Optimized model pipeline.
            
        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        pass
    
    @abc.abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        pass
    
    @abc.abstractmethod
    def get_best_score(self) -> float:
        """
        Get the best score achieved during optimization.
        
        Returns:
            Best score.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        pass


class ModelEvaluator(abc.ABC):
    """
    Interface for model evaluation components.
    
    Responsible for evaluating trained models and analyzing their performance.
    """
    
    @abc.abstractmethod
    def evaluate(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
            
        Raises:
            ValueError: If the model cannot be evaluated.
        """
        pass
    
    @abc.abstractmethod
    def analyze_predictions(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze model predictions in detail.
        
        Args:
            model: Trained model pipeline.
            x_test: Test features.
            y_test: Test targets.
            y_pred: Model predictions.
            **kwargs: Additional arguments for analysis.
            
        Returns:
            Dictionary of analysis results.
            
        Raises:
            ValueError: If predictions cannot be analyzed.
        """
        pass


class ModelSerializer(abc.ABC):
    """
    Interface for model serialization components.
    
    Responsible for saving and loading trained models.
    """
    
    @abc.abstractmethod
    def save_model(self, model: Pipeline, path: str, **kwargs) -> None:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            **kwargs: Additional arguments for saving.
            
        Raises:
            IOError: If the model cannot be saved.
        """
        pass
    
    @abc.abstractmethod
    def load_model(self, path: str, **kwargs) -> Pipeline:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model.
            **kwargs: Additional arguments for loading.
            
        Returns:
            Loaded model pipeline.
            
        Raises:
            IOError: If the model cannot be loaded.
            ValueError: If the loaded file is not a valid model.
        """
        pass