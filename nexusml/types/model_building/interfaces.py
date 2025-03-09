"""
Type definitions for the model building interfaces.

This module provides type hints for the model building interfaces to improve type safety.
"""

import abc
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union

import pandas as pd
from sklearn.pipeline import Pipeline

# Type variable for generic types
T = TypeVar('T')


class ModelBuilder(Protocol):
    """
    Interface for model building components.
    
    Responsible for creating and configuring machine learning models.
    """
    
    def build_model(self, **kwargs: Any) -> Pipeline: ...
    
    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs: Any
    ) -> Pipeline: ...
    
    def get_default_parameters(self) -> Dict[str, Any]: ...
    
    def get_param_grid(self) -> Dict[str, List[Any]]: ...


class ConfigurableModelBuilder(ModelBuilder, Protocol):
    """
    Interface for configurable model builders.
    
    Extends the ModelBuilder interface with methods for configuration.
    """
    
    def get_config(self) -> Dict[str, Any]: ...
    
    def set_config(self, config: Dict[str, Any]) -> None: ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool: ...


class ModelTrainer(Protocol):
    """
    Interface for model training components.
    
    Responsible for training machine learning models on prepared data.
    """
    
    def train(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs: Any
    ) -> Pipeline: ...
    
    def cross_validate(
        self, model: Pipeline, x: pd.DataFrame, y: pd.DataFrame, **kwargs: Any
    ) -> Dict[str, List[float]]: ...


class ConfigurableModelTrainer(ModelTrainer, Protocol):
    """
    Interface for configurable model trainers.
    
    Extends the ModelTrainer interface with methods for configuration.
    """
    
    def get_config(self) -> Dict[str, Any]: ...
    
    def set_config(self, config: Dict[str, Any]) -> None: ...
    
    def validate_config(self, config: Dict[str, Any]) -> bool: ...


class HyperparameterOptimizer(Protocol):
    """
    Interface for hyperparameter optimization components.
    
    Responsible for optimizing model hyperparameters.
    """
    
    def optimize(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs: Any
    ) -> Pipeline: ...
    
    def get_best_params(self) -> Dict[str, Any]: ...
    
    def get_best_score(self) -> float: ...


class ModelEvaluator(Protocol):
    """
    Interface for model evaluation components.
    
    Responsible for evaluating trained models and analyzing their performance.
    """
    
    def evaluate(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs: Any
    ) -> Dict[str, Any]: ...
    
    def analyze_predictions(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...


class ModelSerializer(Protocol):
    """
    Interface for model serialization components.
    
    Responsible for saving and loading trained models.
    """
    
    def save_model(self, model: Pipeline, path: str, **kwargs: Any) -> None: ...
    
    def load_model(self, path: str, **kwargs: Any) -> Pipeline: ...