"""
Model Training Package

This package provides model training components for the NexusML suite.
It includes interfaces, base classes, and implementations for training
machine learning models.
"""

# Import interfaces from model_building
from nexusml.core.model_building.interfaces import (
    ModelTrainer,
    ConfigurableModelTrainer,
    HyperparameterOptimizer,
)

# Import base classes from model_building
from nexusml.core.model_building.base import (
    BaseModelTrainer,
    BaseConfigurableModelTrainer,
    BaseHyperparameterOptimizer,
)

# Import trainers
from nexusml.core.model_training.trainers.standard import StandardModelTrainer
from nexusml.core.model_training.trainers.cross_validation import CrossValidationTrainer
from nexusml.core.model_training.trainers.hyperparameter_optimizer import (
    GridSearchOptimizer,
    RandomizedSearchOptimizer,
)

# Define the public API
__all__ = [
    # Interfaces
    "ModelTrainer",
    "ConfigurableModelTrainer",
    "HyperparameterOptimizer",
    
    # Base classes
    "BaseModelTrainer",
    "BaseConfigurableModelTrainer",
    "BaseHyperparameterOptimizer",
    
    # Trainers
    "StandardModelTrainer",
    "CrossValidationTrainer",
    "GridSearchOptimizer",
    "RandomizedSearchOptimizer",
]