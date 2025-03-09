"""
Model Building Package

This package provides model building components for the NexusML suite.
It includes interfaces, base classes, and implementations for building
machine learning models.
"""

# Import interfaces
from nexusml.core.model_building.interfaces import (
    ModelBuilder,
    ConfigurableModelBuilder,
    ModelTrainer,
    ConfigurableModelTrainer,
    HyperparameterOptimizer,
    ModelEvaluator,
    ModelSerializer,
)

# Import base classes
from nexusml.core.model_building.base import (
    BaseModelBuilder,
    BaseConfigurableModelBuilder,
    BaseModelTrainer,
    BaseConfigurableModelTrainer,
    BaseHyperparameterOptimizer,
    BaseModelEvaluator,
    BaseModelSerializer,
)

# Import model builders
from nexusml.core.model_building.builders.random_forest import RandomForestBuilder
from nexusml.core.model_building.builders.gradient_boosting import GradientBoostingBuilder
from nexusml.core.model_building.builders.ensemble import EnsembleBuilder

# Import compatibility functions
from nexusml.core.model_building.compatibility import (
    build_enhanced_model,
    optimize_hyperparameters,
)

# Define the public API
__all__ = [
    # Interfaces
    "ModelBuilder",
    "ConfigurableModelBuilder",
    "ModelTrainer",
    "ConfigurableModelTrainer",
    "HyperparameterOptimizer",
    "ModelEvaluator",
    "ModelSerializer",
    
    # Base classes
    "BaseModelBuilder",
    "BaseConfigurableModelBuilder",
    "BaseModelTrainer",
    "BaseConfigurableModelTrainer",
    "BaseHyperparameterOptimizer",
    "BaseModelEvaluator",
    "BaseModelSerializer",
    
    # Model builders
    "RandomForestBuilder",
    "GradientBoostingBuilder",
    "EnsembleBuilder",
    
    # Compatibility functions
    "build_enhanced_model",
    "optimize_hyperparameters",
]