"""
Model Training Base Module

This module re-exports the ModelTrainer class from the model_building.base module
to maintain backward compatibility and proper import structure.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.core.model_building.base import (
    ModelTrainer,
    BaseModelTrainer,
    BaseConfigurableModelTrainer,
    ConfigurableModelTrainer,
)

# Set up logging
logger = logging.getLogger(__name__)

# Re-export the ModelTrainer classes
__all__ = [
    "ModelTrainer",
    "BaseModelTrainer",
    "BaseConfigurableModelTrainer",
    "ConfigurableModelTrainer",
]