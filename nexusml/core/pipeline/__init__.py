"""
Pipeline Package

This package contains the interfaces, base implementations, adapters, and stages for the NexusML pipeline.
"""

# Import interfaces
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    PipelineComponent,
    Predictor,
)

# Import base implementations
from nexusml.core.pipeline.base import (
    BaseDataLoader,
    BaseDataPreprocessor,
    BaseFeatureEngineer,
    BaseModelBuilder,
    BaseModelEvaluator,
    BaseModelSerializer,
    BaseModelTrainer,
    BasePipelineComponent,
    BasePredictor,
)

# Import context and factory
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry

# Import stages package
from nexusml.core.pipeline.stages import *

# Define __all__ to control what gets imported with "from nexusml.core.pipeline import *"
__all__ = [
    # Interfaces
    "DataLoader",
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelBuilder",
    "ModelEvaluator",
    "ModelSerializer",
    "ModelTrainer",
    "PipelineComponent",
    "Predictor",
    
    # Base implementations
    "BaseDataLoader",
    "BaseDataPreprocessor",
    "BaseFeatureEngineer",
    "BaseModelBuilder",
    "BaseModelEvaluator",
    "BaseModelSerializer",
    "BaseModelTrainer",
    "BasePipelineComponent",
    "BasePredictor",
    
    # Context and factory
    "PipelineContext",
    "PipelineFactory",
    "PipelineOrchestrator",
    "ComponentRegistry",
]
