"""
Pipeline Package

This package contains the interfaces, base implementations, adapters, and stages for the NexusML pipeline.
"""

# Import interfaces
# Import base implementations
from nexusml.src.pipeline.base import (
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
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.factory import PipelineFactory
from nexusml.src.pipeline.interfaces import (
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
from nexusml.src.pipeline.orchestrator import PipelineOrchestrator
from nexusml.src.pipeline.registry import ComponentRegistry

# Import stages package
from nexusml.src.pipeline.stages import *

# Define __all__ to control what gets imported with "from nexusml.src.pipeline import *"
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
