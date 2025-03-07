"""
Pipeline Adapters Module

This module provides adapter classes that maintain backward compatibility
with the existing code while delegating to the new components that use
the configuration system.
"""

from nexusml.core.pipeline.adapters.data_adapter import (
    DataComponentFactory,
    LegacyDataLoaderAdapter,
    LegacyDataPreprocessorAdapter,
)
from nexusml.core.pipeline.adapters.feature_adapter import (
    GenericFeatureEngineerAdapter,
    enhanced_masterformat_mapping_adapter,
)
from nexusml.core.pipeline.adapters.model_adapter import (
    LegacyModelBuilderAdapter,
    LegacyModelEvaluatorAdapter,
    LegacyModelSerializerAdapter,
    LegacyModelTrainerAdapter,
    ModelComponentFactory,
)

__all__ = [
    # Data adapters
    "LegacyDataLoaderAdapter",
    "LegacyDataPreprocessorAdapter",
    "DataComponentFactory",
    # Feature adapters
    "GenericFeatureEngineerAdapter",
    "enhanced_masterformat_mapping_adapter",
    # Model adapters
    "LegacyModelBuilderAdapter",
    "LegacyModelTrainerAdapter",
    "LegacyModelEvaluatorAdapter",
    "LegacyModelSerializerAdapter",
    "ModelComponentFactory",
]
