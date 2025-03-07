"""
Pipeline Adapters Module

This module provides adapter classes that maintain backward compatibility
with the existing code while delegating to the new components that use
the configuration system.
"""

from nexusml.core.pipeline.adapters.feature_adapter import (
    GenericFeatureEngineerAdapter,
    enhanced_masterformat_mapping_adapter,
)

__all__ = [
    "GenericFeatureEngineerAdapter",
    "enhanced_masterformat_mapping_adapter",
]
