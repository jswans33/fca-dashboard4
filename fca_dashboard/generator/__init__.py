"""
Compatibility layer for the generator module.

This module re-exports the generator functionality from the nexusml module
to maintain backward compatibility with existing code.
"""

# Re-export from nexusml
from nexusml import (
    AnthropicClient,
    BatchProcessor,
    OmniClassDescriptionGenerator,
    extract_omniclass_data,
    generate_descriptions,
)

# For backward compatibility
from nexusml.ingest.generator.omniclass_description_generator import (
    ApiClientError,
    DescriptionGeneratorError,
)

__all__ = [
    "extract_omniclass_data",
    "OmniClassDescriptionGenerator",
    "generate_descriptions",
    "BatchProcessor",
    "AnthropicClient",
    "ApiClientError",
    "DescriptionGeneratorError",
]
