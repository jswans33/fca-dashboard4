"""
NexusML - Modern machine learning classification engine
"""

__version__ = "0.1.0"

# Import key functionality to expose at the top level
from nexusml.ingest import (
    AnthropicClient,
    BatchProcessor,
    OmniClassDescriptionGenerator,
    extract_omniclass_data,
    generate_descriptions,
)

__all__ = [
    "extract_omniclass_data",
    "OmniClassDescriptionGenerator",
    "generate_descriptions",
    "BatchProcessor",
    "AnthropicClient",
]
