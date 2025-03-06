"""
Data ingestion functionality for NexusML.
"""

# Import ingest functions to expose at the package level
# These will be populated as we migrate the ingest functionality
from nexusml.ingest.generator import (
    AnthropicClient,
    BatchProcessor,
    OmniClassDescriptionGenerator,
    extract_omniclass_data,
    generate_descriptions,
)

__all__ = [
    'extract_omniclass_data',
    'OmniClassDescriptionGenerator',
    'generate_descriptions',
    'BatchProcessor',
    'AnthropicClient',
]