"""
Generator module for NexusML.

This module provides utilities for generating data for the NexusML module,
including OmniClass data extraction and description generation.
"""

from nexusml.ingest.generator.omniclass import extract_omniclass_data
from nexusml.ingest.generator.omniclass_description_generator import (
    AnthropicClient,
    BatchProcessor,
    OmniClassDescriptionGenerator,
    generate_descriptions,
)

__all__ = [
    'extract_omniclass_data',
    'OmniClassDescriptionGenerator',
    'generate_descriptions',
    'BatchProcessor',
    'AnthropicClient',
]