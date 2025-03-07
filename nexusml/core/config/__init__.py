"""
Configuration system for NexusML.

This package provides a unified configuration system for the NexusML suite,
centralizing all settings and providing validation through Pydantic models.

Note: The legacy configuration files are maintained for backward compatibility
and are planned for removal in future work chunks. Once all code is updated to
use the new unified configuration system, these files will be removed.
"""

from nexusml.core.config.configuration import NexusMLConfig
from nexusml.core.config.provider import ConfigurationProvider

__all__ = ["NexusMLConfig", "ConfigurationProvider"]
