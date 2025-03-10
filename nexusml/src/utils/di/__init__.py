"""
Dependency Injection module for NexusML.

This module provides a dependency injection container system for the NexusML suite,
allowing for better testability, extensibility, and adherence to SOLID principles.

The module includes:
- DIContainer: A container for registering and resolving dependencies
- ContainerProvider: A singleton provider for accessing the container
- Decorators: Utilities for dependency injection and registration
"""

from nexusml.core.di.container import DIContainer
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.di.provider import ContainerProvider

__all__ = ["DIContainer", "ContainerProvider", "inject", "injectable"]
