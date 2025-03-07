"""
Component Registry Module

This module provides the ComponentRegistry class, which is responsible for
registering and retrieving component implementations.
"""

from typing import Any, Dict, Generic, Optional, Type, TypeVar, cast

T = TypeVar("T")


class ComponentRegistryError(Exception):
    """Exception raised for errors in the ComponentRegistry."""

    pass


class ComponentRegistry:
    """
    Registry for pipeline component implementations.

    This class manages the registration and retrieval of component implementations.
    It allows registering multiple implementations of the same component type
    and setting a default implementation for each type.

    Example:
        >>> registry = ComponentRegistry()
        >>> registry.register(DataLoader, "csv", CSVDataLoader)
        >>> registry.register(DataLoader, "excel", ExcelDataLoader)
        >>> registry.set_default_implementation(DataLoader, "csv")
        >>> loader = registry.get_default_implementation(DataLoader)
        >>> # Use the loader...
    """

    def __init__(self):
        """Initialize a new ComponentRegistry."""
        self._registry: Dict[Type, Dict[str, Type]] = {}
        self._defaults: Dict[Type, str] = {}

    def register(
        self, component_type: Type[T], name: str, implementation: Type[T]
    ) -> None:
        """
        Register a component implementation.

        Args:
            component_type: The interface or base class of the component.
            name: A unique name for this implementation.
            implementation: The implementation class.

        Raises:
            ComponentRegistryError: If an implementation with the same name already exists.
        """
        if component_type not in self._registry:
            self._registry[component_type] = {}

        if name in self._registry[component_type]:
            raise ComponentRegistryError(
                f"Implementation '{name}' for {component_type.__name__} already exists"
            )

        self._registry[component_type][name] = implementation

    def get_implementation(self, component_type: Type[T], name: str) -> Type[T]:
        """
        Get a specific component implementation.

        Args:
            component_type: The interface or base class of the component.
            name: The name of the implementation to retrieve.

        Returns:
            The implementation class.

        Raises:
            ComponentRegistryError: If the implementation does not exist.
        """
        if (
            component_type not in self._registry
            or name not in self._registry[component_type]
        ):
            raise ComponentRegistryError(
                f"Implementation '{name}' for {component_type.__name__} not found"
            )

        return self._registry[component_type][name]

    def get_implementations(self, component_type: Type[T]) -> Dict[str, Type[T]]:
        """
        Get all implementations of a component type.

        Args:
            component_type: The interface or base class of the component.

        Returns:
            A dictionary mapping implementation names to implementation classes.
        """
        if component_type not in self._registry:
            return {}

        return self._registry[component_type]

    def set_default_implementation(self, component_type: Type[T], name: str) -> None:
        """
        Set the default implementation for a component type.

        Args:
            component_type: The interface or base class of the component.
            name: The name of the implementation to set as default.

        Raises:
            ComponentRegistryError: If the implementation does not exist.
        """
        # Verify the implementation exists
        self.get_implementation(component_type, name)

        # Set as default
        self._defaults[component_type] = name

    def get_default_implementation(self, component_type: Type[T]) -> Type[T]:
        """
        Get the default implementation for a component type.

        Args:
            component_type: The interface or base class of the component.

        Returns:
            The default implementation class.

        Raises:
            ComponentRegistryError: If no default implementation is set.
        """
        if component_type not in self._defaults:
            raise ComponentRegistryError(
                f"No default implementation set for {component_type.__name__}"
            )

        name = self._defaults[component_type]
        return self.get_implementation(component_type, name)

    def has_implementation(self, component_type: Type, name: str) -> bool:
        """
        Check if an implementation exists.

        Args:
            component_type: The interface or base class of the component.
            name: The name of the implementation to check.

        Returns:
            True if the implementation exists, False otherwise.
        """
        return (
            component_type in self._registry and name in self._registry[component_type]
        )

    def clear_implementations(self, component_type: Type) -> None:
        """
        Clear all implementations of a component type.

        Args:
            component_type: The interface or base class of the component.
        """
        if component_type in self._registry:
            self._registry[component_type] = {}

        if component_type in self._defaults:
            del self._defaults[component_type]
