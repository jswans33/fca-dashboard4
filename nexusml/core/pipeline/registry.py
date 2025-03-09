"""
Component Registry Module

This module provides the ComponentRegistry class, which is responsible for
registering and retrieving component implementations for the pipeline system.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, cast

# Set up logging
logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentRegistryError(Exception):
    """Exception raised for errors in the ComponentRegistry."""
    pass


class ComponentRegistry:
    """
    Registry for pipeline components.

    The ComponentRegistry class is responsible for registering and retrieving
    component implementations for the pipeline system. It allows registering
    components by type and name, and retrieving them later.

    Attributes:
        _components: Dictionary mapping component types to dictionaries of name-to-class mappings.
    """

    def __init__(self):
        """
        Initialize a new ComponentRegistry.
        """
        self._components: Dict[str, Dict[str, Type[Any]]] = {}
        logger.info("ComponentRegistry initialized")

    def register(self, component_type: str, name: str, component_class: Type[Any]) -> None:
        """
        Register a component.

        Args:
            component_type: Type of component (e.g., "stage", "transformer", "pipeline").
            name: Name of the component.
            component_class: Component class to register.

        Raises:
            ComponentRegistryError: If a component with the same type and name is already registered.
        """
        # Initialize the component type dictionary if it doesn't exist
        if component_type not in self._components:
            self._components[component_type] = {}

        # Check if the component is already registered
        if name in self._components[component_type]:
            raise ComponentRegistryError(
                f"Component of type '{component_type}' with name '{name}' is already registered"
            )

        # Register the component
        self._components[component_type][name] = component_class
        logger.debug(f"Registered component: {component_type}/{name}")

    def get(self, component_type: str, name: str) -> Optional[Type[Any]]:
        """
        Get a component by type and name.

        Args:
            component_type: Type of component.
            name: Name of the component.

        Returns:
            Component class if found, None otherwise.
        """
        # Check if the component type exists
        if component_type not in self._components:
            return None

        # Return the component if it exists
        return self._components[component_type].get(name)

    def get_all(self, component_type: str) -> Dict[str, Type[Any]]:
        """
        Get all components of a specific type.

        Args:
            component_type: Type of components to get.

        Returns:
            Dictionary mapping component names to component classes.
        """
        return self._components.get(component_type, {}).copy()

    def get_types(self) -> List[str]:
        """
        Get all registered component types.

        Returns:
            List of component types.
        """
        return list(self._components.keys())

    def get_names(self, component_type: str) -> List[str]:
        """
        Get all registered component names for a specific type.

        Args:
            component_type: Type of components to get names for.

        Returns:
            List of component names.
        """
        return list(self._components.get(component_type, {}).keys())

    def has_type(self, component_type: str) -> bool:
        """
        Check if a component type is registered.

        Args:
            component_type: Type of component to check.

        Returns:
            True if the component type is registered, False otherwise.
        """
        return component_type in self._components

    def has_component(self, component_type: str, name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            component_type: Type of component to check.
            name: Name of the component to check.

        Returns:
            True if the component is registered, False otherwise.
        """
        return component_type in self._components and name in self._components[component_type]

    def register_from_module(self, module: Any) -> None:
        """
        Register components from a module.

        This method looks for a `register_components` function in the module
        and calls it with this registry as an argument.

        Args:
            module: Module to register components from.

        Raises:
            ComponentRegistryError: If the module doesn't have a `register_components` function.
        """
        if not hasattr(module, "register_components"):
            raise ComponentRegistryError(
                f"Module {module.__name__} doesn't have a register_components function"
            )

        try:
            module.register_components(self)
            logger.info(f"Registered components from module {module.__name__}")
        except Exception as e:
            raise ComponentRegistryError(
                f"Error registering components from module {module.__name__}: {str(e)}"
            ) from e

    def clear(self) -> None:
        """
        Clear all registered components.
        """
        self._components.clear()
        logger.info("Cleared all registered components")

    def clear_type(self, component_type: str) -> None:
        """
        Clear all registered components of a specific type.

        Args:
            component_type: Type of components to clear.
        """
        if component_type in self._components:
            self._components[component_type].clear()
            logger.info(f"Cleared all registered components of type {component_type}")
