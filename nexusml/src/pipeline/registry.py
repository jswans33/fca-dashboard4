"""
Component Registry Module

This module provides the ComponentRegistry class, which is responsible for
registering and retrieving component implementations for the pipeline system.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

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
        _default_implementations: Dictionary mapping component types to default implementation names.
    """

    def __init__(self):
        """
        Initialize a new ComponentRegistry.
        """
        self._components: Dict[str, Dict[str, Type[Any]]] = {}
        self._default_implementations: Dict[str, str] = {}
        logger.info("ComponentRegistry initialized")

    def register(self, component_type: Union[str, Type[Any]], name: str, component_class: Type[Any]) -> None:
        """
        Register a component.

        Args:
            component_type: Type of component (e.g., "stage", "transformer", "pipeline") or a Type object.
            name: Name of the component.
            component_class: Component class to register.

        Raises:
            ComponentRegistryError: If a component with the same type and name is already registered.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
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

    def get(self, component_type: Union[str, Type[Any]], name: str) -> Optional[Type[Any]]:
        """
        Get a component by type and name.

        Args:
            component_type: Type of component or a Type object.
            name: Name of the component.

        Returns:
            Component class if found, None otherwise.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
        # Check if the component type exists
        if component_type not in self._components:
            return None

        # Return the component if it exists
        return self._components[component_type].get(name)

    def get_all(self, component_type: Union[str, Type[Any]]) -> Dict[str, Type[Any]]:
        """
        Get all components of a specific type.

        Args:
            component_type: Type of components to get or a Type object.

        Returns:
            Dictionary mapping component names to component classes.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
        return self._components.get(component_type, {}).copy()

    def get_types(self) -> List[str]:
        """
        Get all registered component types.

        Returns:
            List of component types.
        """
        return list(self._components.keys())

    def get_names(self, component_type: Union[str, Type[Any]]) -> List[str]:
        """
        Get all registered component names for a specific type.

        Args:
            component_type: Type of components to get names for or a Type object.

        Returns:
            List of component names.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
        return list(self._components.get(component_type, {}).keys())

    def has_type(self, component_type: Union[str, Type[Any]]) -> bool:
        """
        Check if a component type is registered.

        Args:
            component_type: Type of component to check or a Type object.

        Returns:
            True if the component type is registered, False otherwise.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
        return component_type in self._components

    def has_component(self, component_type: Union[str, Type[Any]], name: str) -> bool:
        """
        Check if a component is registered.

        Args:
            component_type: Type of component to check or a Type object.
            name: Name of the component to check.

        Returns:
            True if the component is registered, False otherwise.
        """
        # Convert Type objects to strings
        if not isinstance(component_type, str):
            component_type = component_type.__name__
            
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
    
    def set_default_implementation(self, component_type: Type[Any], name: str) -> None:
        """
        Set the default implementation for a component type.

        Args:
            component_type: The component type.
            name: The name of the default implementation.

        Raises:
            ComponentRegistryError: If the component type or implementation is not registered.
        """
        # Convert the type to a string for storage
        type_name = component_type.__name__
        
        # Check if the component type is registered
        if type_name not in self._components:
            raise ComponentRegistryError(f"Component type '{type_name}' is not registered")
        
        # Check if the implementation is registered
        if name not in self._components[type_name]:
            raise ComponentRegistryError(
                f"Implementation '{name}' for component type '{type_name}' is not registered"
            )
        
        # Set the default implementation
        self._default_implementations[type_name] = name
        logger.debug(f"Set default implementation for {type_name} to {name}")
    
    def get_default_implementation(self, component_type: Type[Any]) -> Optional[Type[Any]]:
        """
        Get the default implementation for a component type.

        Args:
            component_type: The component type.

        Returns:
            The default implementation class if set, None otherwise.
        """
        # Convert the type to a string for lookup
        type_name = component_type.__name__
        
        # Check if a default implementation is set
        if type_name not in self._default_implementations:
            return None
        
        # Get the default implementation name
        default_name = self._default_implementations[type_name]
        
        # Return the implementation class
        return self._components[type_name].get(default_name)
