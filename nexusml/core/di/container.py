"""
Dependency Injection Container for NexusML.

This module provides the DIContainer class, which is responsible for
registering and resolving dependencies in the NexusML suite.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, get_type_hints

T = TypeVar("T")
TFactory = Callable[["DIContainer"], T]


class DIException(Exception):
    """Base exception for dependency injection errors."""

    pass


class DependencyNotRegisteredError(DIException):
    """Exception raised when a dependency is not registered in the container."""

    pass


class DIContainer:
    """
    Dependency Injection Container for managing dependencies.

    The DIContainer is responsible for registering and resolving dependencies,
    supporting singleton instances, factories, and direct instance registration.

    Attributes:
        _factories: Dictionary mapping types to factory functions
        _singletons: Dictionary mapping types to singleton instances
        _instances: Dictionary mapping types to specific instances
    """

    def __init__(self) -> None:
        """Initialize a new DIContainer with empty registrations."""
        self._factories: Dict[Type[Any], TFactory[Any]] = {}
        self._singletons: Dict[Type[Any], bool] = {}
        self._instances: Dict[Type[Any], Any] = {}

    def register(
        self,
        interface_type: Type[T],
        implementation_type: Optional[Type[T]] = None,
        singleton: bool = False,
    ) -> None:
        """
        Register a type with the container.

        Args:
            interface_type: The type to register (interface or concrete class)
            implementation_type: The implementation type (if different from interface_type)
            singleton: Whether the type should be treated as a singleton

        Note:
            If implementation_type is None, interface_type is used as the implementation.
        """
        if implementation_type is None:
            implementation_type = interface_type

        def factory(container: DIContainer) -> T:
            # Get constructor parameters
            init_params = get_type_hints(implementation_type.__init__).copy()  # type: ignore
            if "return" in init_params:
                del init_params["return"]

            # Resolve dependencies for constructor parameters
            kwargs = {}
            for param_name, param_type in init_params.items():
                if param_name != "self":
                    kwargs[param_name] = container.resolve(param_type)

            # Create instance
            return implementation_type(**kwargs)  # type: ignore

        self._factories[interface_type] = factory
        self._singletons[interface_type] = singleton

    def register_factory(
        self, interface_type: Type[T], factory: TFactory[T], singleton: bool = False
    ) -> None:
        """
        Register a factory function for creating instances.

        Args:
            interface_type: The type to register
            factory: A factory function that creates instances of the type
            singleton: Whether the type should be treated as a singleton
        """
        self._factories[interface_type] = factory
        self._singletons[interface_type] = singleton

    def register_instance(self, interface_type: Type[T], instance: T) -> None:
        """
        Register an existing instance with the container.

        Args:
            interface_type: The type to register
            instance: The instance to register
        """
        self._instances[interface_type] = instance

    def resolve(self, interface_type: Type[T]) -> T:
        """
        Resolve a dependency from the container.

        Args:
            interface_type: The type to resolve

        Returns:
            An instance of the requested type

        Raises:
            DependencyNotRegisteredError: If the type is not registered
        """
        # Check if we have a pre-registered instance
        if interface_type in self._instances:
            return cast(T, self._instances[interface_type])

        # Check if we have a factory for this type
        if interface_type not in self._factories:
            raise DependencyNotRegisteredError(
                f"Type {interface_type.__name__} is not registered in the container"
            )

        # Get the factory
        factory = self._factories[interface_type]

        # Check if this is a singleton
        if self._singletons.get(interface_type, False):
            if interface_type not in self._instances:
                self._instances[interface_type] = factory(self)
            return cast(T, self._instances[interface_type])

        # Create a new instance
        return factory(self)

    def clear(self) -> None:
        """Clear all registrations from the container."""
        self._factories.clear()
        self._singletons.clear()
        self._instances.clear()
