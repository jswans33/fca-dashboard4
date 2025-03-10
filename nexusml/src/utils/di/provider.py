"""
Container Provider for NexusML Dependency Injection.

This module provides the ContainerProvider class, which implements
the singleton pattern for accessing the DIContainer.
"""

from typing import Optional, Type

from nexusml.core.di.container import DIContainer


class ContainerProvider:
    """
    Singleton provider for accessing the DIContainer.

    This class ensures that only one DIContainer instance is used
    throughout the application, following the singleton pattern.

    Attributes:
        _instance: The singleton instance of ContainerProvider
        _container: The DIContainer instance
    """

    _instance: Optional["ContainerProvider"] = None
    _container: Optional[DIContainer] = None

    def __new__(cls) -> "ContainerProvider":
        """
        Create or return the singleton instance of ContainerProvider.

        Returns:
            The singleton ContainerProvider instance
        """
        if cls._instance is None:
            cls._instance = super(ContainerProvider, cls).__new__(cls)
            cls._instance._container = None
        return cls._instance

    @property
    def container(self) -> DIContainer:
        """
        Get the DIContainer instance, creating it if it doesn't exist.

        Returns:
            The DIContainer instance
        """
        if self._container is None:
            self._container = DIContainer()
            self._register_defaults()
        return self._container

    def _register_defaults(self) -> None:
        """
        Register default dependencies in the container.

        This method is called when the container is first created.
        Override this method to register default dependencies.
        """
        pass

    def reset(self) -> None:
        """
        Reset the container, clearing all registrations.

        This method is primarily used for testing.
        """
        if self._container is not None:
            self._container.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.

        This method is primarily used for testing.
        """
        cls._instance = None

    def register_implementation(
        self, interface_type: Type, implementation_type: Type, singleton: bool = False
    ) -> None:
        """
        Register an implementation type for an interface.

        Args:
            interface_type: The interface type
            implementation_type: The implementation type
            singleton: Whether the implementation should be a singleton
        """
        self.container.register(interface_type, implementation_type, singleton)

    def register_instance(self, interface_type: Type, instance: object) -> None:
        """
        Register an instance for an interface.

        Args:
            interface_type: The interface type
            instance: The instance to register
        """
        self.container.register_instance(interface_type, instance)

    def register_factory(
        self, interface_type: Type, factory, singleton: bool = False
    ) -> None:
        """
        Register a factory function for an interface.

        Args:
            interface_type: The interface type
            factory: The factory function
            singleton: Whether the factory should produce singletons
        """
        self.container.register_factory(interface_type, factory, singleton)
