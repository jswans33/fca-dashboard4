"""
Tests for the ContainerProvider class.

This module contains tests for the ContainerProvider class, including
singleton behavior and initialization.
"""

import pytest

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider


# Define test classes
class TestService:
    """Test service class."""

    def get_value(self) -> str:
        return "TestService"


class TestContainerProvider:
    """Tests for the ContainerProvider class."""

    def setup_method(self):
        """Set up the test environment."""
        # Reset the singleton instance before each test
        ContainerProvider.reset_instance()

    def test_singleton_behavior(self):
        """Test that ContainerProvider is a singleton."""
        provider1 = ContainerProvider()
        provider2 = ContainerProvider()

        assert provider1 is provider2  # Same instance

    def test_container_initialization(self):
        """Test that container is initialized correctly."""
        provider = ContainerProvider()
        container = provider.container

        assert isinstance(container, DIContainer)

        # Getting the container again should return the same instance
        container2 = provider.container
        assert container is container2

    def test_register_implementation(self):
        """Test registering an implementation."""
        provider = ContainerProvider()
        provider.register_implementation(TestService, TestService, singleton=True)

        # Resolve the service
        service = provider.container.resolve(TestService)
        assert isinstance(service, TestService)
        assert service.get_value() == "TestService"

        # Resolve again to verify singleton behavior
        service2 = provider.container.resolve(TestService)
        assert service is service2  # Same instance

    def test_register_instance(self):
        """Test registering an instance."""
        provider = ContainerProvider()
        instance = TestService()
        provider.register_instance(TestService, instance)

        # Resolve the service
        service = provider.container.resolve(TestService)
        assert service is instance  # Same instance
        assert service.get_value() == "TestService"

    def test_register_factory(self):
        """Test registering a factory."""
        provider = ContainerProvider()

        def factory(container: DIContainer) -> TestService:
            return TestService()

        provider.register_factory(TestService, factory, singleton=True)

        # Resolve the service
        service = provider.container.resolve(TestService)
        assert isinstance(service, TestService)
        assert service.get_value() == "TestService"

        # Resolve again to verify singleton behavior
        service2 = provider.container.resolve(TestService)
        assert service is service2  # Same instance

    def test_reset(self):
        """Test resetting the container."""
        provider = ContainerProvider()
        provider.register_implementation(TestService, TestService)

        # Verify registration works
        service = provider.container.resolve(TestService)
        assert isinstance(service, TestService)

        # Reset the container
        provider.reset()

        # Verify registration is cleared
        with pytest.raises(Exception):
            provider.container.resolve(TestService)

    def test_reset_instance(self):
        """Test resetting the singleton instance."""
        provider1 = ContainerProvider()

        # Reset the singleton instance
        ContainerProvider.reset_instance()

        # Get a new instance
        provider2 = ContainerProvider()

        # Verify it's a different instance
        assert provider1 is not provider2

    def test_register_defaults(self):
        """Test that _register_defaults is called."""

        # Create a subclass that overrides _register_defaults
        class CustomProvider(ContainerProvider):
            def _register_defaults(self) -> None:
                self.container.register(TestService, singleton=True)

        # Reset the singleton instance
        ContainerProvider.reset_instance()

        # Create a custom provider
        provider = CustomProvider()

        # Verify that _register_defaults was called
        service = provider.container.resolve(TestService)
        assert isinstance(service, TestService)
        assert service.get_value() == "TestService"
