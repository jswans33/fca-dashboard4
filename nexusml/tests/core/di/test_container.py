"""
Tests for the DIContainer class.

This module contains tests for the DIContainer class, including
registration, resolution, and singleton behavior.
"""

from typing import Protocol

import pytest

from nexusml.core.di.container import DependencyNotRegisteredError, DIContainer


# Define test classes and interfaces
class IService(Protocol):
    """Test service interface."""

    def get_value(self) -> str: ...


class ServiceA:
    """Test service implementation A."""

    def get_value(self) -> str:
        return "ServiceA"


class ServiceB:
    """Test service implementation B."""

    def get_value(self) -> str:
        return "ServiceB"


class Client:
    """Test client with dependency."""

    def __init__(self, service: IService):
        self.service = service

    def use_service(self) -> str:
        return f"Client using {self.service.get_value()}"


class ComplexService:
    """Test service with dependencies."""

    def __init__(self, service_a: ServiceA, service_b: ServiceB):
        self.service_a = service_a
        self.service_b = service_b

    def get_values(self) -> str:
        return f"{self.service_a.get_value()} and {self.service_b.get_value()}"


class TestDIContainer:
    """Tests for the DIContainer class."""

    def test_register_and_resolve(self):
        """Test registering and resolving a type."""
        container = DIContainer()
        container.register(IService, ServiceA)

        service = container.resolve(IService)
        assert isinstance(service, ServiceA)
        assert service.get_value() == "ServiceA"

    def test_register_concrete_type(self):
        """Test registering and resolving a concrete type."""
        container = DIContainer()
        container.register(ServiceA)

        service = container.resolve(ServiceA)
        assert isinstance(service, ServiceA)
        assert service.get_value() == "ServiceA"

    def test_register_factory(self):
        """Test registering and resolving with a factory function."""
        container = DIContainer()

        def factory(c: DIContainer) -> IService:
            return ServiceB()

        container.register_factory(IService, factory)

        service = container.resolve(IService)
        assert isinstance(service, ServiceB)
        assert service.get_value() == "ServiceB"

    def test_register_instance(self):
        """Test registering and resolving an instance."""
        container = DIContainer()
        instance = ServiceA()
        container.register_instance(IService, instance)

        service = container.resolve(IService)
        assert service is instance  # Same instance
        assert service.get_value() == "ServiceA"

    def test_singleton_behavior(self):
        """Test singleton behavior."""
        container = DIContainer()
        container.register(ServiceA, singleton=True)

        service1 = container.resolve(ServiceA)
        service2 = container.resolve(ServiceA)

        assert service1 is service2  # Same instance
        assert service1.get_value() == "ServiceA"

    def test_non_singleton_behavior(self):
        """Test non-singleton behavior."""
        container = DIContainer()
        container.register(ServiceA, singleton=False)

        service1 = container.resolve(ServiceA)
        service2 = container.resolve(ServiceA)

        assert service1 is not service2  # Different instances
        assert service1.get_value() == "ServiceA"
        assert service2.get_value() == "ServiceA"

    def test_factory_singleton_behavior(self):
        """Test singleton behavior with factory."""
        container = DIContainer()

        def factory(c: DIContainer) -> ServiceA:
            return ServiceA()

        container.register_factory(ServiceA, factory, singleton=True)

        service1 = container.resolve(ServiceA)
        service2 = container.resolve(ServiceA)

        assert service1 is service2  # Same instance
        assert service1.get_value() == "ServiceA"

    def test_dependency_resolution(self):
        """Test resolving dependencies."""
        container = DIContainer()
        container.register(ServiceA)
        container.register(ServiceB)
        container.register(ComplexService)

        service = container.resolve(ComplexService)
        assert isinstance(service, ComplexService)
        assert isinstance(service.service_a, ServiceA)
        assert isinstance(service.service_b, ServiceB)
        assert service.get_values() == "ServiceA and ServiceB"

    def test_dependency_not_registered(self):
        """Test exception when dependency is not registered."""
        container = DIContainer()

        with pytest.raises(DependencyNotRegisteredError):
            container.resolve(ServiceA)

    def test_clear(self):
        """Test clearing the container."""
        container = DIContainer()
        container.register(ServiceA)

        # Verify registration works
        service = container.resolve(ServiceA)
        assert isinstance(service, ServiceA)

        # Clear the container
        container.clear()

        # Verify registration is cleared
        with pytest.raises(DependencyNotRegisteredError):
            container.resolve(ServiceA)
