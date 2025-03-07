"""
Tests for the dependency injection decorators.

This module contains tests for the inject and injectable decorators.
"""

from typing import Protocol

import pytest

from nexusml.core.di.container import DIContainer
from nexusml.core.di.decorators import inject, injectable, injectable_with_params
from nexusml.core.di.provider import ContainerProvider


# Define test classes and interfaces
class ILogger(Protocol):
    """Test logger interface."""

    def log(self, message: str) -> None: ...


class ConsoleLogger:
    """Test logger implementation."""

    def log(self, message: str) -> None:
        pass  # In a real implementation, this would log to console


class IRepository(Protocol):
    """Test repository interface."""

    def get_data(self) -> str: ...


class DatabaseRepository:
    """Test repository implementation."""

    def get_data(self) -> str:
        return "Data from database"


class UserService:
    """Test service with injected dependencies."""

    @inject
    def __init__(self, logger: ILogger, repository: IRepository):
        self.logger = logger
        self.repository = repository

    def get_user_data(self) -> str:
        self.logger.log("Getting user data")
        return self.repository.get_data()


class ConfigService:
    """Test service with some manual dependencies."""

    @inject
    def __init__(self, logger: ILogger, config_path: str = "default_config.json"):
        self.logger = logger
        self.config_path = config_path


# These decorators are applied at import time, but we need to ensure they're registered
# in each test's container, so we'll also register them explicitly in setup_method
@injectable
class SingletonService:
    """Test service registered as injectable."""

    def get_value(self) -> str:
        return "SingletonService"


@injectable_with_params(singleton=True)
class AnotherSingletonService:
    """Test service registered as injectable with parameters."""

    def get_value(self) -> str:
        return "AnotherSingletonService"


class TestDecorators:
    """Tests for the dependency injection decorators."""

    def setup_method(self):
        """Set up the test environment."""
        # Reset the container provider
        ContainerProvider.reset_instance()
        provider = ContainerProvider()
        provider.reset()

        # Register test dependencies
        provider.register_implementation(ILogger, ConsoleLogger)
        provider.register_implementation(IRepository, DatabaseRepository)

        # Explicitly register the classes decorated with @injectable
        # This ensures they're registered in this test's container
        provider.container.register(SingletonService)  # Use container.register directly
        provider.container.register(
            AnotherSingletonService, singleton=True
        )  # Use container.register directly

    def test_inject_decorator(self):
        """Test the inject decorator."""
        # Create a service with injected dependencies
        # The @inject decorator should automatically resolve the dependencies
        service = UserService()  # type: ignore # Pylance doesn't understand @inject

        # Verify dependencies were injected
        assert isinstance(service.logger, ConsoleLogger)
        assert isinstance(service.repository, DatabaseRepository)
        assert service.get_user_data() == "Data from database"

    def test_inject_with_manual_parameters(self):
        """Test the inject decorator with manual parameters."""
        # Create a service with some manual parameters
        # The @inject decorator should only resolve the missing dependencies
        repo = DatabaseRepository()
        service = UserService(repository=repo)  # type: ignore # Pylance doesn't understand @inject

        # Verify dependencies were injected
        assert isinstance(service.logger, ConsoleLogger)
        assert service.repository is repo
        assert service.get_user_data() == "Data from database"

    def test_inject_with_default_parameters(self):
        """Test the inject decorator with default parameters."""
        # Create a service with default parameters
        # The @inject decorator should resolve the logger dependency
        service = ConfigService()  # type: ignore # Pylance doesn't understand @inject

        # Verify dependencies were injected
        assert isinstance(service.logger, ConsoleLogger)
        assert service.config_path == "default_config.json"

        # Create a service with manual parameters
        # The @inject decorator should not override the provided parameters
        service = ConfigService(config_path="custom_config.json")  # type: ignore # Pylance doesn't understand @inject

        # Verify dependencies were injected
        assert isinstance(service.logger, ConsoleLogger)
        assert service.config_path == "custom_config.json"

    def test_injectable_decorator(self):
        """Test the injectable decorator."""
        # Resolve the service from the container
        service = ContainerProvider().container.resolve(SingletonService)

        # Verify the service was resolved
        assert isinstance(service, SingletonService)
        assert service.get_value() == "SingletonService"

    def test_injectable_with_params_decorator(self):
        """Test the injectable_with_params decorator."""
        # Resolve the service from the container
        service1 = ContainerProvider().container.resolve(AnotherSingletonService)
        service2 = ContainerProvider().container.resolve(AnotherSingletonService)

        # Verify the service was resolved and is a singleton
        assert isinstance(service1, AnotherSingletonService)
        assert service1.get_value() == "AnotherSingletonService"
        assert service1 is service2  # Same instance

    def test_inject_with_missing_dependency(self):
        """Test the inject decorator with a missing dependency."""
        # Reset the container
        ContainerProvider.reset_instance()
        provider = ContainerProvider()
        provider.reset()

        # Only register the logger
        provider.register_implementation(ILogger, ConsoleLogger)

        # This should raise an exception because IRepository is not registered
        with pytest.raises(Exception):
            UserService()  # type: ignore # Pylance doesn't understand @inject

    def test_inject_method(self):
        """Test the inject decorator on a method."""

        class MethodInjectionService:
            @inject
            def process(self, logger: ILogger, data: str = "default"):
                return f"{logger.__class__.__name__} processed {data}"

        # Create a service
        service = MethodInjectionService()

        # Call the method with injection
        # The @inject decorator should resolve the logger dependency
        result = service.process()  # type: ignore # Pylance doesn't understand @inject
        assert result == "ConsoleLogger processed default"

        # Call the method with manual parameters
        # The @inject decorator should not override the provided parameters
        result = service.process(data="custom")  # type: ignore # Pylance doesn't understand @inject
        assert result == "ConsoleLogger processed custom"

        # Call the method with all manual parameters
        # The @inject decorator should not be used at all
        logger = ConsoleLogger()
        result = service.process(logger, "manual")
        assert result == "ConsoleLogger processed manual"
