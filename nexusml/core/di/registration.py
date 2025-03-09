"""
Dependency Injection Registration Module

This module provides functions for registering components with the DI container.
It serves as a central place for configuring the dependency injection container
with all the components needed by the NexusML suite.
"""

import logging
from typing import Any, Dict, Optional, Type

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.di.pipeline_registration import register_pipeline_components
from nexusml.core.eav_manager import EAVManager
from nexusml.core.feature_engineering import GenericFeatureEngineer
from nexusml.core.model import EquipmentClassifier
from nexusml.core.pipeline.components.feature_engineer import StandardFeatureEngineer
from nexusml.core.pipeline.interfaces import FeatureEngineer

# Set up logging
logger = logging.getLogger(__name__)


def register_core_components(
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register core components with the DI container.

    This function registers all the core components needed by the NexusML suite,
    including data components, feature engineering components, and model components.

    Args:
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()

    # Register EAVManager
    provider.register_implementation(EAVManager, EAVManager, singleton=True)
    logger.info("Registered EAVManager with DI container")

    # Register FeatureEngineer implementations
    provider.register_implementation(
        FeatureEngineer, StandardFeatureEngineer, singleton=False
    )
    provider.register_implementation(
        GenericFeatureEngineer, GenericFeatureEngineer, singleton=False
    )
    logger.info("Registered FeatureEngineer implementations with DI container")

    # Register EquipmentClassifier
    provider.register_implementation(
        EquipmentClassifier, EquipmentClassifier, singleton=False
    )
    logger.info("Registered EquipmentClassifier with DI container")
    
    # Register pipeline components
    register_pipeline_components(provider)
    logger.info("Registered pipeline components with DI container")


def register_custom_implementation(
    interface_type: Type,
    implementation_type: Type,
    singleton: bool = False,
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register a custom implementation with the DI container.

    This function allows registering custom implementations for interfaces,
    which is useful for testing and extending the system.

    Args:
        interface_type: The interface type to register.
        implementation_type: The implementation type to register.
        singleton: Whether the implementation should be a singleton.
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()
    provider.register_implementation(
        interface_type, implementation_type, singleton=singleton
    )
    logger.info(
        f"Registered custom implementation {implementation_type.__name__} for {interface_type.__name__}"
    )


def register_instance(
    interface_type: Type,
    instance: Any,
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register an instance with the DI container.

    This function allows registering pre-created instances with the container,
    which is useful for testing and configuration.

    Args:
        interface_type: The interface type to register.
        instance: The instance to register.
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()
    provider.register_instance(interface_type, instance)
    logger.info(
        f"Registered instance of {type(instance).__name__} for {interface_type.__name__}"
    )


def register_factory(
    interface_type: Type,
    factory,
    singleton: bool = False,
    container_provider: Optional[ContainerProvider] = None,
) -> None:
    """
    Register a factory function with the DI container.

    This function allows registering factory functions for creating instances,
    which is useful for complex creation logic.

    Args:
        interface_type: The interface type to register.
        factory: The factory function to register.
        singleton: Whether the factory should produce singletons.
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()
    provider.register_factory(interface_type, factory, singleton=singleton)
    logger.info(f"Registered factory for {interface_type.__name__}")


def configure_container(
    config: Dict[str, Any], container_provider: Optional[ContainerProvider] = None
) -> None:
    """
    Configure the DI container with the provided configuration.

    This function allows configuring the container with a dictionary of settings,
    which is useful for loading configuration from files.

    Args:
        config: Configuration dictionary.
        container_provider: The container provider to use. If None, creates a new one.
    """
    provider = container_provider or ContainerProvider()

    # Register components based on configuration
    for component_config in config.get("components", []):
        interface = component_config.get("interface")
        implementation = component_config.get("implementation")
        singleton = component_config.get("singleton", False)

        if interface and implementation:
            # Import the types dynamically
            interface_parts = interface.split(".")
            implementation_parts = implementation.split(".")

            interface_module = __import__(
                ".".join(interface_parts[:-1]), fromlist=[interface_parts[-1]]
            )
            implementation_module = __import__(
                ".".join(implementation_parts[:-1]), fromlist=[implementation_parts[-1]]
            )

            interface_type = getattr(interface_module, interface_parts[-1])
            implementation_type = getattr(
                implementation_module, implementation_parts[-1]
            )

            provider.register_implementation(
                interface_type, implementation_type, singleton=singleton
            )
            logger.info(
                f"Registered {implementation} for {interface} from configuration"
            )

    logger.info("Container configured successfully")


# Initialize the container with default registrations when the module is imported
register_core_components()
