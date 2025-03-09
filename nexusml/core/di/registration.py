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
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelSerializer
)
from nexusml.core.model_building.base import ModelBuilder
from nexusml.core.model_building.builders.random_forest import RandomForestBuilder

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
    
    # Create a simple DataLoader implementation
    class SimpleDataLoader(DataLoader):
        def __init__(self, file_path=None):
            self.file_path = file_path
            
        def load_data(self, data_path=None, **kwargs):
            path = data_path or self.file_path
            if path is None:
                raise ValueError("No data path provided")
                
            import pandas as pd
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            elif path.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")
    
    # Create a simple DataPreprocessor implementation
    class SimpleDataPreprocessor(DataPreprocessor):
        def preprocess(self, data, **kwargs):
            """Preprocess the input data."""
            logger.info("Preprocessing data")
            
            # In a real implementation, this would clean and prepare the data
            # For this example, we'll just return the data as is
            return data
        
        def verify_required_columns(self, data):
            """Verify that all required columns exist in the DataFrame."""
            logger.info("Verifying required columns")
            
            # Define required columns
            required_columns = ["description", "service_life"]
            
            # Check if required columns exist
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                # For this example, we'll add missing columns with default values
                logger.warning(f"Missing required columns: {missing_columns}")
                for col in missing_columns:
                    if col == "description":
                        data[col] = "Unknown"
                    elif col == "service_life":
                        data[col] = 15.0  # Default service life
            
            return data
    
    # Create a simple ModelSerializer implementation
    class SimpleModelSerializer(ModelSerializer):
        def save_model(self, model, path, **kwargs):
            import pickle
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(model, f)
            return path
            
        def load_model(self, path, **kwargs):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    
    # Register DataLoader
    provider.register_implementation(
        DataLoader, SimpleDataLoader, singleton=False
    )
    logger.info("Registered DataLoader with DI container")
    # Create a simple FeatureEngineer implementation
    class SimpleFeatureEngineer(FeatureEngineer):
        def __init__(self, config=None):
            self.config = config or {}
            self.is_fitted = False
            
        def fit(self, data, **kwargs):
            """Fit the feature engineer to the data."""
            logger.info("Fitting feature engineer")
            self.is_fitted = True
            return self
            
        def transform(self, data, **kwargs):
            """Transform the data using the fitted feature engineer."""
            if not self.is_fitted:
                self.fit(data)
            
            logger.info("Transforming data with feature engineer")
            
            # Add combined_text column
            if "description" in data.columns:
                data["combined_text"] = data["description"]
            else:
                data["combined_text"] = "Unknown description"
            
            # Add service_life column if it doesn't exist
            if "service_life" not in data.columns:
                data["service_life"] = 15.0  # Default service life
            
            # Add required target columns for the orchestrator
            required_target_columns = [
                "category_name",
                "uniformat_code",
                "mcaa_system_category",
                "Equipment_Type",
                "System_Subtype",
            ]
            
            for col in required_target_columns:
                if col not in data.columns:
                    data[col] = "Unknown"  # Default value for target columns
            
            return data
            
        def engineer_features(self, data, **kwargs):
            """Engineer features from the input data."""
            return self.fit(data).transform(data)
    
    # Register DataPreprocessor
    provider.register_implementation(
        DataPreprocessor, SimpleDataPreprocessor, singleton=False
    )
    logger.info("Registered DataPreprocessor with DI container")
    
    # Register SimpleFeatureEngineer
    provider.register_implementation(
        FeatureEngineer, SimpleFeatureEngineer, singleton=False
    )
    
    # Register ModelBuilder
    provider.register_implementation(
        ModelBuilder, RandomForestBuilder, singleton=False
    )
    logger.info("Registered ModelBuilder with DI container")
    
    # Register ModelTrainer
    from nexusml.core.model_training.base import ModelTrainer, BaseModelTrainer
    provider.register_implementation(
        ModelTrainer, BaseModelTrainer, singleton=False
    )
    logger.info("Registered ModelTrainer with DI container")
    
    # Register ModelEvaluator
    from nexusml.core.model_building.base import ModelEvaluator, BaseModelEvaluator
    provider.register_implementation(
        ModelEvaluator, BaseModelEvaluator, singleton=False
    )
    logger.info("Registered ModelEvaluator with DI container")
    
    # Register ModelSerializer
    provider.register_implementation(
        ModelSerializer, SimpleModelSerializer, singleton=False
    )
    logger.info("Registered ModelSerializer with DI container")
    logger.info("Registered ModelSerializer with DI container")
    
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
