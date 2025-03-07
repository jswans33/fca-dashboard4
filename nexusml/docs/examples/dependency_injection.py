#!/usr/bin/env python
"""
Dependency Injection Example for NexusML

This example demonstrates how to use the dependency injection system in NexusML.
It covers:
- Registering dependencies
- Resolving dependencies
- Creating components with dependencies
- Using the container with the factory
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
)
from nexusml.core.pipeline.registry import ComponentRegistry


# Define some simple components for demonstration
class CSVDataLoader(DataLoader):
    """Simple CSV data loader."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize a new CSVDataLoader.

        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file (if None, uses self.file_path)
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the CSV file cannot be found
            ValueError: If the CSV format is invalid
        """
        path = data_path or self.file_path
        if path is None:
            raise ValueError("No data path provided")

        print(f"Loading data from {path}")
        # For demonstration, return a dummy DataFrame
        return pd.DataFrame(
            {"description": ["Item 1", "Item 2", "Item 3"], "value": [10, 20, 30]}
        )

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration
        """
        return {"file_path": self.file_path}


class SimplePreprocessor(DataPreprocessor):
    """Simple data preprocessor."""

    def preprocess(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Preprocess the input data.

        Args:
            data: Input DataFrame to preprocess
            **kwargs: Additional arguments for preprocessing

        Returns:
            Preprocessed DataFrame

        Raises:
            ValueError: If the data cannot be preprocessed
        """
        print("Preprocessing data")
        # For demonstration, just return the input data
        return data

    def verify_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Verify that all required columns exist in the DataFrame.

        Args:
            data: Input DataFrame to verify

        Returns:
            DataFrame with all required columns

        Raises:
            ValueError: If required columns cannot be created
        """
        required_columns = ["description", "value"]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Required column '{column}' not found")
        return data


class SimpleFeatureEngineer(FeatureEngineer):
    """Simple feature engineer."""

    def __init__(self, preprocessor: DataPreprocessor):
        """
        Initialize a new SimpleFeatureEngineer.

        Args:
            preprocessor: Data preprocessor to use
        """
        self.preprocessor = preprocessor

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features
            **kwargs: Additional arguments for feature engineering

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If features cannot be engineered
        """
        # First preprocess the data
        processed_data = self.preprocessor.preprocess(data)

        print("Engineering features")
        # For demonstration, add a new feature
        processed_data["feature"] = processed_data["value"] * 2
        return processed_data

    def fit(self, data: pd.DataFrame, **kwargs) -> "SimpleFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the feature engineer cannot be fit to the data
        """
        print("Fitting feature engineer")
        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If the data cannot be transformed
        """
        return self.engineer_features(data, **kwargs)


# Define a component with multiple dependencies
class DataProcessor:
    """Component that processes data using multiple dependencies."""

    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer):
        """
        Initialize a new DataProcessor.

        Args:
            data_loader: Data loader to use
            feature_engineer: Feature engineer to use
        """
        self.data_loader = data_loader
        self.feature_engineer = feature_engineer

    def process(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process data from the specified path.

        Args:
            data_path: Path to the data file

        Returns:
            Processed DataFrame

        Raises:
            FileNotFoundError: If the data file cannot be found
            ValueError: If the data format is invalid
        """
        # Load data
        data = self.data_loader.load_data(data_path)

        # Engineer features
        processed_data = self.feature_engineer.engineer_features(data)

        return processed_data


def example_basic_di():
    """Example of basic dependency injection."""
    print("\n=== Example: Basic Dependency Injection ===")

    # Create a DI container
    container = DIContainer()

    # Register components
    container.register(DataLoader, CSVDataLoader)
    container.register(DataPreprocessor, SimplePreprocessor)

    # Resolve components
    data_loader = container.resolve(DataLoader)
    preprocessor = container.resolve(DataPreprocessor)

    print("Components created:")
    print(f"  Data Loader: {type(data_loader).__name__}")
    print(f"  Preprocessor: {type(preprocessor).__name__}")

    # Use the components
    data = data_loader.load_data("dummy.csv")
    processed_data = preprocessor.preprocess(data)

    print("Data processed:")
    print(processed_data)

    return container


def example_nested_dependencies():
    """Example of nested dependencies."""
    print("\n=== Example: Nested Dependencies ===")

    # Create a DI container
    container = DIContainer()

    # Register components
    container.register(DataLoader, CSVDataLoader)
    container.register(DataPreprocessor, SimplePreprocessor)
    container.register(FeatureEngineer, SimpleFeatureEngineer)

    # Register component with dependencies
    container.register(DataProcessor)

    # Resolve the component with dependencies
    processor = container.resolve(DataProcessor)

    # Cast to specific implementation types for demonstration
    feature_engineer = cast(SimpleFeatureEngineer, processor.feature_engineer)

    print("Components created:")
    print(f"  Processor: {type(processor).__name__}")
    print(f"  Data Loader: {type(processor.data_loader).__name__}")
    print(f"  Feature Engineer: {type(processor.feature_engineer).__name__}")
    print(
        f"  Preprocessor (used by Feature Engineer): {type(feature_engineer.preprocessor).__name__}"
    )

    # Use the component
    processed_data = processor.process("dummy.csv")

    print("Data processed:")
    print(processed_data)

    return container


def example_singleton_vs_transient():
    """Example of singleton vs. transient dependencies."""
    print("\n=== Example: Singleton vs. Transient Dependencies ===")

    # Create a DI container
    container = DIContainer()

    # Register components
    container.register(DataLoader, CSVDataLoader, singleton=True)  # Singleton
    container.register(
        DataPreprocessor, SimplePreprocessor, singleton=False
    )  # Transient

    # Resolve components multiple times
    data_loader1 = container.resolve(DataLoader)
    data_loader2 = container.resolve(DataLoader)

    preprocessor1 = container.resolve(DataPreprocessor)
    preprocessor2 = container.resolve(DataPreprocessor)

    print("Singleton vs. Transient:")
    print(
        f"  Data Loader (Singleton): data_loader1 is data_loader2 = {data_loader1 is data_loader2}"
    )
    print(
        f"  Preprocessor (Transient): preprocessor1 is preprocessor2 = {preprocessor1 is preprocessor2}"
    )

    return container


def example_factory_functions():
    """Example of factory functions."""
    print("\n=== Example: Factory Functions ===")

    # Create a DI container
    container = DIContainer()

    # Define a factory function
    def create_data_loader(container: DIContainer) -> DataLoader:
        """Factory function for creating a data loader."""
        print("Creating data loader with factory function")
        return CSVDataLoader(file_path="factory_data.csv")

    # Register the factory function
    container.register_factory(DataLoader, create_data_loader)

    # Resolve the component
    data_loader = container.resolve(DataLoader)

    # Cast to specific implementation type for demonstration
    csv_data_loader = cast(CSVDataLoader, data_loader)

    print("Component created with factory function:")
    print(f"  Data Loader: {type(data_loader).__name__}")
    print(f"  File Path: {csv_data_loader.file_path}")

    return container


def example_direct_instance_registration():
    """Example of direct instance registration."""
    print("\n=== Example: Direct Instance Registration ===")

    # Create a DI container
    container = DIContainer()

    # Create an instance
    data_loader = CSVDataLoader(file_path="instance_data.csv")

    # Register the instance
    container.register_instance(DataLoader, data_loader)

    # Resolve the component
    resolved_data_loader = container.resolve(DataLoader)

    # Cast to specific implementation type for demonstration
    csv_resolved_data_loader = cast(CSVDataLoader, resolved_data_loader)

    print("Direct instance registration:")
    print(
        f"  Original instance is resolved instance = {data_loader is resolved_data_loader}"
    )
    print(f"  File Path: {csv_resolved_data_loader.file_path}")

    return container


def example_optional_dependencies():
    """Example of optional dependencies."""
    print("\n=== Example: Optional Dependencies ===")

    # Define a component with optional dependencies
    class ComponentWithOptionalDependencies:
        """Component with optional dependencies."""

        def __init__(
            self,
            data_loader: DataLoader,
            optional_dependency: Optional[DataPreprocessor] = None,
        ):
            """
            Initialize a new ComponentWithOptionalDependencies.

            Args:
                data_loader: Data loader to use
                optional_dependency: Optional preprocessor to use
            """
            self.data_loader = data_loader
            self.optional_dependency = optional_dependency

    # Create a DI container
    container = DIContainer()

    # Register required dependency
    container.register(DataLoader, CSVDataLoader)

    # Register the component
    container.register(ComponentWithOptionalDependencies)

    # Resolve the component
    component1 = container.resolve(ComponentWithOptionalDependencies)

    print("Component with missing optional dependency:")
    print(f"  Data Loader: {type(component1.data_loader).__name__}")
    print(f"  Optional Dependency: {component1.optional_dependency}")

    # Now register the optional dependency
    container.register(DataPreprocessor, SimplePreprocessor)

    # Resolve the component again
    component2 = container.resolve(ComponentWithOptionalDependencies)

    print("\nComponent with resolved optional dependency:")
    print(f"  Data Loader: {type(component2.data_loader).__name__}")
    print(f"  Optional Dependency: {type(component2.optional_dependency).__name__}")

    return container


def example_pipeline_factory():
    """Example of using the pipeline factory with DI container."""
    print("\n=== Example: Pipeline Factory with DI Container ===")

    # Create a registry and container
    registry = ComponentRegistry()
    container = DIContainer()

    # Register components in the registry
    registry.register(DataLoader, "csv", CSVDataLoader)
    registry.register(DataPreprocessor, "simple", SimplePreprocessor)
    registry.register(FeatureEngineer, "simple", SimpleFeatureEngineer)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "csv")
    registry.set_default_implementation(DataPreprocessor, "simple")
    registry.set_default_implementation(FeatureEngineer, "simple")

    # Create a factory
    factory = PipelineFactory(registry, container)

    # Create components
    data_loader = factory.create_data_loader(file_path="factory_data.csv")
    preprocessor = factory.create_data_preprocessor()
    feature_engineer = factory.create_feature_engineer()

    # Cast to specific implementation types for demonstration
    csv_data_loader = cast(CSVDataLoader, data_loader)
    simple_feature_engineer = cast(SimpleFeatureEngineer, feature_engineer)

    print("Components created with factory:")
    print(f"  Data Loader: {type(data_loader).__name__}")
    print(f"  Preprocessor: {type(preprocessor).__name__}")
    print(f"  Feature Engineer: {type(feature_engineer).__name__}")
    print(
        f"  Preprocessor (used by Feature Engineer): {type(simple_feature_engineer.preprocessor).__name__}"
    )

    # Use the components
    data = data_loader.load_data()
    processed_data = preprocessor.preprocess(data)
    features = feature_engineer.engineer_features(processed_data)

    print("Data processed:")
    print(features)

    return factory


def main():
    """Main function to demonstrate dependency injection in NexusML."""
    print("NexusML Dependency Injection Example")
    print("====================================")

    # Example 1: Basic Dependency Injection
    container1 = example_basic_di()

    # Example 2: Nested Dependencies
    container2 = example_nested_dependencies()

    # Example 3: Singleton vs. Transient Dependencies
    container3 = example_singleton_vs_transient()

    # Example 4: Factory Functions
    container4 = example_factory_functions()

    # Example 5: Direct Instance Registration
    container5 = example_direct_instance_registration()

    # Example 6: Optional Dependencies
    container6 = example_optional_dependencies()

    # Example 7: Pipeline Factory
    factory = example_pipeline_factory()

    print("\n=== Dependency Injection Example Completed ===")


if __name__ == "__main__":
    main()
