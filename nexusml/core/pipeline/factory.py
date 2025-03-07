"""
Pipeline Factory Module

This module provides the PipelineFactory class, which is responsible for
creating pipeline components with proper dependencies.
"""

import inspect
from typing import Any, Dict, Optional, Type, TypeVar, cast, get_type_hints

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.core.pipeline.registry import ComponentRegistry

T = TypeVar("T")


class PipelineFactoryError(Exception):
    """Exception raised for errors in the PipelineFactory."""

    pass


class PipelineFactory:
    """
    Factory for creating pipeline components.

    This class is responsible for creating pipeline components with proper dependencies.
    It uses the ComponentRegistry to look up component implementations and the
    DIContainer to resolve dependencies.

    Example:
        >>> registry = ComponentRegistry()
        >>> container = DIContainer()
        >>> factory = PipelineFactory(registry, container)
        >>> data_loader = factory.create_data_loader()
        >>> preprocessor = factory.create_data_preprocessor()
        >>> # Use the components...
    """

    def __init__(self, registry: ComponentRegistry, container: DIContainer):
        """
        Initialize a new PipelineFactory.

        Args:
            registry: The component registry to use for looking up implementations.
            container: The dependency injection container to use for resolving dependencies.
        """
        self.registry = registry
        self.container = container

    def create(
        self, component_type: Type[T], name: Optional[str] = None, **kwargs
    ) -> T:
        """
        Create a component of the specified type.

        This method looks up the component implementation in the registry and creates
        an instance with dependencies resolved from the container.

        Args:
            component_type: The interface or base class of the component to create.
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of the component.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        try:
            # Get the implementation class
            if name is not None:
                implementation = self.registry.get_implementation(component_type, name)
            else:
                try:
                    implementation = self.registry.get_default_implementation(
                        component_type
                    )
                except Exception as e:
                    raise PipelineFactoryError(
                        f"No default implementation for {component_type.__name__}. "
                        f"Please specify a name or set a default implementation."
                    ) from e

            # Get the constructor signature
            signature = inspect.signature(implementation.__init__)
            parameters = signature.parameters

            # Prepare arguments for the constructor
            args: Dict[str, Any] = {}

            # Add dependencies from the container
            for param_name, param in parameters.items():
                if param_name == "self":
                    continue

                # If the parameter is provided in kwargs, use that
                if param_name in kwargs:
                    args[param_name] = kwargs[param_name]
                    continue

                # Try to get the parameter type
                param_type = param.annotation
                if param_type is inspect.Parameter.empty:
                    # Try to get the type from type hints
                    type_hints = get_type_hints(implementation.__init__)
                    if param_name in type_hints:
                        param_type = type_hints[param_name]
                    else:
                        # Skip parameters without type hints
                        continue

                # Try to resolve the dependency from the container
                try:
                    args[param_name] = self.container.resolve(param_type)
                except Exception:
                    # If the parameter has a default value, skip it
                    if param.default is not inspect.Parameter.empty:
                        continue
                    # Otherwise, try to create it using the factory
                    try:
                        args[param_name] = self.create(param_type)
                    except Exception as e:
                        # If we can't create it, and it's not in kwargs, raise an error
                        if param_name not in kwargs:
                            raise PipelineFactoryError(
                                f"Could not resolve dependency '{param_name}' "
                                f"of type '{param_type}' for {implementation.__name__}"
                            ) from e

            # Create the component
            return implementation(**args)

        except Exception as e:
            if isinstance(e, PipelineFactoryError):
                raise
            raise PipelineFactoryError(
                f"Error creating {component_type.__name__}: {str(e)}"
            ) from e

    def create_data_loader(self, name: Optional[str] = None, **kwargs) -> DataLoader:
        """
        Create a data loader component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of DataLoader.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(DataLoader, name, **kwargs)

    def create_data_preprocessor(
        self, name: Optional[str] = None, **kwargs
    ) -> DataPreprocessor:
        """
        Create a data preprocessor component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of DataPreprocessor.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(DataPreprocessor, name, **kwargs)

    def create_feature_engineer(
        self, name: Optional[str] = None, **kwargs
    ) -> FeatureEngineer:
        """
        Create a feature engineer component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of FeatureEngineer.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(FeatureEngineer, name, **kwargs)

    def create_model_builder(
        self, name: Optional[str] = None, **kwargs
    ) -> ModelBuilder:
        """
        Create a model builder component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of ModelBuilder.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(ModelBuilder, name, **kwargs)

    def create_model_trainer(
        self, name: Optional[str] = None, **kwargs
    ) -> ModelTrainer:
        """
        Create a model trainer component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of ModelTrainer.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(ModelTrainer, name, **kwargs)

    def create_model_evaluator(
        self, name: Optional[str] = None, **kwargs
    ) -> ModelEvaluator:
        """
        Create a model evaluator component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of ModelEvaluator.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(ModelEvaluator, name, **kwargs)

    def create_model_serializer(
        self, name: Optional[str] = None, **kwargs
    ) -> ModelSerializer:
        """
        Create a model serializer component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of ModelSerializer.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(ModelSerializer, name, **kwargs)

    def create_predictor(self, name: Optional[str] = None, **kwargs) -> Predictor:
        """
        Create a predictor component.

        Args:
            name: The name of the specific implementation to create. If None, uses the default.
            **kwargs: Additional arguments to pass to the component constructor.

        Returns:
            An instance of Predictor.

        Raises:
            PipelineFactoryError: If the component cannot be created.
        """
        return self.create(Predictor, name, **kwargs)
