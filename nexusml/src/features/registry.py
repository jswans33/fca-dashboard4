"""
Transformer Registry Module

This module provides a registry for feature transformers in the NexusML suite.
It follows the Registry pattern to allow for dynamic registration and creation of transformers.
"""

from typing import Any, Dict, List, Optional, Type, cast

from nexusml.src.features.interfaces import FeatureTransformer, TransformerRegistry


class DefaultTransformerRegistry(TransformerRegistry):
    """
    Default implementation of the TransformerRegistry interface.

    This registry maintains a dictionary of transformer classes and provides methods
    for registering, retrieving, and creating transformers.
    """

    def __init__(self, name: str = "DefaultTransformerRegistry"):
        """
        Initialize the transformer registry.

        Args:
            name: Name of the registry.
        """
        self.name = name
        self._transformers: Dict[str, Type[FeatureTransformer]] = {}

    def register_transformer(
        self, name: str, transformer_class: Type[FeatureTransformer]
    ) -> None:
        """
        Register a transformer class with the registry.

        Args:
            name: Name to register the transformer under.
            transformer_class: Transformer class to register.

        Raises:
            ValueError: If the name is already registered or the class is not a transformer.
        """
        # Check if the name is already registered
        if name in self._transformers:
            raise ValueError(f"Transformer '{name}' is already registered")

        # Check if the class is a transformer
        if not issubclass(transformer_class, FeatureTransformer):
            raise ValueError(
                f"Class '{transformer_class.__name__}' is not a FeatureTransformer"
            )

        # Register the transformer
        self._transformers[name] = transformer_class

    def get_transformer_class(self, name: str) -> Type[FeatureTransformer]:
        """
        Get a transformer class from the registry.

        Args:
            name: Name of the transformer class to get.

        Returns:
            Transformer class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._transformers:
            raise KeyError(f"Transformer '{name}' is not registered")

        return self._transformers[name]

    def create_transformer(self, name: str, **kwargs: Any) -> FeatureTransformer:
        """
        Create a transformer instance from the registry.

        Args:
            name: Name of the transformer class to create.
            **kwargs: Arguments to pass to the transformer constructor.

        Returns:
            Transformer instance.

        Raises:
            KeyError: If the name is not registered.
            ValueError: If the transformer cannot be created with the given arguments.
        """
        # Get the transformer class
        transformer_class = self.get_transformer_class(name)

        # Create and return the transformer instance
        try:
            return transformer_class(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create transformer '{name}': {str(e)}") from e

    def get_registered_transformers(self) -> Dict[str, Type[FeatureTransformer]]:
        """
        Get all registered transformers.

        Returns:
            Dictionary mapping transformer names to transformer classes.
        """
        return self._transformers.copy()


# Singleton instance of the transformer registry
_default_registry = DefaultTransformerRegistry()


def get_default_registry() -> DefaultTransformerRegistry:
    """
    Get the default transformer registry.

    Returns:
        Default transformer registry.
    """
    return _default_registry


def register_transformer(
    name: str, transformer_class: Type[FeatureTransformer]
) -> None:
    """
    Register a transformer class with the default registry.

    Args:
        name: Name to register the transformer under.
        transformer_class: Transformer class to register.

    Raises:
        ValueError: If the name is already registered or the class is not a transformer.
    """
    _default_registry.register_transformer(name, transformer_class)


def get_transformer_class(name: str) -> Type[FeatureTransformer]:
    """
    Get a transformer class from the default registry.

    Args:
        name: Name of the transformer class to get.

    Returns:
        Transformer class.

    Raises:
        KeyError: If the name is not registered.
    """
    return _default_registry.get_transformer_class(name)


def create_transformer(name: str, **kwargs: Any) -> FeatureTransformer:
    """
    Create a transformer instance from the default registry.

    Args:
        name: Name of the transformer class to create.
        **kwargs: Arguments to pass to the transformer constructor.

    Returns:
        Transformer instance.

    Raises:
        KeyError: If the name is not registered.
        ValueError: If the transformer cannot be created with the given arguments.
    """
    return _default_registry.create_transformer(name, **kwargs)


def get_registered_transformers() -> Dict[str, Type[FeatureTransformer]]:
    """
    Get all registered transformers from the default registry.

    Returns:
        Dictionary mapping transformer names to transformer classes.
    """
    return _default_registry.get_registered_transformers()
