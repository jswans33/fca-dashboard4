"""
Decorators for Dependency Injection in NexusML.

This module provides decorators for simplifying dependency injection
in the NexusML suite, including constructor injection and class registration.
"""

import contextlib
import functools
import inspect
from typing import (
    Any,
    Callable,
    Type,
    TypeVar,
    cast,
    get_type_hints,
    overload,
)

from nexusml.core.di.provider import ContainerProvider

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def inject(func: F) -> F:
    """
    Decorator for injecting dependencies into a constructor or method.

    This decorator automatically resolves dependencies for parameters
    based on their type annotations.

    Args:
        func: The function or method to inject dependencies into

    Returns:
        A wrapped function that automatically resolves dependencies

    Example:
        ```python
        class MyService:
            @inject
            def __init__(self, dependency: SomeDependency):
                self.dependency = dependency
        ```
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        container = ContainerProvider().container

        # Get type hints for the function
        hints = get_type_hints(func)
        if "return" in hints:
            del hints["return"]

        # For each parameter that isn't provided, try to resolve it from the container
        for param_name in sig.parameters:
            # Skip self parameter for methods
            if param_name == "self" and args:
                continue

            # Skip parameters that are already provided
            if param_name in kwargs or (
                args and len(args) > list(sig.parameters.keys()).index(param_name)
            ):
                continue
            # Try to resolve the parameter from the container
            if param_name in hints:
                param_type = hints[param_name]
                with contextlib.suppress(Exception):
                    kwargs[param_name] = container.resolve(param_type)
                    # If resolution fails, let the function handle the missing parameter
                    pass

        return func(*args, **kwargs)

    return cast(F, wrapper)


# Make injectable work both as @injectable and @injectable(singleton=True)
@overload
def injectable(cls: Type[T]) -> Type[T]: ...


@overload
def injectable(*, singleton: bool = False) -> Callable[[Type[T]], Type[T]]: ...


def injectable(cls=None, *, singleton=False):
    """
    Decorator for registering a class with the DI container.

    This decorator registers the class with the container and
    optionally marks it as a singleton.

    Can be used in two ways:
    1. As a simple decorator: @injectable
    2. With parameters: @injectable(singleton=True)

    Args:
        cls: The class to register (when used as @injectable)
        singleton: Whether the class should be treated as a singleton

    Returns:
        The original class (unchanged) or a decorator function

    Example:
        ```python
        @injectable
        class MyService:
            def __init__(self, dependency: SomeDependency):
                self.dependency = dependency

        @injectable(singleton=True)
        class MySingletonService:
            def __init__(self, dependency: SomeDependency):
                self.dependency = dependency
        ```
    """
    # Used as @injectable without parentheses
    if cls is not None:
        ContainerProvider().container.register(cls, singleton=singleton)
        return cls

    # Used as @injectable(singleton=True) with parentheses
    def decorator(cls: Type[T]) -> Type[T]:
        ContainerProvider().container.register(cls, singleton=singleton)
        return cls

    return decorator


# Keep the alternative syntax for backward compatibility
def injectable_with_params(singleton: bool = False) -> Callable[[Type[T]], Type[T]]:
    """
    Parameterized version of the injectable decorator.

    This function returns a decorator that registers a class with the container
    and optionally marks it as a singleton.

    Args:
        singleton: Whether the class should be treated as a singleton

    Returns:
        A decorator function

    Example:
        ```python
        @injectable_with_params(singleton=True)
        class MyService:
            def __init__(self, dependency: SomeDependency):
                self.dependency = dependency
        ```
    """
    return injectable(singleton=singleton)
