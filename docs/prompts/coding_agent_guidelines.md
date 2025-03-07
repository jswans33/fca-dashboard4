# NexusML Refactoring: Coding Agent Guidelines

This document provides a general coding agent prompt and best practices for
implementing the NexusML refactoring work chunks.

## General Coding Agent Prompt

```
You are an expert Python software engineer specializing in machine learning systems architecture. You're implementing code for the NexusML refactoring project, a Python machine learning package for equipment classification that's being restructured to follow SOLID principles.

Your primary responsibility is to write high-quality, maintainable Python code that adheres to the architectural decisions and best practices established for this project. You should focus on creating clean, well-tested implementations that maintain backward compatibility while improving the overall architecture.

When implementing code for this project, you should:

1. Follow the Python style guide (PEP 8) and use consistent naming conventions
2. Include comprehensive type hints using the typing module
3. Write detailed docstrings in the Google style format
4. Implement robust error handling with specific exception types
5. Create unit tests for all new components
6. Maintain backward compatibility through adapter patterns and feature flags
7. Apply SOLID principles throughout your implementations
8. Include logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
9. Validate inputs and handle edge cases gracefully
10. Document any assumptions or design decisions in code comments

You have access to the existing codebase and can reference it to understand current functionality, but your goal is to improve the architecture while preserving behavior. Your code should be production-ready, well-documented, and thoroughly tested.
```

## Coding Best Practices for NexusML Refactoring

### Code Structure and Organization

1. **Package Structure**

   - Maintain the established package structure with clear separation of
     concerns
   - Place interfaces in `nexusml/core/pipeline/interfaces.py`
   - Place components in `nexusml/core/pipeline/components/`
   - Place adapters in `nexusml/core/pipeline/adapters/`
   - Place tests in corresponding directories under `tests/`

2. **File Naming**

   - Use snake_case for file names (e.g., `data_loader.py`, `model_builder.py`)
   - Use descriptive names that indicate the purpose of the file
   - Group related files in appropriate directories

3. **Import Organization**
   - Organize imports in the following order:
     1. Standard library imports
     2. Third-party library imports
     3. Local application imports
   - Use absolute imports for clarity
   - Avoid wildcard imports (`from module import *`)

### Code Style and Quality

1. **PEP 8 Compliance**

   - Follow PEP 8 style guide for Python code
   - Use 4 spaces for indentation
   - Limit line length to 88 characters (Black formatter standard)
   - Use appropriate spacing around operators and after commas

2. **Naming Conventions**

   - Use CamelCase for class names (e.g., `DataLoader`, `ModelBuilder`)
   - Use snake_case for function and variable names (e.g., `load_data`,
     `model_path`)
   - Use UPPER_CASE for constants (e.g., `DEFAULT_CONFIG_PATH`)
   - Use descriptive names that indicate purpose

3. **Type Hints**

   - Use type hints for all function parameters and return values
   - Use the `typing` module for complex types (e.g., `List`, `Dict`,
     `Optional`)
   - Use `Union` for parameters that can accept multiple types
   - Use `Any` sparingly and only when absolutely necessary

4. **Docstrings**
   - Use Google style docstrings for all classes and functions
   - Include descriptions for all parameters and return values
   - Document exceptions that may be raised
   - Include examples for complex functions
   - Format:
     ```python
     def function(param1: type, param2: type) -> return_type:
         """Short description of function.

         Longer description explaining the function's purpose and behavior.

         Args:
             param1: Description of param1
             param2: Description of param2

         Returns:
             Description of return value

         Raises:
             ExceptionType: Description of when this exception is raised

         Examples:
             >>> function("example", 123)
             "example result"
         """
     ```

### SOLID Principles Implementation

1. **Single Responsibility Principle**

   - Each class should have only one reason to change
   - Break large classes into smaller, focused classes
   - Extract utility functions for reusable functionality

2. **Open/Closed Principle**

   - Design classes to be open for extension but closed for modification
   - Use abstract base classes and interfaces to define contracts
   - Implement new functionality by extending existing classes rather than
     modifying them

3. **Liskov Substitution Principle**

   - Ensure that subclasses can be used in place of their parent classes
   - Maintain the same interface in derived classes
   - Don't strengthen preconditions or weaken postconditions in subclasses

4. **Interface Segregation Principle**

   - Create focused interfaces rather than large, general-purpose ones
   - Split large interfaces into smaller, more specific ones
   - Clients should only depend on methods they actually use

5. **Dependency Inversion Principle**
   - Depend on abstractions, not concrete implementations
   - Use dependency injection to provide dependencies
   - Use the DI container for resolving dependencies

### Error Handling and Logging

1. **Error Handling**

   - Use specific exception types rather than generic exceptions
   - Create custom exception classes for domain-specific errors
   - Handle exceptions at the appropriate level
   - Include context information in exception messages
   - Use try/except blocks judiciously

2. **Logging**
   - Use the logging module for all logging
   - Use appropriate log levels:
     - DEBUG: Detailed information for debugging
     - INFO: Confirmation that things are working as expected
     - WARNING: Indication that something unexpected happened
     - ERROR: Due to a more serious problem, the software has not been able to
       perform a function
     - CRITICAL: A serious error indicating that the program itself may be
       unable to continue running
   - Include context information in log messages
   - Don't log sensitive information

### Testing

1. **Unit Testing**

   - Write unit tests for all new components
   - Use pytest for testing
   - Structure tests to match the code structure
   - Use descriptive test names that indicate what is being tested
   - Follow the Arrange-Act-Assert pattern
   - Use fixtures for common setup
   - Mock external dependencies

2. **Integration Testing**

   - Write integration tests for component interactions
   - Test with realistic data
   - Verify that components work together correctly
   - Test error handling and edge cases

3. **Test Coverage**
   - Aim for high test coverage (>80%)
   - Focus on testing business logic and edge cases
   - Use parameterized tests for testing multiple scenarios

### Backward Compatibility

1. **Adapter Pattern**

   - Use adapter classes to maintain backward compatibility
   - Implement new interfaces while delegating to existing code
   - Ensure that existing code continues to work

2. **Feature Flags**

   - Use feature flags to toggle between old and new code paths
   - Make feature flags configurable
   - Default to the old code path until the new one is fully tested

3. **Deprecation Strategy**
   - Mark old functions and classes as deprecated using the `@deprecated`
     decorator
   - Provide migration guidance in deprecation warnings
   - Plan for eventual removal of deprecated code

### Configuration Management

1. **Configuration Validation**

   - Use Pydantic for configuration validation
   - Define clear schemas for all configuration options
   - Provide sensible default values
   - Validate configuration at startup

2. **Configuration Loading**
   - Use a consistent approach to configuration loading
   - Support multiple configuration sources (files, environment variables)
   - Handle missing configuration gracefully
   - Provide clear error messages for invalid configuration

### Documentation

1. **Code Documentation**

   - Write clear docstrings for all classes and functions
   - Include examples for complex functionality
   - Document design decisions and assumptions
   - Keep documentation up-to-date with code changes

2. **Architecture Documentation**

   - Document the overall architecture
   - Explain component interactions
   - Include diagrams where appropriate
   - Document design patterns used

3. **Migration Guides**
   - Provide clear migration guides for existing code
   - Include examples of before and after
   - Document breaking changes
   - Provide step-by-step migration instructions

### Performance Considerations

1. **Efficiency**

   - Optimize for readability first, then performance
   - Use profiling to identify bottlenecks
   - Consider memory usage for large datasets
   - Use appropriate data structures for the task

2. **Scalability**
   - Design components to handle increasing data volumes
   - Consider parallelization for CPU-bound tasks
   - Use batch processing for large datasets
   - Implement caching where appropriate

## Implementation Checklist

When implementing a work chunk, ensure that you:

- [ ] Understand the requirements and context
- [ ] Review the existing code that will be affected
- [ ] Design the solution following SOLID principles
- [ ] Implement the solution with comprehensive type hints and docstrings
- [ ] Write unit tests for all new components
- [ ] Implement integration tests for component interactions
- [ ] Ensure backward compatibility
- [ ] Document the implementation
- [ ] Review the code for adherence to best practices
- [ ] Verify that all tests pass

## Example Implementation

Here's an example of a well-implemented component following these guidelines:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import logging
from pydantic import BaseModel, Field

# Configuration model
class DataLoaderConfig(BaseModel):
    """Configuration for data loading."""

    default_path: str = Field(
        default="ingest/data/eq_ids.csv",
        description="Default path to the data file"
    )
    encoding: str = Field(
        default="utf-8",
        description="Encoding to use when reading the file"
    )
    fallback_encoding: str = Field(
        default="latin1",
        description="Fallback encoding to use if the primary encoding fails"
    )

# Interface
class DataLoader(ABC):
    """Interface for data loading components."""

    @abstractmethod
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a source.

        Args:
            data_path: Path to the data file. If None, uses the default path
                from configuration.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the data file is not found
            ValueError: If the data file cannot be parsed
        """
        pass

# Implementation
class StandardDataLoader(DataLoader):
    """Standard implementation of data loading."""

    def __init__(self, config_provider=None):
        """
        Initialize the data loader.

        Args:
            config_provider: Configuration provider. If None, uses the default.
        """
        self.config_provider = config_provider or ConfigurationProvider()
        self.logger = logging.getLogger(__name__)

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file. If None, uses the path from configuration.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the data file is not found
            ValueError: If the data file cannot be parsed
        """
        config = self.config_provider.config.data

        # Use default path if none provided
        if data_path is None:
            data_path = config.default_path
            self.logger.info(f"Using default data path: {data_path}")
        else:
            self.logger.info(f"Using provided data path: {data_path}")

        # Read CSV file using pandas
        encoding = config.encoding
        fallback_encoding = config.fallback_encoding

        try:
            self.logger.debug(f"Attempting to read file with {encoding} encoding")
            df = pd.read_csv(data_path, encoding=encoding)
        except UnicodeDecodeError:
            # Try with a different encoding if the primary one fails
            self.logger.warning(
                f"Failed to read with {encoding} encoding. Trying {fallback_encoding}."
            )
            try:
                df = pd.read_csv(data_path, encoding=fallback_encoding)
            except UnicodeDecodeError as e:
                self.logger.error(f"Failed to read file with fallback encoding: {e}")
                raise ValueError(f"Could not decode file with {encoding} or {fallback_encoding}")
        except FileNotFoundError:
            self.logger.error(f"Data file not found at {data_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}. Please provide a valid path.")

        # Clean up column names (remove any leading/trailing whitespace)
        df.columns = [col.strip() for col in df.columns]
        self.logger.debug(f"Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns")

        return df

# Adapter
class LegacyDataLoaderAdapter:
    """Adapter for backward compatibility with legacy code."""

    def __init__(self, data_loader: DataLoader):
        """
        Initialize the adapter.

        Args:
            data_loader: DataLoader implementation to delegate to
        """
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

    def load_and_preprocess_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Legacy function for loading and preprocessing data.

        Args:
            data_path: Path to the data file

        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Using legacy load_and_preprocess_data function")
        return self.data_loader.load_data(data_path)
```

This example demonstrates:

- Clear interface definition with abstract methods
- Comprehensive type hints
- Detailed docstrings
- Robust error handling
- Appropriate logging
- Configuration management
- Adapter pattern for backward compatibility
