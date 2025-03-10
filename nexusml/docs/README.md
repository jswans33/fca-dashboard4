# NexusML Documentation

Welcome to the NexusML documentation! This directory contains comprehensive documentation for the NexusML package, a Python machine learning package for equipment classification.

## Documentation Structure

The documentation is organized into the following sections:

- [Architecture](architecture/README.md): Documentation for the system architecture
- [CLI](cli/README.md): Documentation for command-line interface tools
- [Examples](examples/README.md): Documentation for example scripts
- [Modules](modules/README.md): Documentation for core modules
- [Scripts](scripts/README.md): Documentation for utility scripts
- [Utils](utils/README.md): Documentation for utility modules

## Getting Started

If you're new to NexusML, we recommend starting with the following resources:

1. [Installation Guide](installation_guide.md): Instructions for installing NexusML
2. [Usage Guide](usage_guide.md): Comprehensive guide for using NexusML
3. [Examples](examples/README.md): Practical examples of using NexusML
4. [Architecture Overview](architecture/overview.md): High-level overview of the system architecture

## Documentation Approach

The NexusML documentation follows a top-down approach:

1. **Level 1**: High-level architecture and concepts
2. **Level 2**: Major components and subsystems
3. **Level 3**: Individual modules, classes, and functions
4. **Level 4**: Examples and verification

This approach ensures a coherent narrative throughout the documentation and helps users understand how components fit together in the overall system.

## Documentation Standards

### Google Style Docstrings

NexusML uses Google style docstrings for all Python code. This format provides a clear structure for documenting parameters, returns, exceptions, and more.

Example:

```python
def function_example(param1, param2):
    """Summary line.

    Extended description of function.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    Raises:
        ValueError: If param1 is equal to param2.
        TypeError: If param1 is not an integer.

    Examples:
        >>> function_example(1, 'test')
        True
        >>> function_example(2, 'test')
        False
    """
```

### Better Comments Integration

NexusML uses the Better Comments extension to enhance code documentation with the following comment types:

```python
# * Important information that needs highlighting
# ! Alert or warning about potential issues
# ? Question that needs resolution or clarification
# TODO: Task that needs to be completed
# // Commented out code that should be reviewed
```

## Documentation Checklist

The [Documentation Checklist](documentation_checklist.md) tracks the progress of the NexusML documentation effort. It follows the top-down approach outlined in the [Documentation Plan](documentation_plan.md).

## Contributing to Documentation

If you'd like to contribute to the NexusML documentation:

1. Follow the existing documentation structure and standards
2. Use Google style docstrings for code documentation
3. Use Better Comments for enhanced code documentation
4. Update the documentation checklist as you complete items
5. Submit a pull request with your changes

## Next Steps

After exploring the documentation, you might want to:

1. Check the [Examples](examples/README.md) for practical usage examples
2. Read the [Architecture Documentation](architecture/README.md) for a deeper understanding of the system design
3. Explore the [API Reference](api_reference.md) for detailed information on classes and methods
4. Review the [Usage Guide](usage_guide.md) for comprehensive usage documentation