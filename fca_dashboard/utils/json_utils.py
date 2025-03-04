"""
JSON utility functions for common JSON data operations.

This module provides utility functions for JSON serialization, deserialization,
validation, formatting, and safe access following CLEAN principles:
- Clear: Functions have descriptive names and clear docstrings.
- Logical: Each function has a single, well-defined purpose.
- Efficient: Optimized for typical JSON-related tasks.
- Adaptable: Allow optional parameters for flexibility.
"""

import json
from typing import Any, Dict, Optional, TypeVar, Union

T = TypeVar("T")


def json_load(file_path: str, encoding: str = "utf-8") -> Any:
    """
    Load JSON data from a file.

    Args:
        file_path: Path to the JSON file.
        encoding: File encoding (default utf-8).

    Returns:
        Parsed JSON data.

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = json_load("data.json")
        >>> print(data)
        {'name': 'Bob'}
    """
    with open(file_path, "r", encoding=encoding) as f:
        return json.load(f)


def json_save(data: Any, file_path: str, encoding: str = "utf-8", indent: int = 2) -> None:
    """
    Save data as JSON to a file.

    Args:
        data: Data to serialize.
        file_path: Path to save the JSON file.
        encoding: File encoding (default utf-8).
        indent: Indentation spaces for formatting (default 2).

    Returns:
        None

    Raises:
        JSONDecodeError: if JSON is invalid.
        FileNotFoundError: if file does not exist.
    
    Example:
        >>> data = {"name": "Bob"}
        >>> json_save(data, "data.json")
    """
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def json_serialize(data: Any, indent: Optional[int] = None) -> str:
    """
    Serialize data to a JSON string.

    Args:
        data: Data to serialize.
        indent: Optional indentation for formatting.

    Returns:
        JSON-formatted string.

    Example:
        >>> json_serialize({"key": "value"})
        '{"key": "value"}'
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)


def json_deserialize(json_str: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Deserialize a JSON string into a Python object.

    Args:
        json_str: JSON-formatted string.
        default: Value to return if deserialization fails.

    Returns:
        Python data object or default.

    Example:
        >>> json_deserialize('{"name": "Bob"}')
        {'name': 'Bob'}
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return default


def json_is_valid(json_str: str) -> bool:
    """
    Check if a string is valid JSON.

    Args:
        json_str: String to validate.

    Returns:
        True if valid JSON, False otherwise.

    Example:
        >>> json_is_valid('{"valid": true}')
        True
        >>> json_is_valid('{invalid json}')
        False
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False


def pretty_print_json(data: Any) -> str:
    """
    Pretty-print JSON data with indentation.

    Args:
        data: JSON data (Python object).

    Returns:
        Pretty-printed JSON string.

    Example:
        >>> pretty_print_json({"key": "value"})
        '{\n  "key": "value"\n}'
    """
    return json.dumps(data, ensure_ascii=False, indent=2)


def safe_get(data: Dict, key: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely get a value from a dictionary.

    Args:
        data: Dictionary to extract value from.
        key: Key to look up.
        default: Default value if key is missing.

    Returns:
        Value associated with key or default.

    Example:
        >>> safe_get({"a": 1}, "a")
        1
        >>> safe_get({"a": 1}, "b", 0)
        0
    """
    return data.get(key, default)


def safe_get_nested(data: Dict, *keys: str, default: Optional[T] = None) -> Union[Any, T]:
    """
    Safely retrieve a nested value from a dictionary.

    Args:
        data: Nested dictionary.
        *keys: Sequence of keys for nested lookup.
        default: Default value if key path is missing.

    Returns:
        Nested value or default.

    Example:
        >>> safe_get_nested({"a": {"b": 2}}, "a", "b")
        2
        >>> safe_get_nested({"a": {"b": 2}}, "a", "c", default="missing")
        'missing'
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
