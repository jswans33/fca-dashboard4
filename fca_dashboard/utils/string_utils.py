"""
String utility functions for common text operations.

This module provides a collection of utility functions for string manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import re
import unicodedata
from typing import Optional


def capitalize(text: str) -> str:
    """
    Capitalize the first letter of a string, preserving leading whitespace.

    Args:
        text: The string to capitalize.

    Returns:
        A string with the first non-space character capitalized.

    Examples:
        >>> capitalize("hello")
        'Hello'
        >>> capitalize("  hello")
        '  Hello'
        >>> capitalize("123abc")
        '123abc'
        >>> capitalize("")
        ''
    """
    if not text:
        return ""
    
    # If the string starts with non-alphabetic characters (except whitespace),
    # return it unchanged
    if text.strip() and not text.strip()[0].isalpha():
        return text
    
    # Preserve leading whitespace and capitalize the first non-space character
    leading_spaces = len(text) - len(text.lstrip())
    return text[:leading_spaces] + text[leading_spaces:].capitalize()


def slugify(text: str) -> str:
    """
    Convert text into a URL-friendly slug.

    This function:
    1. Converts to lowercase
    2. Removes accents/diacritics
    3. Replaces spaces and special characters with hyphens

    Args:
        text: The string to convert to a slug.

    Returns:
        URL-friendly slug.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("Héllo, Wörld!")
        'hello-world'
    """
    if not text:
        return ""

    # Normalize and remove accents
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))

    # Replace non-alphanumeric characters with hyphens
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    
    # Replace multiple hyphens with a single hyphen
    text = re.sub(r"-+", "-", text)
    
    return text


def truncate(text: str, length: int, suffix: str = "...") -> str:
    """
    Limit the length of a string and add a suffix if truncated.

    Args:
        text: The string to truncate.
        length: Maximum allowed length before truncation.
        suffix: String appended after truncation (default "...").

    Returns:
        Truncated string.

    Examples:
        >>> truncate("Hello World", 5)
        'Hello...'
        >>> truncate("Hello", 10)
        'Hello'
    """
    if not text:
        return ""

    if length <= 0:
        return suffix

    return text if len(text) <= length else text[:length] + suffix


def is_empty(text: Optional[str]) -> bool:
    """
    Check if a string is empty or contains only whitespace.

    Args:
        text: The string to check.

    Returns:
        True if empty or whitespace, False otherwise.

    Raises:
        TypeError: if text is None.

    Examples:
        >>> is_empty("   ")
        True
        >>> is_empty("Hello")
        False
    """
    if text is None:
        raise TypeError("Cannot check emptiness of None")

    return not bool(text.strip())
