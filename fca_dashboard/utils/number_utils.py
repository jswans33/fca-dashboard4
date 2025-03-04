"""
Number utility functions for common numeric operations.

This module provides a collection of utility functions for number formatting,
rounding, and random number generation that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Union, overload

# Type aliases for numeric types
NumericType = Union[int, float, Decimal]


def format_currency(
    value: Optional[NumericType],
    symbol: str = "$",
    decimal_places: int = 2,
    thousands_sep: str = ",",
    decimal_sep: str = ".",
    default: str = "",
) -> str:
    """
    Format a number as a currency string.

    Args:
        value: The numeric value to format.
        symbol: Currency symbol to prepend (default: "$").
        decimal_places: Number of decimal places to show (default: 2).
        thousands_sep: Character to use as thousands separator (default: ",").
        decimal_sep: Character to use as decimal separator (default: ".").
        default: Value to return if input is None (default: "").

    Returns:
        Formatted currency string.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> format_currency(1234.56)
        '$1,234.56'
        >>> format_currency(1234.56, symbol="€", decimal_sep=",")
        '€1,234,56'
    """
    if value is None:
        return default

    # Validate input type
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Handle negative values
    is_negative = value < 0
    abs_value = abs(value)

    # Round to specified decimal places
    if isinstance(value, Decimal):
        rounded_value = abs_value.quantize(Decimal(f"0.{'0' * decimal_places}"), rounding=ROUND_HALF_UP)
    else:
        rounded_value = round(abs_value, decimal_places)

    # Convert to string and split into integer and decimal parts
    str_value = str(rounded_value)
    if "." in str_value:
        int_part, dec_part = str_value.split(".")
    else:
        int_part, dec_part = str_value, ""

    # Format integer part with thousands separator
    formatted_int = ""
    for i, char in enumerate(reversed(int_part)):
        if i > 0 and i % 3 == 0:
            formatted_int = thousands_sep + formatted_int
        formatted_int = char + formatted_int

    # Format decimal part
    if decimal_places > 0:
        # Pad with zeros if needed
        dec_part = dec_part.ljust(decimal_places, "0")
        # Truncate if too long
        dec_part = dec_part[:decimal_places]
        formatted_value = formatted_int + decimal_sep + dec_part
    else:
        formatted_value = formatted_int

    # Add currency symbol and handle negative values
    if is_negative:
        return f"-{symbol}{formatted_value}"
    else:
        return f"{symbol}{formatted_value}"


def round_to(value: NumericType, places: int = 0) -> NumericType:
    """
    Round a number to a specified number of decimal places with ROUND_HALF_UP rounding.

    This function handles both positive and negative decimal places:
    - Positive places round to that many decimal places
    - Zero places round to the nearest integer
    - Negative places round to tens, hundreds, etc.

    Args:
        value: The numeric value to round.
        places: Number of decimal places to round to (default: 0).

    Returns:
        Rounded value of the same type as the input.

    Raises:
        TypeError: If value is not a numeric type.

    Examples:
        >>> round_to(1.234, 2)
        1.23
        >>> round_to(1.235, 2)
        1.24
        >>> round_to(123, -1)
        120
        >>> round_to(125, -1)
        130
    """
    if not isinstance(value, (int, float, Decimal)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")

    # Preserve the original type
    original_type = type(value)
    
    # Convert to Decimal for consistent rounding behavior
    if not isinstance(value, Decimal):
        decimal_value = Decimal(str(value))
    else:
        decimal_value = value
    
    # Calculate the factor based on places
    factor = Decimal("10") ** places
    
    if places >= 0:
        # For positive places (decimal places)
        result = decimal_value.quantize(Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP)
    else:
        # For negative places (tens, hundreds, etc.)
        # First divide by factor, round to integer, then multiply back
        factor = Decimal("10") ** abs(places)
        result = (decimal_value / factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * factor
    
    # Return the result in the original type
    if original_type == int or (places == 0 and original_type == float):
        # Convert to int if original was int or if rounding to integer (places=0)
        return int(result)
    elif original_type == float:
        return float(result)
    else:
        return result  # Already a Decimal


def random_number(min_value: int, max_value: int) -> int:
    """
    Generate a random integer within a specified range.

    Args:
        min_value: The minimum value (inclusive).
        max_value: The maximum value (inclusive).

    Returns:
        A random integer between min_value and max_value (inclusive).

    Raises:
        ValueError: If min_value is greater than max_value.
        TypeError: If min_value or max_value is not an integer.

    Examples:
        >>> # Returns a random number between 1 and 10
        >>> random_number(1, 10)
        7
        >>> # Returns a random number between -10 and 10
        >>> random_number(-10, 10)
        -3
    """
    # Validate input types
    if not isinstance(min_value, int):
        raise TypeError(f"min_value must be an integer, got {type(min_value).__name__}")
    if not isinstance(max_value, int):
        raise TypeError(f"max_value must be an integer, got {type(max_value).__name__}")

    # Validate range
    if min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")

    return random.randint(min_value, max_value)
