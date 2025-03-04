"""
Date and time utility functions for common operations.

This module provides a collection of utility functions for date and time manipulation
that follow CLEAN code principles:
- Clear: Functions have descriptive names and docstrings
- Logical: Each function has a single, well-defined purpose
- Efficient: Operations are optimized for performance
- Adaptable: Functions accept optional parameters for flexibility
"""
import datetime
from typing import Optional, Union

from dateutil import parser


def format_date(
    date: Optional[datetime.datetime], 
    format_str: str = "%b %d, %Y", 
    default: str = ""
) -> str:
    """
    Format a datetime object into a readable string.

    Args:
        date: The datetime object to format.
        format_str: The format string to use (default: "%b %d, %Y").
        default: The default value to return if date is None.

    Returns:
        A formatted date string or the default value if date is None.

    Examples:
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0))
        'May 15, 2023'
        >>> format_date(datetime.datetime(2023, 5, 15, 14, 30, 0), "%Y-%m-%d")
        '2023-05-15'
    """
    if date is None:
        return default
    
    return date.strftime(format_str)


def time_since(date: Optional[datetime.datetime], default: str = "") -> str:
    """
    Calculate the relative time between the given date and now.

    Args:
        date: The datetime to calculate the time since.
        default: The default value to return if date is None.

    Returns:
        A human-readable string representing the time difference (e.g., "2 hours ago").

    Examples:
        >>> # Assuming current time is 2023-05-15 14:30:00
        >>> time_since(datetime.datetime(2023, 5, 15, 13, 30, 0))
        '1 hour ago'
        >>> time_since(datetime.datetime(2023, 5, 14, 14, 30, 0))
        '1 day ago'
    """
    if date is None:
        return default
    
    now = datetime.datetime.now()
    diff = now - date
    
    # Handle future dates
    if diff.total_seconds() < 0:
        diff = -diff
        is_future = True
    else:
        is_future = False
    
    seconds = int(diff.total_seconds())
    minutes = seconds // 60
    hours = minutes // 60
    days = diff.days
    months = days // 30  # Approximate
    years = days // 365  # Approximate
    
    if years > 0:
        time_str = f"{years} year{'s' if years != 1 else ''}"
    elif months > 0:
        time_str = f"{months} month{'s' if months != 1 else ''}"
    elif days > 0:
        time_str = f"{days} day{'s' if days != 1 else ''}"
    elif hours > 0:
        time_str = f"{hours} hour{'s' if hours != 1 else ''}"
    elif minutes > 0:
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"
    
    return f"in {time_str}" if is_future else f"{time_str} ago"


def parse_date(
    date_str: Optional[Union[str, datetime.datetime]], 
    format: Optional[str] = None
) -> Optional[datetime.datetime]:
    """
    Convert a string into a datetime object.

    Args:
        date_str: The string to parse or a datetime object to return as-is.
        format: Optional format string for parsing (if None, tries to infer format).

    Returns:
        A datetime object or None if the input is None or empty.

    Raises:
        ValueError: If the string cannot be parsed as a date.

    Examples:
        >>> parse_date("2023-05-15")
        datetime.datetime(2023, 5, 15, 0, 0)
        >>> parse_date("15/05/2023", format="%d/%m/%Y")
        datetime.datetime(2023, 5, 15, 0, 0)
    """
    if date_str is None or (isinstance(date_str, str) and not date_str.strip()):
        return None
    
    if isinstance(date_str, datetime.datetime):
        return date_str
    
    if format:
        return datetime.datetime.strptime(date_str, format)
    
    # Handle common natural language date expressions
    if isinstance(date_str, str):
        date_str = date_str.lower().strip()
        now = datetime.datetime.now()
        
        if date_str == "today":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "yesterday":
            return (now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == "tomorrow":
            return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str.endswith(" days ago"):
            try:
                days = int(date_str.split(" ")[0])
                return (now - datetime.timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            except (ValueError, IndexError):
                pass
    
    # Try to parse using dateutil's flexible parser
    try:
        return parser.parse(date_str)
    except (ValueError, parser.ParserError) as err:
        raise ValueError(f"Could not parse date string: {date_str}") from err
