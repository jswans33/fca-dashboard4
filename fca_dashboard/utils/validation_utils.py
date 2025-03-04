"""
Validation utilities for common data formats.

This module provides functions to validate common data formats such as
email addresses, phone numbers, and URLs.
"""
import re
from typing import Any


def is_valid_email(email: Any) -> bool:
    """
    Validate if the input is a properly formatted email address.

    Args:
        email: The email address to validate.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    if not isinstance(email, str):
        return False

    # RFC 5322 compliant email regex pattern with additional validations
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}$'
    
    # Basic pattern match
    if not re.match(pattern, email):
        return False
    
    # Additional validations
    if '..' in email:  # No consecutive dots
        return False
    if email.endswith('.'):  # No trailing dot
        return False
    if ' ' in email:  # No spaces
        return False
    
    # Check domain part
    domain = email.split('@')[1]
    if domain.startswith('-') or domain.endswith('-'):  # No leading/trailing hyphens in domain
        return False
    
    # Check for hyphens at the end of domain parts
    domain_parts = domain.split('.')
    return all(not part.endswith('-') for part in domain_parts)


def is_valid_phone(phone: Any) -> bool:
    """
    Validate if the input is a properly formatted phone number.

    Accepts various formats including:
    - 10 digits: 1234567890
    - Hyphenated: 123-456-7890
    - Parentheses: (123) 456-7890
    - International: +1 123-456-7890
    - Dots: 123.456.7890
    - Spaces: 123 456 7890

    Args:
        phone: The phone number to validate.

    Returns:
        bool: True if the phone number is valid, False otherwise.
    """
    if not isinstance(phone, str):
        return False

    # Check for specific invalid formats first
    if phone == "":
        return False
    
    # Check for spaces around hyphens
    if " - " in phone:
        return False
    
    # Check for missing space after parentheses in format like (123)456-7890
    if re.search(r'\)[0-9]', phone):
        return False
    
    # Remove all non-alphanumeric characters for normalization
    normalized = re.sub(r'[^0-9+]', '', phone)
    
    # Check for letters in the phone number
    if re.search(r'[a-zA-Z]', phone):
        return False
    
    # Check for international format (starting with +)
    if normalized.startswith('+'):
        # International numbers should have at least 8 digits after the country code
        return len(normalized) >= 9 and normalized[1:].isdigit()
    
    # For US/Canada numbers, expect 10 digits
    return len(normalized) == 10 and normalized.isdigit()


def is_valid_url(url: Any) -> bool:
    """
    Validate if the input is a properly formatted URL.

    Validates URLs with http or https protocols.

    Args:
        url: The URL to validate.

    Returns:
        bool: True if the URL is valid, False otherwise.
    """
    if not isinstance(url, str):
        return False

    # Check for specific invalid formats first
    if url == "":
        return False
    
    # Check for spaces
    if ' ' in url:
        return False
    
    # Check for double dots
    if '..' in url:
        return False
    
    # Check for trailing dot
    if url.endswith('.'):
        return False
    
    # URL regex pattern that validates common URL formats
    pattern = r'^(https?:\/\/)' + \
              r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
              r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
              r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
              r'(#[-a-zA-Z0-9%_]+)?$'
    
    # Basic pattern match
    if not re.match(pattern, url):
        return False
    
    # Check for domain part
    domain_part = url.split('://')[1].split('/')[0].split(':')[0]
    return not (domain_part.startswith('-') or domain_part.endswith('-'))
