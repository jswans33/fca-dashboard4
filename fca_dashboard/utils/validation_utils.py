"""
Validation utilities for common data formats.

This module provides functions to validate common data formats such as
email addresses, phone numbers, URLs, and classification data formats.
"""
import re
from typing import Any, Dict, List, Optional, Union

import pandas as pd


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
              r'((([a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})|' + \
              r'(localhost)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))' + \
              r'(:\d+)?(\/[-a-zA-Z0-9%_.~#+]*)*' + \
              r'(\?[;&a-zA-Z0-9%_.~+=-]*)?' + \
              r'(#[-a-zA-Z0-9%_]+)?$'
    
    # Basic pattern match
    if not re.match(pattern, url):
        return False
    
    # Check for domain part
    domain_part = url.split('://')[1].split('/')[0].split(':')[0]
    return not (domain_part.startswith('-') or domain_part.endswith('-'))


def is_valid_omniclass_data(data: Union[str, pd.DataFrame], required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate if the input data follows the OmniClass unified training data format.
    
    Args:
        data: The data to validate, either a path to a CSV file or a pandas DataFrame.
        required_columns: List of required column names. If None, uses the default
                         required columns: ['OmniClass_Code', 'OmniClass_Title', 'Description'].
    
    Returns:
        Dict containing validation results:
            - 'valid': bool indicating if the data is valid
            - 'errors': List of error messages if any
            - 'stats': Dict with statistics about the data
    """
    errors = []
    stats = {}
    
    # Default required columns
    if required_columns is None:
        required_columns = ['OmniClass_Code', 'OmniClass_Title', 'Description']
    
    # Load data if it's a file path
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Failed to read CSV file: {str(e)}"],
                'stats': {}
            }
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        return {
            'valid': False,
            'errors': ["Input must be a DataFrame or a path to a CSV file"],
            'stats': {}
        }
    
    # Check if DataFrame is empty
    if df.empty:
        return {
            'valid': False,
            'errors': ["DataFrame is empty"],
            'stats': {'row_count': 0}
        }
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Collect statistics
    stats['row_count'] = len(df)
    stats['column_count'] = len(df.columns)
    stats['columns'] = df.columns.tolist()
    
    # Check for null values in required columns
    if not missing_columns:
        for col in required_columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null values")
            stats[f'{col}_null_count'] = null_count
    
    # Check OmniClass_Code format if the column exists
    if 'OmniClass_Code' in df.columns:
        # OmniClass codes can follow various patterns like XX-XX XX XX or XX-XX XX XX XX
        # We'll use a more flexible pattern that allows for variations
        invalid_codes = df[~df['OmniClass_Code'].str.match(r'^\d{2}-\d{2}([ ]\d{2})*$', na=True)]
        if not invalid_codes.empty:
            errors.append(f"Found {len(invalid_codes)} rows with invalid OmniClass_Code format")
            stats['invalid_code_count'] = len(invalid_codes)
            stats['invalid_code_examples'] = invalid_codes['OmniClass_Code'].head(5).tolist()
    
    # Check for duplicate OmniClass codes if the column exists
    if 'OmniClass_Code' in df.columns:
        duplicates = df['OmniClass_Code'].duplicated()
        duplicate_count = duplicates.sum()
        if duplicate_count > 0:
            errors.append(f"Found {duplicate_count} duplicate OmniClass_Code values")
            stats['duplicate_code_count'] = duplicate_count
            stats['duplicate_code_examples'] = df[duplicates]['OmniClass_Code'].head(5).tolist()
    
    # Check for empty titles if the column exists
    if 'OmniClass_Title' in df.columns:
        empty_titles = df['OmniClass_Title'].isna() | (df['OmniClass_Title'] == '')
        empty_title_count = empty_titles.sum()
        if empty_title_count > 0:
            errors.append(f"Found {empty_title_count} rows with empty OmniClass_Title")
            stats['empty_title_count'] = empty_title_count
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'stats': stats
    }
