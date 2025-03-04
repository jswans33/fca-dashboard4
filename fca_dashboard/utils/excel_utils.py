"""
Excel utility module for the FCA Dashboard application.

This module provides backward compatibility for the refactored Excel utilities.
All functions and classes are re-exported from the new excel package.

DEPRECATED: Use the new excel package instead:
    from fca_dashboard.utils.excel import ...
"""

import warnings

# Show deprecation warning
warnings.warn(
    "The excel_utils module is deprecated. Use the new excel package instead: "
    "from fca_dashboard.utils.excel import ...",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all functions and classes from the excel package
from fca_dashboard.utils.excel import (
    # Base
    ExcelUtilError,
    # Analysis utils
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    # Validation utils
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    clean_sheet_name,
    # Conversion utils
    convert_excel_to_csv,
    detect_duplicate_columns,
    detect_empty_rows,
    detect_header_row,
    detect_unnamed_columns,
    extract_excel_with_config,
    # Column utils
    get_column_names,
    # File utils
    get_excel_file_type,
    # Sheet utils
    get_sheet_names,
    is_excel_file,
    is_valid_excel_file,
    load_excel_config,
    merge_excel_files,
    normalize_sheet_names,
    # Extraction utils
    read_excel_with_header_detection,
    save_excel_to_database,
    validate_columns_exist,
    validate_dataframe,
)

# For backward compatibility
__all__ = [
    # Base
    'ExcelUtilError',
    
    # File utils
    'get_excel_file_type',
    'is_excel_file',
    'is_valid_excel_file',
    
    # Sheet utils
    'get_sheet_names',
    'clean_sheet_name',
    'normalize_sheet_names',
    
    # Column utils
    'get_column_names',
    'validate_columns_exist',
    
    # Conversion utils
    'convert_excel_to_csv',
    'merge_excel_files',
    'save_excel_to_database',
    
    # Analysis utils
    'analyze_excel_structure',
    'analyze_unique_values',
    'analyze_column_statistics',
    'analyze_text_columns',
    'detect_empty_rows',
    'detect_header_row',
    'detect_duplicate_columns',
    'detect_unnamed_columns',
    
    # Extraction utils
    'read_excel_with_header_detection',
    'extract_excel_with_config',
    'load_excel_config',
    
    # Validation utils
    'check_missing_values',
    'check_duplicate_rows',
    'check_value_ranges',
    'check_data_types',
    'validate_dataframe',
]