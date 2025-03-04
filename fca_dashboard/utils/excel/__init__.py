"""
Excel utilities package for the FCA Dashboard application.

This package provides utilities for working with Excel files,
including file type detection, validation, data extraction, and analysis.
"""

# Re-export all functions and classes from the modules
from fca_dashboard.utils.excel.analysis_utils import (
    analyze_column_statistics,
    analyze_excel_structure,
    analyze_text_columns,
    analyze_unique_values,
    detect_duplicate_columns,
    detect_empty_rows,
    detect_header_row,
    detect_unnamed_columns,
)
from fca_dashboard.utils.excel.base import ExcelUtilError
from fca_dashboard.utils.excel.column_utils import (
    get_column_names,
    validate_columns_exist,
)
from fca_dashboard.utils.excel.conversion_utils import (
    convert_excel_to_csv,
    get_database_schema,
    merge_excel_files,
    save_excel_to_database,
)
from fca_dashboard.utils.excel.extraction_utils import (
    extract_excel_with_config,
    load_excel_config,
    read_excel_with_header_detection,
)
from fca_dashboard.utils.excel.file_utils import (
    get_excel_file_type,
    is_excel_file,
    is_valid_excel_file,
)
from fca_dashboard.utils.excel.sheet_utils import (
    clean_sheet_name,
    get_sheet_names,
    normalize_sheet_names,
)
from fca_dashboard.utils.excel.validation_utils import (
    check_data_types,
    check_duplicate_rows,
    check_missing_values,
    check_value_ranges,
    validate_dataframe,
)

# For backward compatibility
# This allows existing code to continue working with the old import paths
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
    'get_database_schema',
    
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