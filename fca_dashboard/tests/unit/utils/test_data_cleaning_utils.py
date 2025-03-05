"""
Unit tests for the data cleaning utilities.

This module contains tests for the data cleaning utilities used in the FCA Dashboard application.
"""

import os
import sys
import unittest
from pathlib import Path

import pandas as pd
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from fca_dashboard.utils.data_cleaning_utils import (
    DataCleaningError,
    clean_dataframe,
    find_header_row,
    remove_copyright_rows,
    standardize_column_names,
)


class TestDataCleaningUtils(unittest.TestCase):
    """Test cases for data cleaning utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample DataFrames for testing
        
        # Sample DataFrame with copyright and header rows
        self.df_with_copyright = pd.DataFrame({
            'Column1': [
                'OmniClass® Copyright © 2022, Construction Specifications Institute, Alexandria, VA',
                '2013-02-26 Final - National Standard - OmniClass Table 11',
                'OmniClass Number',
                '11-11 00 00',
                '11-12 00 00'
            ],
            'Column2': [
                'All rights reserved.',
                '',
                'OmniClass Title',
                'Assembly Facility',
                'Education Facility'
            ]
        })
        
        # Sample DataFrame with header row but no copyright
        self.df_with_header = pd.DataFrame({
            'Column1': [
                'OmniClass Number',
                '11-11 00 00',
                '11-12 00 00'
            ],
            'Column2': [
                'OmniClass Title',
                'Assembly Facility',
                'Education Facility'
            ]
        })
        
        # Sample DataFrame with no header row
        self.df_no_header = pd.DataFrame({
            'Column1': [
                '11-11 00 00',
                '11-12 00 00'
            ],
            'Column2': [
                'Assembly Facility',
                'Education Facility'
            ]
        })

    def test_find_header_row(self):
        """Test finding the header row in a DataFrame."""
        # Test with copyright and header rows
        header_idx = find_header_row(self.df_with_copyright)
        self.assertEqual(header_idx, 2)
        
        # Test with header row but no copyright
        header_idx = find_header_row(self.df_with_header)
        self.assertEqual(header_idx, 0)
        
        # Test with no header row
        header_idx = find_header_row(self.df_no_header)
        self.assertIsNone(header_idx)
        
        # Test with custom header patterns
        header_idx = find_header_row(self.df_with_copyright, header_patterns=['omniclass title'])
        self.assertEqual(header_idx, 2)

    def test_remove_copyright_rows(self):
        """Test removing copyright rows from a DataFrame."""
        # Test with copyright rows
        df_cleaned = remove_copyright_rows(self.df_with_copyright)
        self.assertEqual(len(df_cleaned), 3)  # Should remove the first 2 rows
        
        # Test with no copyright rows
        df_cleaned = remove_copyright_rows(self.df_no_header)
        self.assertEqual(len(df_cleaned), 2)  # Should remain unchanged
        
        # Test with custom patterns
        df_cleaned = remove_copyright_rows(self.df_with_copyright, patterns=['National Standard'])
        self.assertEqual(len(df_cleaned), 4)  # Should remove only the second row

    def test_standardize_column_names(self):
        """Test standardizing column names in a DataFrame."""
        # Test with standard mapping
        df = pd.DataFrame({
            'OmniClass Number': ['11-11 00 00'],
            'OmniClass Title': ['Assembly Facility']
        })
        df_standardized = standardize_column_names(df)
        self.assertIn('OmniClass_Code', df_standardized.columns)
        self.assertIn('OmniClass_Title', df_standardized.columns)
        
        # Test with custom mapping
        df = pd.DataFrame({
            'Number': ['11-11 00 00'],
            'Title': ['Assembly Facility']
        })
        mapping = {'Number': 'Code', 'Title': 'Description'}
        df_standardized = standardize_column_names(df, column_mapping=mapping)
        self.assertIn('Code', df_standardized.columns)
        self.assertIn('Description', df_standardized.columns)

    def test_clean_dataframe(self):
        """Test cleaning a DataFrame."""
        # Test with copyright and header rows
        df_cleaned = clean_dataframe(self.df_with_copyright)
        self.assertEqual(len(df_cleaned), 2)  # Should have 2 data rows
        self.assertIn('OmniClass_Code', df_cleaned.columns)
        self.assertIn('OmniClass_Title', df_cleaned.columns)
        
        # Test with header row but no copyright
        df_cleaned = clean_dataframe(self.df_with_header)
        self.assertEqual(len(df_cleaned), 2)  # Should have 2 data rows
        self.assertIn('OmniClass_Code', df_cleaned.columns)
        self.assertIn('OmniClass_Title', df_cleaned.columns)
        
        # Test with no header row
        with self.assertRaises(DataCleaningError):
            clean_dataframe(self.df_no_header)
        
        # Test with custom column mapping
        mapping = {'Column1': 'Code', 'Column2': 'Description'}
        df_cleaned = clean_dataframe(self.df_with_copyright, column_mapping=mapping)
        self.assertIn('Code', df_cleaned.columns)
        self.assertIn('Description', df_cleaned.columns)


if __name__ == '__main__':
    unittest.main()