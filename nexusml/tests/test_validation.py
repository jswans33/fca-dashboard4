"""
Test script for the validation module.

This script tests the functionality of the validation module components.
"""

import unittest
import pandas as pd
import numpy as np

from nexusml.core.validation import (
    ValidationLevel,
    ValidationResult,
    ValidationReport,
    ValidationRule,
    ColumnExistenceRule,
    ColumnTypeRule,
    NonNullRule,
    ValueRangeRule,
    UniqueValuesRule,
    AllowedValuesRule,
    RegexPatternRule,
    CrossColumnComparisonRule,
    RowCountRule,
    ColumnCountRule,
    BaseValidator,
    ConfigDrivenValidator,
    BasicColumnValidator,
    BasicRowValidator,
    BasicDataFrameValidator,
    CompositeValidator,
    SchemaValidator,
)


class TestValidationRules(unittest.TestCase):
    """Test case for validation rules."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test DataFrame
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, np.nan],
            'score': [90, 85, 95, 80, 75],
            'grade': ['A', 'B', 'A', 'B', 'C'],
        })
    
    def test_column_existence_rule(self):
        """Test ColumnExistenceRule."""
        # Test for existing column
        rule = ColumnExistenceRule('name')
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for non-existing column
        rule = ColumnExistenceRule('email')
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
        
        # Test for optional column
        rule = ColumnExistenceRule('email', required=False)
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
    
    def test_column_type_rule(self):
        """Test ColumnTypeRule."""
        # Test for correct type
        rule = ColumnTypeRule('age', 'float')
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for incorrect type
        rule = ColumnTypeRule('name', 'int')
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
    
    def test_non_null_rule(self):
        """Test NonNullRule."""
        # Test for column with no nulls
        rule = NonNullRule('name')
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for column with nulls
        rule = NonNullRule('age')
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
        
        # Test with max_null_fraction
        rule = NonNullRule('age', max_null_fraction=0.2)
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
    
    def test_value_range_rule(self):
        """Test ValueRangeRule."""
        # Test for values within range
        rule = ValueRangeRule('age', min_value=20, max_value=50)
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for values outside range
        rule = ValueRangeRule('age', min_value=30, max_value=40)
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
    
    def test_unique_values_rule(self):
        """Test UniqueValuesRule."""
        # Test for unique values
        rule = UniqueValuesRule('id')
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for non-unique values
        # Add a duplicate value
        df_with_duplicates = self.df.copy()
        df_with_duplicates.loc[5] = [1, 'Alice', 25, 90, 'A']  # Duplicate id
        
        rule = UniqueValuesRule('id')
        result = rule.validate(df_with_duplicates)
        self.assertFalse(result.valid)
    
    def test_allowed_values_rule(self):
        """Test AllowedValuesRule."""
        # Test for allowed values
        rule = AllowedValuesRule('grade', ['A', 'B', 'C'])
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for disallowed values
        rule = AllowedValuesRule('grade', ['A', 'B'])
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
    
    def test_regex_pattern_rule(self):
        """Test RegexPatternRule."""
        # Test for matching pattern
        rule = RegexPatternRule('name', r'^[A-Z][a-z]+$')
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for non-matching pattern
        # Add a name that doesn't match the pattern
        df_with_invalid_name = self.df.copy()
        df_with_invalid_name.loc[5] = [6, '123', 45, 70, 'D']
        
        rule = RegexPatternRule('name', r'^[A-Z][a-z]+$')
        result = rule.validate(df_with_invalid_name)
        self.assertFalse(result.valid)
    
    def test_cross_column_comparison_rule(self):
        """Test CrossColumnComparisonRule."""
        # Test for valid comparison
        df_with_comparison = self.df.copy()
        df_with_comparison['min_score'] = [70, 75, 80, 70, 70]
        
        rule = CrossColumnComparisonRule('score', 'min_score', 'ge')
        result = rule.validate(df_with_comparison)
        self.assertTrue(result.valid)
        
        # Test for invalid comparison
        df_with_comparison.loc[2, 'min_score'] = 100  # Make one comparison fail
        
        rule = CrossColumnComparisonRule('score', 'min_score', 'ge')
        result = rule.validate(df_with_comparison)
        self.assertFalse(result.valid)
    
    def test_row_count_rule(self):
        """Test RowCountRule."""
        # Test for valid row count
        rule = RowCountRule(min_rows=3, max_rows=10)
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for invalid row count
        rule = RowCountRule(min_rows=10)
        result = rule.validate(self.df)
        self.assertFalse(result.valid)
    
    def test_column_count_rule(self):
        """Test ColumnCountRule."""
        # Test for valid column count
        rule = ColumnCountRule(min_columns=3, max_columns=10)
        result = rule.validate(self.df)
        self.assertTrue(result.valid)
        
        # Test for invalid column count
        rule = ColumnCountRule(min_columns=10)
        result = rule.validate(self.df)
        self.assertFalse(result.valid)


class TestValidators(unittest.TestCase):
    """Test case for validators."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test DataFrame
        self.df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, np.nan],
            'score': [90, 85, 95, 80, 75],
            'grade': ['A', 'B', 'A', 'B', 'C'],
        })
    
    def test_base_validator(self):
        """Test BaseValidator."""
        validator = BaseValidator()
        validator.add_rule(ColumnExistenceRule('id'))
        validator.add_rule(ColumnExistenceRule('name'))
        
        report = validator.validate(self.df)
        self.assertTrue(report.is_valid())
        self.assertEqual(len(report.results), 2)
    
    def test_config_driven_validator(self):
        """Test ConfigDrivenValidator."""
        config = {
            'required_columns': [
                {'name': 'id', 'data_type': 'int', 'required': True},
                {'name': 'name', 'data_type': 'str', 'required': True},
                {'name': 'age', 'data_type': 'float', 'min_value': 0, 'max_value': 100},
                {'name': 'grade', 'allowed_values': ['A', 'B', 'C', 'D', 'F']},
            ],
            'row_count': {'min': 1},
        }
        
        validator = ConfigDrivenValidator(config)
        report = validator.validate(self.df)
        
        # Should pass all validations
        self.assertTrue(report.is_valid())
        
        # Add an invalid configuration
        config['required_columns'].append({'name': 'email', 'required': True})
        
        validator = ConfigDrivenValidator(config)
        report = validator.validate(self.df)
        
        # Should fail because 'email' column is missing
        self.assertFalse(report.is_valid())
    
    def test_basic_column_validator(self):
        """Test BasicColumnValidator."""
        validator = BasicColumnValidator('age')
        validator.add_rule(NonNullRule('age'))
        
        report = validator.validate_column(self.df, 'age')
        
        # Should fail because 'age' column has null values
        self.assertFalse(report.is_valid())
        
        # Test with a column that has no nulls
        validator = BasicColumnValidator('name')
        validator.add_rule(NonNullRule('name'))
        
        report = validator.validate_column(self.df, 'name')
        
        # Should pass
        self.assertTrue(report.is_valid())
    
    def test_basic_row_validator(self):
        """Test BasicRowValidator."""
        # Create a row validator that checks if age is greater than 30
        validator = BasicRowValidator()
        
        # We need to create a custom rule for this
        class AgeGreaterThan30Rule(ValidationRule):
            def validate(self, data):
                if not isinstance(data, pd.DataFrame) or len(data) != 1:
                    return ValidationResult(False, ValidationLevel.ERROR, "Expected a single row DataFrame")
                
                age = data['age'].iloc[0]
                if pd.isna(age):
                    return ValidationResult(False, ValidationLevel.ERROR, "Age is null")
                
                if age > 30:
                    return ValidationResult(True, ValidationLevel.INFO, f"Age {age} is greater than 30")
                else:
                    return ValidationResult(False, ValidationLevel.ERROR, f"Age {age} is not greater than 30")
            
            def get_name(self):
                return "AgeGreaterThan30Rule"
            
            def get_description(self):
                return "Checks if age is greater than 30"
        
        validator.add_rule(AgeGreaterThan30Rule())
        
        # Test with a row where age > 30
        row = self.df.iloc[2]  # Charlie, age 35
        report = validator.validate_row(row)
        self.assertTrue(report.is_valid())
        
        # Test with a row where age <= 30
        row = self.df.iloc[0]  # Alice, age 25
        report = validator.validate_row(row)
        self.assertFalse(report.is_valid())
    
    def test_basic_dataframe_validator(self):
        """Test BasicDataFrameValidator."""
        validator = BasicDataFrameValidator()
        validator.add_rule(RowCountRule(min_rows=1))
        validator.add_rule(ColumnCountRule(min_columns=3))
        
        report = validator.validate_dataframe(self.df)
        
        # Should pass
        self.assertTrue(report.is_valid())
        
        # Test with invalid DataFrame
        empty_df = pd.DataFrame()
        report = validator.validate_dataframe(empty_df)
        
        # Should fail
        self.assertFalse(report.is_valid())
    
    def test_composite_validator(self):
        """Test CompositeValidator."""
        # Create individual validators
        column_validator = BasicColumnValidator('age')
        column_validator.add_rule(NonNullRule('age'))
        
        dataframe_validator = BasicDataFrameValidator()
        dataframe_validator.add_rule(RowCountRule(min_rows=1))
        
        # Create composite validator
        validator = CompositeValidator()
        validator.add_validator(column_validator)
        validator.add_validator(dataframe_validator)
        
        report = validator.validate(self.df)
        
        # Should fail because 'age' column has null values
        self.assertFalse(report.is_valid())
        
        # Fix the null values
        df_fixed = self.df.copy()
        df_fixed['age'] = df_fixed['age'].fillna(0)
        
        report = validator.validate(df_fixed)
        
        # Should pass
        self.assertTrue(report.is_valid())
    
    def test_schema_validator(self):
        """Test SchemaValidator."""
        schema = {
            'id': 'int',
            'name': 'str',
            'age': 'float',
            'score': 'int',
            'grade': 'str',
        }
        
        validator = SchemaValidator(schema)
        report = validator.validate_dataframe(self.df)
        
        # Should pass
        self.assertTrue(report.is_valid())
        
        # Test with invalid schema
        schema['email'] = 'str'
        
        validator = SchemaValidator(schema)
        report = validator.validate_dataframe(self.df)
        
        # Should fail because 'email' column is missing
        self.assertFalse(report.is_valid())


if __name__ == '__main__':
    unittest.main()