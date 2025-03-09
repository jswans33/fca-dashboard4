"""
Validation Example

This script demonstrates how to use the validation components in the NexusML suite.
"""

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
    validate_dataframe,
    validate_column,
)


def main():
    """Run the validation example."""
    print("NexusML Validation Example")
    print("==========================")
    
    # Create a sample DataFrame
    print("\nCreating sample DataFrame...")
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, np.nan],
        'score': [90, 85, 95, 80, 75],
        'grade': ['A', 'B', 'A', 'B', 'C'],
    })
    print(df)
    
    # Example 1: Using individual validation rules
    print("\nExample 1: Using individual validation rules")
    print("-------------------------------------------")
    
    # Check if 'name' column exists
    rule = ColumnExistenceRule('name')
    result = rule.validate(df)
    print(f"Column 'name' exists: {result.valid}")
    
    # Check if 'email' column exists
    rule = ColumnExistenceRule('email')
    result = rule.validate(df)
    print(f"Column 'email' exists: {result.valid}")
    
    # Check if 'age' column has no null values
    rule = NonNullRule('age')
    result = rule.validate(df)
    print(f"Column 'age' has no nulls: {result.valid}")
    
    # Check if 'age' column has values between 20 and 50
    rule = ValueRangeRule('age', min_value=20, max_value=50)
    result = rule.validate(df)
    print(f"Column 'age' values between 20 and 50: {result.valid}")
    
    # Example 2: Using a validator with multiple rules
    print("\nExample 2: Using a validator with multiple rules")
    print("----------------------------------------------")
    
    validator = BaseValidator("SampleValidator")
    validator.add_rule(ColumnExistenceRule('id'))
    validator.add_rule(ColumnExistenceRule('name'))
    validator.add_rule(ColumnExistenceRule('age'))
    validator.add_rule(ColumnTypeRule('id', 'int'))
    validator.add_rule(ColumnTypeRule('name', 'str'))
    validator.add_rule(ColumnTypeRule('age', 'float'))
    validator.add_rule(UniqueValuesRule('id'))
    
    report = validator.validate(df)
    print(f"Validation passed: {report.is_valid()}")
    print(f"Number of errors: {len(report.get_errors())}")
    print(f"Number of warnings: {len(report.get_warnings())}")
    print(f"Number of info messages: {len(report.get_info())}")
    
    # Print all validation results
    print("\nValidation results:")
    for result in report.results:
        print(f"  {result}")
    
    # Example 3: Using a configuration-driven validator
    print("\nExample 3: Using a configuration-driven validator")
    print("-----------------------------------------------")
    
    config = {
        'required_columns': [
            {'name': 'id', 'data_type': 'int', 'required': True},
            {'name': 'name', 'data_type': 'str', 'required': True},
            {'name': 'age', 'data_type': 'float', 'min_value': 0, 'max_value': 100},
            {'name': 'grade', 'allowed_values': ['A', 'B', 'C', 'D', 'F']},
            {'name': 'email', 'required': False},  # Optional column
        ],
        'row_count': {'min': 1},
        'column_count': {'min': 4},
    }
    
    validator = ConfigDrivenValidator(config)
    report = validator.validate(df)
    
    print(f"Validation passed: {report.is_valid()}")
    print(f"Number of errors: {len(report.get_errors())}")
    print(f"Number of warnings: {len(report.get_warnings())}")
    print(f"Number of info messages: {len(report.get_info())}")
    
    # Print error messages
    if not report.is_valid():
        print("\nValidation errors:")
        for error in report.get_errors():
            print(f"  {error.message}")
    
    # Example 4: Using a schema validator
    print("\nExample 4: Using a schema validator")
    print("----------------------------------")
    
    schema = {
        'id': 'int',
        'name': 'str',
        'age': 'float',
        'score': 'int',
        'grade': 'str',
    }
    
    validator = SchemaValidator(schema)
    report = validator.validate_dataframe(df)
    
    print(f"Schema validation passed: {report.is_valid()}")
    
    # Example 5: Using convenience functions
    print("\nExample 5: Using convenience functions")
    print("------------------------------------")
    
    # Validate a specific column
    report = validate_column(df, 'age', config={'required': True, 'type': 'float', 'min_value': 0})
    print(f"Column 'age' validation passed: {report.is_valid()}")
    
    # Validate the entire DataFrame
    report = validate_dataframe(df, config=config)
    print(f"DataFrame validation passed: {report.is_valid()}")
    
    # Example 6: Creating a custom validation rule
    print("\nExample 6: Creating a custom validation rule")
    print("------------------------------------------")
    
    # Define a custom rule that checks if the average score is above a threshold
    class AverageScoreRule(ValidationRule):
        def __init__(self, column, threshold, level=ValidationLevel.ERROR):
            self.column = column
            self.threshold = threshold
            self.level = level
        
        def validate(self, data):
            if not isinstance(data, pd.DataFrame):
                return ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Expected DataFrame, got {type(data).__name__}",
                    context={"column": self.column}
                )
            
            if self.column not in data.columns:
                return ValidationResult(
                    valid=False,
                    level=self.level,
                    message=f"Column '{self.column}' does not exist",
                    context={"column": self.column}
                )
            
            average = data[self.column].mean()
            
            if average >= self.threshold:
                return ValidationResult(
                    valid=True,
                    level=ValidationLevel.INFO,
                    message=f"Average {self.column} ({average:.2f}) is above threshold ({self.threshold})",
                    context={"column": self.column, "average": average, "threshold": self.threshold}
                )
            else:
                return ValidationResult(
                    valid=False,
                    level=self.level,
                    message=f"Average {self.column} ({average:.2f}) is below threshold ({self.threshold})",
                    context={"column": self.column, "average": average, "threshold": self.threshold}
                )
        
        def get_name(self):
            return f"AverageScore({self.column}, {self.threshold})"
        
        def get_description(self):
            return f"Checks if the average value in column '{self.column}' is above {self.threshold}"
    
    # Use the custom rule
    rule = AverageScoreRule('score', 85)
    result = rule.validate(df)
    print(f"Average score above 85: {result.valid}")
    print(f"Message: {result.message}")
    
    # Use a higher threshold
    rule = AverageScoreRule('score', 90)
    result = rule.validate(df)
    print(f"Average score above 90: {result.valid}")
    print(f"Message: {result.message}")


if __name__ == "__main__":
    main()