"""
Validation Rules Module

This module provides concrete implementations of validation rules for data validation.
Each rule follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a single validation check.
"""

import re
from typing import Any, Dict, List, Optional, Pattern, Set, Union, cast

import numpy as np
import pandas as pd

from nexusml.core.validation.interfaces import (
    ValidationLevel,
    ValidationResult,
    ValidationRule,
)


class ColumnExistenceRule(ValidationRule):
    """
    Rule that checks if a column exists in a DataFrame.
    """
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        required: bool = True,
    ):
        """
        Initialize a column existence rule.
        
        Args:
            column: Column name to check.
            level: Validation level for this rule.
            required: Whether the column is required. If False, this rule will
                     always pass but will add an info message if the column is missing.
        """
        self.column = column
        self.level = level
        self.required = required
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the column exists in the DataFrame.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Expected DataFrame, got {type(data).__name__}",
                context={"column": self.column}
            )
        
        column_exists = self.column in data.columns
        
        if column_exists:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' exists",
                context={"column": self.column}
            )
        elif not self.required:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Optional column '{self.column}' does not exist",
                context={"column": self.column, "optional": True}
            )
        else:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Required column '{self.column}' does not exist",
                context={"column": self.column}
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"ColumnExistence({self.column})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        if self.required:
            return f"Checks if required column '{self.column}' exists"
        else:
            return f"Checks if optional column '{self.column}' exists"


class ColumnTypeRule(ValidationRule):
    """
    Rule that checks if a column has the expected data type.
    """
    
    def __init__(
        self,
        column: str,
        expected_type: Union[str, type],
        level: ValidationLevel = ValidationLevel.ERROR,
    ):
        """
        Initialize a column type rule.
        
        Args:
            column: Column name to check.
            expected_type: Expected data type. Can be a string ('int', 'float', 'str', etc.)
                          or a Python type (int, float, str, etc.).
            level: Validation level for this rule.
        """
        self.column = column
        self.expected_type = expected_type
        self.level = level
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the column has the expected data type.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Get the actual type of the column
        actual_type = data[self.column].dtype
        
        # Convert expected_type to string for comparison
        expected_type_str = (
            self.expected_type if isinstance(self.expected_type, str)
            else self.expected_type.__name__
        )
        
        # Check if the types match
        type_matches = False
        
        # Handle special cases for pandas/numpy types
        if expected_type_str in ('int', 'integer'):
            type_matches = pd.api.types.is_integer_dtype(actual_type)
        elif expected_type_str in ('float', 'number', 'numeric'):
            type_matches = pd.api.types.is_numeric_dtype(actual_type)
        elif expected_type_str in ('str', 'string', 'text'):
            type_matches = pd.api.types.is_string_dtype(actual_type)
        elif expected_type_str in ('bool', 'boolean'):
            type_matches = pd.api.types.is_bool_dtype(actual_type)
        elif expected_type_str in ('datetime', 'date'):
            type_matches = pd.api.types.is_datetime64_dtype(actual_type)
        else:
            # For other types, compare the type names
            type_matches = str(actual_type) == expected_type_str
        
        if type_matches:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' has expected type '{expected_type_str}'",
                context={"column": self.column, "expected_type": expected_type_str, "actual_type": str(actual_type)}
            )
        else:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Column '{self.column}' has type '{actual_type}', expected '{expected_type_str}'",
                context={"column": self.column, "expected_type": expected_type_str, "actual_type": str(actual_type)}
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        expected_type_str = (
            self.expected_type if isinstance(self.expected_type, str)
            else self.expected_type.__name__
        )
        return f"ColumnType({self.column}, {expected_type_str})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        expected_type_str = (
            self.expected_type if isinstance(self.expected_type, str)
            else self.expected_type.__name__
        )
        return f"Checks if column '{self.column}' has type '{expected_type_str}'"


class NonNullRule(ValidationRule):
    """
    Rule that checks if a column has no null values.
    """
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_null_fraction: float = 0.0,
    ):
        """
        Initialize a non-null rule.
        
        Args:
            column: Column name to check.
            level: Validation level for this rule.
            max_null_fraction: Maximum allowed fraction of null values (0.0 to 1.0).
                              If the fraction of null values is less than or equal to this value,
                              the validation will pass.
        """
        self.column = column
        self.level = level
        self.max_null_fraction = max_null_fraction
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the column has no null values.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Count null values
        null_count = data[self.column].isna().sum()
        total_count = len(data)
        null_fraction = null_count / total_count if total_count > 0 else 0.0
        
        if null_fraction <= self.max_null_fraction:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' has {null_count} null values ({null_fraction:.2%}), within limit of {self.max_null_fraction:.2%}",
                context={
                    "column": self.column,
                    "null_count": int(null_count),
                    "total_count": int(total_count),
                    "null_fraction": float(null_fraction),
                    "max_null_fraction": float(self.max_null_fraction)
                }
            )
        else:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Column '{self.column}' has {null_count} null values ({null_fraction:.2%}), exceeding limit of {self.max_null_fraction:.2%}",
                context={
                    "column": self.column,
                    "null_count": int(null_count),
                    "total_count": int(total_count),
                    "null_fraction": float(null_fraction),
                    "max_null_fraction": float(self.max_null_fraction)
                }
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"NonNull({self.column}, {self.max_null_fraction:.2f})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        if self.max_null_fraction == 0.0:
            return f"Checks if column '{self.column}' has no null values"
        else:
            return f"Checks if column '{self.column}' has at most {self.max_null_fraction:.2%} null values"


class ValueRangeRule(ValidationRule):
    """
    Rule that checks if numeric values in a column are within a specified range.
    """
    
    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ):
        """
        Initialize a value range rule.
        
        Args:
            column: Column name to check.
            min_value: Minimum allowed value (inclusive). If None, no minimum is enforced.
            max_value: Maximum allowed value (inclusive). If None, no maximum is enforced.
            level: Validation level for this rule.
        """
        self.column = column
        self.min_value = min_value
        self.max_value = max_value
        self.level = level
        
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified")
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that values in the column are within the specified range.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Convert column to numeric, coercing errors to NaN
        numeric_values = pd.to_numeric(data[self.column], errors='coerce')
        
        # Count out-of-range values
        out_of_range_mask = pd.Series(False, index=numeric_values.index)
        
        if self.min_value is not None:
            out_of_range_mask |= numeric_values < self.min_value
        
        if self.max_value is not None:
            out_of_range_mask |= numeric_values > self.max_value
        
        # Exclude NaN values from the check
        out_of_range_mask &= ~numeric_values.isna()
        
        out_of_range_count = out_of_range_mask.sum()
        total_count = len(numeric_values) - numeric_values.isna().sum()
        
        if out_of_range_count == 0:
            range_str = self._get_range_str()
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"All values in column '{self.column}' are within range {range_str}",
                context={
                    "column": self.column,
                    "min_value": self.min_value,
                    "max_value": self.max_value,
                    "out_of_range_count": int(out_of_range_count),
                    "total_count": int(total_count)
                }
            )
        else:
            range_str = self._get_range_str()
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"{out_of_range_count} values in column '{self.column}' are outside range {range_str}",
                context={
                    "column": self.column,
                    "min_value": self.min_value,
                    "max_value": self.max_value,
                    "out_of_range_count": int(out_of_range_count),
                    "total_count": int(total_count)
                }
            )
    
    def _get_range_str(self) -> str:
        """Get a string representation of the range."""
        if self.min_value is not None and self.max_value is not None:
            return f"[{self.min_value}, {self.max_value}]"
        elif self.min_value is not None:
            return f"[{self.min_value}, inf)"
        else:
            return f"(-inf, {self.max_value}]"
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"ValueRange({self.column}, {self._get_range_str()})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        range_str = self._get_range_str()
        return f"Checks if values in column '{self.column}' are within range {range_str}"


class UniqueValuesRule(ValidationRule):
    """
    Rule that checks if a column has unique values.
    """
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_duplicate_fraction: float = 0.0,
    ):
        """
        Initialize a unique values rule.
        
        Args:
            column: Column name to check.
            level: Validation level for this rule.
            max_duplicate_fraction: Maximum allowed fraction of duplicate values (0.0 to 1.0).
                                   If the fraction of duplicate values is less than or equal to this value,
                                   the validation will pass.
        """
        self.column = column
        self.level = level
        self.max_duplicate_fraction = max_duplicate_fraction
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the column has unique values.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Count duplicate values
        duplicate_count = data[self.column].duplicated().sum()
        total_count = len(data)
        duplicate_fraction = duplicate_count / total_count if total_count > 0 else 0.0
        
        if duplicate_fraction <= self.max_duplicate_fraction:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' has {duplicate_count} duplicate values ({duplicate_fraction:.2%}), within limit of {self.max_duplicate_fraction:.2%}",
                context={
                    "column": self.column,
                    "duplicate_count": int(duplicate_count),
                    "total_count": int(total_count),
                    "duplicate_fraction": float(duplicate_fraction),
                    "max_duplicate_fraction": float(self.max_duplicate_fraction)
                }
            )
        else:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Column '{self.column}' has {duplicate_count} duplicate values ({duplicate_fraction:.2%}), exceeding limit of {self.max_duplicate_fraction:.2%}",
                context={
                    "column": self.column,
                    "duplicate_count": int(duplicate_count),
                    "total_count": int(total_count),
                    "duplicate_fraction": float(duplicate_fraction),
                    "max_duplicate_fraction": float(self.max_duplicate_fraction)
                }
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"UniqueValues({self.column}, {self.max_duplicate_fraction:.2f})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        if self.max_duplicate_fraction == 0.0:
            return f"Checks if column '{self.column}' has unique values"
        else:
            return f"Checks if column '{self.column}' has at most {self.max_duplicate_fraction:.2%} duplicate values"


class AllowedValuesRule(ValidationRule):
    """
    Rule that checks if values in a column are from a set of allowed values.
    """
    
    def __init__(
        self,
        column: str,
        allowed_values: Union[List[Any], Set[Any]],
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ):
        """
        Initialize an allowed values rule.
        
        Args:
            column: Column name to check.
            allowed_values: Set of allowed values.
            level: Validation level for this rule.
            max_invalid_fraction: Maximum allowed fraction of invalid values (0.0 to 1.0).
                                 If the fraction of invalid values is less than or equal to this value,
                                 the validation will pass.
        """
        self.column = column
        self.allowed_values = set(allowed_values)
        self.level = level
        self.max_invalid_fraction = max_invalid_fraction
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that values in the column are from the set of allowed values.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Count invalid values
        invalid_mask = ~data[self.column].isin(self.allowed_values)
        # Exclude NaN values from the check
        invalid_mask &= ~data[self.column].isna()
        
        invalid_count = invalid_mask.sum()
        total_count = len(data) - data[self.column].isna().sum()
        invalid_fraction = invalid_count / total_count if total_count > 0 else 0.0
        
        if invalid_fraction <= self.max_invalid_fraction:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' has {invalid_count} invalid values ({invalid_fraction:.2%}), within limit of {self.max_invalid_fraction:.2%}",
                context={
                    "column": self.column,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction),
                    "allowed_values": list(self.allowed_values)
                }
            )
        else:
            # Get a sample of invalid values
            invalid_values = data.loc[invalid_mask, self.column].unique()
            invalid_sample = list(invalid_values[:5])  # Limit to 5 examples
            
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Column '{self.column}' has {invalid_count} invalid values ({invalid_fraction:.2%}), exceeding limit of {self.max_invalid_fraction:.2%}. Examples: {invalid_sample}",
                context={
                    "column": self.column,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction),
                    "allowed_values": list(self.allowed_values),
                    "invalid_sample": invalid_sample
                }
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"AllowedValues({self.column}, {len(self.allowed_values)} values, {self.max_invalid_fraction:.2f})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        if self.max_invalid_fraction == 0.0:
            return f"Checks if all values in column '{self.column}' are from the set of allowed values"
        else:
            return f"Checks if at most {self.max_invalid_fraction:.2%} of values in column '{self.column}' are not from the set of allowed values"


class RegexPatternRule(ValidationRule):
    """
    Rule that checks if string values in a column match a regular expression pattern.
    """
    
    def __init__(
        self,
        column: str,
        pattern: Union[str, Pattern],
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ):
        """
        Initialize a regex pattern rule.
        
        Args:
            column: Column name to check.
            pattern: Regular expression pattern to match. Can be a string or a compiled pattern.
            level: Validation level for this rule.
            max_invalid_fraction: Maximum allowed fraction of invalid values (0.0 to 1.0).
                                 If the fraction of invalid values is less than or equal to this value,
                                 the validation will pass.
        """
        self.column = column
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.level = level
        self.max_invalid_fraction = max_invalid_fraction
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that string values in the column match the regular expression pattern.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
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
        
        # Convert column to string
        str_values = data[self.column].astype(str)
        
        # Count invalid values (those that don't match the pattern)
        invalid_mask = ~str_values.str.match(self.pattern.pattern)
        # Exclude NaN values from the check
        invalid_mask &= ~data[self.column].isna()
        
        invalid_count = invalid_mask.sum()
        total_count = len(data) - data[self.column].isna().sum()
        invalid_fraction = invalid_count / total_count if total_count > 0 else 0.0
        
        if invalid_fraction <= self.max_invalid_fraction:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"Column '{self.column}' has {invalid_count} values not matching pattern ({invalid_fraction:.2%}), within limit of {self.max_invalid_fraction:.2%}",
                context={
                    "column": self.column,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction),
                    "pattern": self.pattern.pattern
                }
            )
        else:
            # Get a sample of invalid values
            invalid_values = data.loc[invalid_mask, self.column].unique()
            invalid_sample = list(invalid_values[:5])  # Limit to 5 examples
            
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Column '{self.column}' has {invalid_count} values not matching pattern ({invalid_fraction:.2%}), exceeding limit of {self.max_invalid_fraction:.2%}. Examples: {invalid_sample}",
                context={
                    "column": self.column,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction),
                    "pattern": self.pattern.pattern,
                    "invalid_sample": invalid_sample
                }
            )
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"RegexPattern({self.column}, {self.pattern.pattern}, {self.max_invalid_fraction:.2f})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        if self.max_invalid_fraction == 0.0:
            return f"Checks if all string values in column '{self.column}' match pattern '{self.pattern.pattern}'"
        else:
            return f"Checks if at most {self.max_invalid_fraction:.2%} of string values in column '{self.column}' do not match pattern '{self.pattern.pattern}'"


class CrossColumnComparisonRule(ValidationRule):
    """
    Rule that compares values between two columns.
    """
    
    def __init__(
        self,
        column1: str,
        column2: str,
        comparison: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ):
        """
        Initialize a cross-column comparison rule.
        
        Args:
            column1: First column name.
            column2: Second column name.
            comparison: Comparison operator. One of: 'eq', 'ne', 'lt', 'le', 'gt', 'ge'.
            level: Validation level for this rule.
            max_invalid_fraction: Maximum allowed fraction of invalid comparisons (0.0 to 1.0).
                                 If the fraction of invalid comparisons is less than or equal to this value,
                                 the validation will pass.
        """
        self.column1 = column1
        self.column2 = column2
        self.comparison = comparison
        self.level = level
        self.max_invalid_fraction = max_invalid_fraction
        
        # Validate comparison operator
        valid_comparisons = {'eq', 'ne', 'lt', 'le', 'gt', 'ge'}
        if comparison not in valid_comparisons:
            raise ValueError(f"Invalid comparison operator: {comparison}. Must be one of: {valid_comparisons}")
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate the comparison between two columns.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Expected DataFrame, got {type(data).__name__}",
                context={"column1": self.column1, "column2": self.column2}
            )
        
        # Check if both columns exist
        missing_columns = []
        if self.column1 not in data.columns:
            missing_columns.append(self.column1)
        if self.column2 not in data.columns:
            missing_columns.append(self.column2)
        
        if missing_columns:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"Columns not found: {', '.join(missing_columns)}",
                context={"column1": self.column1, "column2": self.column2, "missing_columns": missing_columns}
            )
        
        # Perform the comparison
        if self.comparison == 'eq':
            invalid_mask = data[self.column1] != data[self.column2]
        elif self.comparison == 'ne':
            invalid_mask = data[self.column1] == data[self.column2]
        elif self.comparison == 'lt':
            invalid_mask = data[self.column1] >= data[self.column2]
        elif self.comparison == 'le':
            invalid_mask = data[self.column1] > data[self.column2]
        elif self.comparison == 'gt':
            invalid_mask = data[self.column1] <= data[self.column2]
        elif self.comparison == 'ge':
            invalid_mask = data[self.column1] < data[self.column2]
        else:
            # This should never happen due to the check in __init__
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Invalid comparison operator: {self.comparison}",
                context={"column1": self.column1, "column2": self.column2, "comparison": self.comparison}
            )
        
        # Exclude rows where either column has NaN
        invalid_mask &= ~(data[self.column1].isna() | data[self.column2].isna())
        
        invalid_count = invalid_mask.sum()
        total_count = len(data) - (data[self.column1].isna() | data[self.column2].isna()).sum()
        invalid_fraction = invalid_count / total_count if total_count > 0 else 0.0
        
        comparison_str = self._get_comparison_str()
        
        if invalid_fraction <= self.max_invalid_fraction:
            return ValidationResult(
                valid=True,
                level=ValidationLevel.INFO,
                message=f"{invalid_count} rows fail comparison '{self.column1} {comparison_str} {self.column2}' ({invalid_fraction:.2%}), within limit of {self.max_invalid_fraction:.2%}",
                context={
                    "column1": self.column1,
                    "column2": self.column2,
                    "comparison": self.comparison,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction)
                }
            )
        else:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"{invalid_count} rows fail comparison '{self.column1} {comparison_str} {self.column2}' ({invalid_fraction:.2%}), exceeding limit of {self.max_invalid_fraction:.2%}",
                context={
                    "column1": self.column1,
                    "column2": self.column2,
                    "comparison": self.comparison,
                    "invalid_count": int(invalid_count),
                    "total_count": int(total_count),
                    "invalid_fraction": float(invalid_fraction),
                    "max_invalid_fraction": float(self.max_invalid_fraction)
                }
            )
    
    def _get_comparison_str(self) -> str:
        """Get a string representation of the comparison operator."""
        comparison_map = {
            'eq': '==',
            'ne': '!=',
            'lt': '<',
            'le': '<=',
            'gt': '>',
            'ge': '>='
        }
        return comparison_map.get(self.comparison, self.comparison)
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"CrossColumnComparison({self.column1}, {self._get_comparison_str()}, {self.column2}, {self.max_invalid_fraction:.2f})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        comparison_str = self._get_comparison_str()
        if self.max_invalid_fraction == 0.0:
            return f"Checks if '{self.column1} {comparison_str} {self.column2}' for all rows"
        else:
            return f"Checks if '{self.column1} {comparison_str} {self.column2}' for at least {1 - self.max_invalid_fraction:.2%} of rows"


class RowCountRule(ValidationRule):
    """
    Rule that checks if the DataFrame has a certain number of rows.
    """
    
    def __init__(
        self,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ):
        """
        Initialize a row count rule.
        
        Args:
            min_rows: Minimum number of rows (inclusive). If None, no minimum is enforced.
            max_rows: Maximum number of rows (inclusive). If None, no maximum is enforced.
            level: Validation level for this rule.
        """
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.level = level
        
        if min_rows is None and max_rows is None:
            raise ValueError("At least one of min_rows or max_rows must be specified")
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the DataFrame has the expected number of rows.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Expected DataFrame, got {type(data).__name__}",
                context={}
            )
        
        row_count = len(data)
        
        # Check if row count is within range
        if self.min_rows is not None and row_count < self.min_rows:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"DataFrame has {row_count} rows, less than minimum of {self.min_rows}",
                context={
                    "row_count": int(row_count),
                    "min_rows": int(self.min_rows),
                    "max_rows": self.max_rows
                }
            )
        
        if self.max_rows is not None and row_count > self.max_rows:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"DataFrame has {row_count} rows, more than maximum of {self.max_rows}",
                context={
                    "row_count": int(row_count),
                    "min_rows": self.min_rows,
                    "max_rows": int(self.max_rows)
                }
            )
        
        # Row count is within range
        range_str = self._get_range_str()
        return ValidationResult(
            valid=True,
            level=ValidationLevel.INFO,
            message=f"DataFrame has {row_count} rows, within range {range_str}",
            context={
                "row_count": int(row_count),
                "min_rows": self.min_rows,
                "max_rows": self.max_rows
            }
        )
    
    def _get_range_str(self) -> str:
        """Get a string representation of the row count range."""
        if self.min_rows is not None and self.max_rows is not None:
            return f"[{self.min_rows}, {self.max_rows}]"
        elif self.min_rows is not None:
            return f"[{self.min_rows}, inf)"
        else:
            return f"(-inf, {self.max_rows}]"
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"RowCount({self._get_range_str()})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        range_str = self._get_range_str()
        return f"Checks if the DataFrame has a row count within range {range_str}"


class ColumnCountRule(ValidationRule):
    """
    Rule that checks if the DataFrame has a certain number of columns.
    """
    
    def __init__(
        self,
        min_columns: Optional[int] = None,
        max_columns: Optional[int] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ):
        """
        Initialize a column count rule.
        
        Args:
            min_columns: Minimum number of columns (inclusive). If None, no minimum is enforced.
            max_columns: Maximum number of columns (inclusive). If None, no maximum is enforced.
            level: Validation level for this rule.
        """
        self.min_columns = min_columns
        self.max_columns = max_columns
        self.level = level
        
        if min_columns is None and max_columns is None:
            raise ValueError("At least one of min_columns or max_columns must be specified")
    
    def validate(self, data: pd.DataFrame) -> ValidationResult:
        """
        Validate that the DataFrame has the expected number of columns.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation result.
        """
        if not isinstance(data, pd.DataFrame):
            return ValidationResult(
                valid=False,
                level=ValidationLevel.ERROR,
                message=f"Expected DataFrame, got {type(data).__name__}",
                context={}
            )
        
        column_count = len(data.columns)
        
        # Check if column count is within range
        if self.min_columns is not None and column_count < self.min_columns:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"DataFrame has {column_count} columns, less than minimum of {self.min_columns}",
                context={
                    "column_count": int(column_count),
                    "min_columns": int(self.min_columns),
                    "max_columns": self.max_columns
                }
            )
        
        if self.max_columns is not None and column_count > self.max_columns:
            return ValidationResult(
                valid=False,
                level=self.level,
                message=f"DataFrame has {column_count} columns, more than maximum of {self.max_columns}",
                context={
                    "column_count": int(column_count),
                    "min_columns": self.min_columns,
                    "max_columns": int(self.max_columns)
                }
            )
        
        # Column count is within range
        range_str = self._get_range_str()
        return ValidationResult(
            valid=True,
            level=ValidationLevel.INFO,
            message=f"DataFrame has {column_count} columns, within range {range_str}",
            context={
                "column_count": int(column_count),
                "min_columns": self.min_columns,
                "max_columns": self.max_columns
            }
        )
    
    def _get_range_str(self) -> str:
        """Get a string representation of the column count range."""
        if self.min_columns is not None and self.max_columns is not None:
            return f"[{self.min_columns}, {self.max_columns}]"
        elif self.min_columns is not None:
            return f"[{self.min_columns}, inf)"
        else:
            return f"(-inf, {self.max_columns}]"
    
    def get_name(self) -> str:
        """Get the name of this validation rule."""
        return f"ColumnCount({self._get_range_str()})"
    
    def get_description(self) -> str:
        """Get a description of this validation rule."""
        range_str = self._get_range_str()
        return f"Checks if the DataFrame has a column count within range {range_str}"