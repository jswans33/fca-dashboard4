"""
Type definitions for the validation module.

This module provides type hints for the validation module to improve type safety.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Protocol, Set, Type, TypeVar, Union

import pandas as pd

# Type variable for generic types
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series
ValidationFunction = Callable[[DataFrame], Dict[str, Any]]

# Protocol classes for structural typing
class HasValidate(Protocol):
    """Protocol for objects that have a validate method."""
    
    def validate(self, data: Any) -> 'ValidationReport':
        """Validate data."""
        ...

class HasAddRule(Protocol):
    """Protocol for objects that have an add_rule method."""
    
    def add_rule(self, rule: 'ValidationRule') -> None:
        """Add a validation rule."""
        ...

class HasGetRules(Protocol):
    """Protocol for objects that have a get_rules method."""
    
    def get_rules(self) -> List['ValidationRule']:
        """Get all validation rules."""
        ...

class ValidationLevel(Enum):
    """Enumeration of validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class ValidationResult:
    """
    Class representing the result of a validation check.
    
    Attributes:
        valid (bool): Whether the validation passed.
        level (ValidationLevel): Severity level of the validation.
        message (str): Description of the validation result.
        context (Dict[str, Any]): Additional context about the validation.
    """
    
    valid: bool
    level: ValidationLevel
    message: str
    context: Dict[str, Any]
    
    def __init__(
        self,
        valid: bool,
        level: ValidationLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None: ...
    
    def __str__(self) -> str: ...
    
    def to_dict(self) -> Dict[str, Any]: ...

class ValidationReport:
    """
    Class representing a collection of validation results.
    
    Attributes:
        results (List[ValidationResult]): List of validation results.
    """
    
    results: List[ValidationResult]
    
    def __init__(self, results: Optional[List[ValidationResult]] = None) -> None: ...
    
    def add_result(self, result: ValidationResult) -> None: ...
    
    def is_valid(self, include_warnings: bool = False, include_info: bool = False) -> bool: ...
    
    def get_errors(self) -> List[ValidationResult]: ...
    
    def get_warnings(self) -> List[ValidationResult]: ...
    
    def get_info(self) -> List[ValidationResult]: ...
    
    def to_dict(self) -> Dict[str, Any]: ...
    
    def __str__(self) -> str: ...

class ValidationRule(Protocol):
    """
    Interface for validation rules.
    
    A validation rule is a single check that can be applied to data.
    """
    
    def validate(self, data: Any) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class DataValidator(Protocol):
    """
    Interface for data validators.
    
    A data validator applies multiple validation rules to data.
    """
    
    def validate(self, data: Any) -> ValidationReport: ...
    
    def add_rule(self, rule: ValidationRule) -> None: ...
    
    def get_rules(self) -> List[ValidationRule]: ...

class ColumnValidator(DataValidator, Protocol):
    """
    Interface for column validators.
    
    A column validator applies validation rules to a specific column in a DataFrame.
    """
    
    def validate_column(self, df: DataFrame, column: str) -> ValidationReport: ...

class RowValidator(DataValidator, Protocol):
    """
    Interface for row validators.
    
    A row validator applies validation rules to rows in a DataFrame.
    """
    
    def validate_row(self, row: Series) -> ValidationReport: ...

class DataFrameValidator(DataValidator, Protocol):
    """
    Interface for DataFrame validators.
    
    A DataFrame validator applies validation rules to an entire DataFrame.
    """
    
    def validate_dataframe(self, df: DataFrame) -> ValidationReport: ...

# Concrete rule types
class ColumnExistenceRule:
    """Rule that checks if a column exists in a DataFrame."""
    
    column: str
    level: ValidationLevel
    required: bool
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        required: bool = True,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class ColumnTypeRule:
    """Rule that checks if a column has the expected data type."""
    
    column: str
    expected_type: Union[str, type]
    level: ValidationLevel
    
    def __init__(
        self,
        column: str,
        expected_type: Union[str, type],
        level: ValidationLevel = ValidationLevel.ERROR,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class NonNullRule:
    """Rule that checks if a column has no null values."""
    
    column: str
    level: ValidationLevel
    max_null_fraction: float
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_null_fraction: float = 0.0,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class ValueRangeRule:
    """Rule that checks if numeric values in a column are within a specified range."""
    
    column: str
    min_value: Optional[float]
    max_value: Optional[float]
    level: ValidationLevel
    
    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class UniqueValuesRule:
    """Rule that checks if a column has unique values."""
    
    column: str
    level: ValidationLevel
    max_duplicate_fraction: float
    
    def __init__(
        self,
        column: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_duplicate_fraction: float = 0.0,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class AllowedValuesRule:
    """Rule that checks if values in a column are from a set of allowed values."""
    
    column: str
    allowed_values: Set[Any]
    level: ValidationLevel
    max_invalid_fraction: float
    
    def __init__(
        self,
        column: str,
        allowed_values: Union[List[Any], Set[Any]],
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class RegexPatternRule:
    """Rule that checks if string values in a column match a regular expression pattern."""
    
    column: str
    pattern: Pattern
    level: ValidationLevel
    max_invalid_fraction: float
    
    def __init__(
        self,
        column: str,
        pattern: Union[str, Pattern],
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class CrossColumnComparisonRule:
    """Rule that compares values between two columns."""
    
    column1: str
    column2: str
    comparison: str
    level: ValidationLevel
    max_invalid_fraction: float
    
    def __init__(
        self,
        column1: str,
        column2: str,
        comparison: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        max_invalid_fraction: float = 0.0,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class RowCountRule:
    """Rule that checks if the DataFrame has a certain number of rows."""
    
    min_rows: Optional[int]
    max_rows: Optional[int]
    level: ValidationLevel
    
    def __init__(
        self,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

class ColumnCountRule:
    """Rule that checks if the DataFrame has a certain number of columns."""
    
    min_columns: Optional[int]
    max_columns: Optional[int]
    level: ValidationLevel
    
    def __init__(
        self,
        min_columns: Optional[int] = None,
        max_columns: Optional[int] = None,
        level: ValidationLevel = ValidationLevel.ERROR,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationResult: ...
    
    def get_name(self) -> str: ...
    
    def get_description(self) -> str: ...

# Concrete validator types
class BaseValidator:
    """Base implementation of the DataValidator interface."""
    
    name: str
    rules: List[ValidationRule]
    
    def __init__(self, name: str = "BaseValidator") -> None: ...
    
    def validate(self, data: Any) -> ValidationReport: ...
    
    def add_rule(self, rule: ValidationRule) -> None: ...
    
    def get_rules(self) -> List[ValidationRule]: ...

class ConfigDrivenValidator(BaseValidator):
    """Validator that uses configuration to define validation rules."""
    
    config: Dict[str, Any]
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: str = "ConfigDrivenValidator",
    ) -> None: ...
    
    def _create_rules_from_config(self) -> None: ...

class BasicColumnValidator(BaseValidator, ColumnValidator):
    """Validator for a single column in a DataFrame."""
    
    column: str
    
    def __init__(
        self,
        column: str,
        rules: Optional[List[ValidationRule]] = None,
        name: Optional[str] = None,
    ) -> None: ...
    
    def validate_column(self, df: DataFrame, column: Optional[str] = None) -> ValidationReport: ...

class BasicRowValidator(BaseValidator, RowValidator):
    """Validator for rows in a DataFrame."""
    
    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        name: str = "RowValidator",
    ) -> None: ...
    
    def validate_row(self, row: Series) -> ValidationReport: ...

class BasicDataFrameValidator(BaseValidator, DataFrameValidator):
    """Validator for an entire DataFrame."""
    
    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        name: str = "DataFrameValidator",
    ) -> None: ...
    
    def validate_dataframe(self, df: DataFrame) -> ValidationReport: ...

class CompositeValidator(BaseValidator):
    """Validator that combines multiple validators."""
    
    validators: List[DataValidator]
    
    def __init__(
        self,
        validators: Optional[List[DataValidator]] = None,
        name: str = "CompositeValidator",
    ) -> None: ...
    
    def validate(self, data: Any) -> ValidationReport: ...
    
    def add_validator(self, validator: DataValidator) -> None: ...
    
    def get_validators(self) -> List[DataValidator]: ...

class SchemaValidator(BaseValidator, DataFrameValidator):
    """Validator that validates a DataFrame against a schema."""
    
    schema: Dict[str, Union[str, type]]
    required_columns: Set[str]
    
    def __init__(
        self,
        schema: Dict[str, Union[str, type]],
        required_columns: Optional[Set[str]] = None,
        name: str = "SchemaValidator",
    ) -> None: ...
    
    def validate_dataframe(self, df: DataFrame) -> ValidationReport: ...

# Adapter types
class ReferenceDataValidator(DataValidator):
    """Adapter for reference data validation functions."""
    
    name: str
    rules: List[Any]
    
    def __init__(self, name: str = "ReferenceDataValidator") -> None: ...
    
    def validate(self, data: Any) -> ValidationReport: ...
    
    def add_rule(self, rule: Any) -> None: ...
    
    def get_rules(self) -> List[Any]: ...

class LegacyDataFrameValidator(DataValidator):
    """Adapter for legacy DataFrame validation functions."""
    
    validation_func: ValidationFunction
    name: str
    rules: List[Any]
    
    def __init__(
        self,
        validation_func: ValidationFunction,
        name: Optional[str] = None,
    ) -> None: ...
    
    def validate(self, data: DataFrame) -> ValidationReport: ...
    
    def add_rule(self, rule: Any) -> None: ...
    
    def get_rules(self) -> List[Any]: ...

# Convenience functions
def create_validator_from_config(
    config: Dict[str, Any],
    name: Optional[str] = None
) -> ConfigDrivenValidator: ...

def validate_dataframe(
    df: DataFrame,
    config: Optional[Dict[str, Any]] = None,
    validator: Optional[DataValidator] = None
) -> ValidationReport: ...

def validate_column(
    df: DataFrame,
    column: str,
    config: Optional[Dict[str, Any]] = None,
    validator: Optional[Union[DataValidator, ColumnValidator]] = None
) -> ValidationReport: ...