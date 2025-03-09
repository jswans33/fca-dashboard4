"""
Validation Interfaces Module

This module defines the interfaces for data validation components in the NexusML suite.
Each interface follows the Interface Segregation Principle (ISP) from SOLID,
defining a minimal set of methods that components must implement.
"""

import abc
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd


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
    
    def __init__(
        self,
        valid: bool,
        level: ValidationLevel,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a validation result.
        
        Args:
            valid: Whether the validation passed.
            level: Severity level of the validation.
            message: Description of the validation result.
            context: Additional context about the validation.
        """
        self.valid = valid
        self.level = level
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """Return a string representation of the validation result."""
        return f"[{self.level.value.upper()}] {'PASS' if self.valid else 'FAIL'}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "valid": self.valid,
            "level": self.level.value,
            "message": self.message,
            "context": self.context
        }


class ValidationReport:
    """
    Class representing a collection of validation results.
    
    Attributes:
        results (List[ValidationResult]): List of validation results.
    """
    
    def __init__(self, results: Optional[List[ValidationResult]] = None):
        """
        Initialize a validation report.
        
        Args:
            results: Initial list of validation results.
        """
        self.results = results or []
    
    def add_result(self, result: ValidationResult) -> None:
        """
        Add a validation result to the report.
        
        Args:
            result: Validation result to add.
        """
        self.results.append(result)
    
    def is_valid(self, include_warnings: bool = False, include_info: bool = False) -> bool:
        """
        Check if all validations passed.
        
        Args:
            include_warnings: Whether to consider warnings as validation failures.
            include_info: Whether to consider info messages as validation failures.
            
        Returns:
            True if all validations passed, False otherwise.
        """
        for result in self.results:
            if not result.valid:
                if result.level == ValidationLevel.ERROR:
                    return False
                if include_warnings and result.level == ValidationLevel.WARNING:
                    return False
                if include_info and result.level == ValidationLevel.INFO:
                    return False
        return True
    
    def get_errors(self) -> List[ValidationResult]:
        """
        Get all error-level validation results.
        
        Returns:
            List of error-level validation results.
        """
        return [r for r in self.results if r.level == ValidationLevel.ERROR and not r.valid]
    
    def get_warnings(self) -> List[ValidationResult]:
        """
        Get all warning-level validation results.
        
        Returns:
            List of warning-level validation results.
        """
        return [r for r in self.results if r.level == ValidationLevel.WARNING and not r.valid]
    
    def get_info(self) -> List[ValidationResult]:
        """
        Get all info-level validation results.
        
        Returns:
            List of info-level validation results.
        """
        return [r for r in self.results if r.level == ValidationLevel.INFO and not r.valid]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the validation report to a dictionary.
        
        Returns:
            Dictionary representation of the validation report.
        """
        return {
            "valid": self.is_valid(),
            "error_count": len(self.get_errors()),
            "warning_count": len(self.get_warnings()),
            "info_count": len(self.get_info()),
            "results": [r.to_dict() for r in self.results]
        }
    
    def __str__(self) -> str:
        """Return a string representation of the validation report."""
        lines = [
            f"Validation Report: {'PASS' if self.is_valid() else 'FAIL'}",
            f"Errors: {len(self.get_errors())}",
            f"Warnings: {len(self.get_warnings())}",
            f"Info: {len(self.get_info())}",
            "Results:"
        ]
        for result in self.results:
            lines.append(f"  {str(result)}")
        return "\n".join(lines)


class ValidationRule(abc.ABC):
    """
    Interface for validation rules.
    
    A validation rule is a single check that can be applied to data.
    """
    
    @abc.abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate the data against this rule.
        
        Args:
            data: Data to validate.
            
        Returns:
            Validation result.
        """
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this validation rule.
        
        Returns:
            Rule name.
        """
        pass
    
    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get a description of this validation rule.
        
        Returns:
            Rule description.
        """
        pass


class DataValidator(abc.ABC):
    """
    Interface for data validators.
    
    A data validator applies multiple validation rules to data.
    """
    
    @abc.abstractmethod
    def validate(self, data: Any) -> ValidationReport:
        """
        Validate the data against all rules.
        
        Args:
            data: Data to validate.
            
        Returns:
            Validation report.
        """
        pass
    
    @abc.abstractmethod
    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a validation rule to this validator.
        
        Args:
            rule: Validation rule to add.
        """
        pass
    
    @abc.abstractmethod
    def get_rules(self) -> List[ValidationRule]:
        """
        Get all validation rules in this validator.
        
        Returns:
            List of validation rules.
        """
        pass


class ColumnValidator(DataValidator):
    """
    Interface for column validators.
    
    A column validator applies validation rules to a specific column in a DataFrame.
    """
    
    @abc.abstractmethod
    def validate_column(self, df: pd.DataFrame, column: str) -> ValidationReport:
        """
        Validate a specific column in a DataFrame.
        
        Args:
            df: DataFrame to validate.
            column: Column name to validate.
            
        Returns:
            Validation report.
        """
        pass


class RowValidator(DataValidator):
    """
    Interface for row validators.
    
    A row validator applies validation rules to rows in a DataFrame.
    """
    
    @abc.abstractmethod
    def validate_row(self, row: pd.Series) -> ValidationReport:
        """
        Validate a single row in a DataFrame.
        
        Args:
            row: Row to validate.
            
        Returns:
            Validation report.
        """
        pass


class DataFrameValidator(DataValidator):
    """
    Interface for DataFrame validators.
    
    A DataFrame validator applies validation rules to an entire DataFrame.
    """
    
    @abc.abstractmethod
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate an entire DataFrame.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Validation report.
        """
        pass