"""
Validators Module

This module provides concrete implementations of data validators for the NexusML suite.
Each validator follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a specific type of validation.
"""

from typing import Any, Dict, List, Optional, Set, Type, Union, cast

import pandas as pd

from nexusml.core.validation.interfaces import (
    ColumnValidator,
    DataFrameValidator,
    DataValidator,
    RowValidator,
    ValidationLevel,
    ValidationReport,
    ValidationResult,
    ValidationRule,
)
from nexusml.core.validation.rules import (
    AllowedValuesRule,
    ColumnCountRule,
    ColumnExistenceRule,
    ColumnTypeRule,
    CrossColumnComparisonRule,
    NonNullRule,
    RegexPatternRule,
    RowCountRule,
    UniqueValuesRule,
    ValueRangeRule,
)


class BaseValidator(DataValidator):
    """
    Base implementation of the DataValidator interface.
    
    Provides common functionality for all validators.
    """
    
    def __init__(self, name: str = "BaseValidator"):
        """
        Initialize a base validator.
        
        Args:
            name: Name of the validator.
        """
        self.name = name
        self.rules: List[ValidationRule] = []
    
    def validate(self, data: Any) -> ValidationReport:
        """
        Validate the data against all rules.
        
        Args:
            data: Data to validate.
            
        Returns:
            Validation report.
        """
        report = ValidationReport()
        
        for rule in self.rules:
            result = rule.validate(data)
            report.add_result(result)
        
        return report
    
    def add_rule(self, rule: ValidationRule) -> None:
        """
        Add a validation rule to this validator.
        
        Args:
            rule: Validation rule to add.
        """
        self.rules.append(rule)
    
    def get_rules(self) -> List[ValidationRule]:
        """
        Get all validation rules in this validator.
        
        Returns:
            List of validation rules.
        """
        return self.rules


class ConfigDrivenValidator(BaseValidator):
    """
    Validator that uses configuration to define validation rules.
    
    This validator creates validation rules based on a configuration dictionary.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: str = "ConfigDrivenValidator",
    ):
        """
        Initialize a configuration-driven validator.
        
        Args:
            config: Configuration dictionary. If None, uses an empty dictionary.
            name: Name of the validator.
        """
        super().__init__(name)
        self.config = config or {}
        self._create_rules_from_config()
    
    def _create_rules_from_config(self) -> None:
        """Create validation rules from the configuration."""
        # Create rules for required columns
        required_columns = self.config.get("required_columns", [])
        for column_info in required_columns:
            column_name = column_info.get("name")
            if column_name:
                # Add column existence rule
                self.add_rule(
                    ColumnExistenceRule(
                        column=column_name,
                        level=ValidationLevel.ERROR,
                        required=True,
                    )
                )
                
                # Add column type rule if specified
                data_type = column_info.get("data_type")
                if data_type:
                    self.add_rule(
                        ColumnTypeRule(
                            column=column_name,
                            expected_type=data_type,
                            level=ValidationLevel.ERROR,
                        )
                    )
                
                # Add non-null rule if specified
                if column_info.get("required", False):
                    max_null_fraction = column_info.get("max_null_fraction", 0.0)
                    self.add_rule(
                        NonNullRule(
                            column=column_name,
                            level=ValidationLevel.ERROR,
                            max_null_fraction=max_null_fraction,
                        )
                    )
                
                # Add value range rule if specified
                min_value = column_info.get("min_value")
                max_value = column_info.get("max_value")
                if min_value is not None or max_value is not None:
                    self.add_rule(
                        ValueRangeRule(
                            column=column_name,
                            min_value=min_value,
                            max_value=max_value,
                            level=ValidationLevel.ERROR,
                        )
                    )
                
                # Add allowed values rule if specified
                allowed_values = column_info.get("allowed_values")
                if allowed_values:
                    self.add_rule(
                        AllowedValuesRule(
                            column=column_name,
                            allowed_values=allowed_values,
                            level=ValidationLevel.ERROR,
                            max_invalid_fraction=column_info.get("max_invalid_fraction", 0.0),
                        )
                    )
                
                # Add regex pattern rule if specified
                pattern = column_info.get("pattern")
                if pattern:
                    self.add_rule(
                        RegexPatternRule(
                            column=column_name,
                            pattern=pattern,
                            level=ValidationLevel.ERROR,
                            max_invalid_fraction=column_info.get("max_invalid_fraction", 0.0),
                        )
                    )
                
                # Add unique values rule if specified
                if column_info.get("unique", False):
                    self.add_rule(
                        UniqueValuesRule(
                            column=column_name,
                            level=ValidationLevel.ERROR,
                            max_duplicate_fraction=column_info.get("max_duplicate_fraction", 0.0),
                        )
                    )
        
        # Create rules for cross-column comparisons
        comparisons = self.config.get("comparisons", [])
        for comparison_info in comparisons:
            column1 = comparison_info.get("column1")
            column2 = comparison_info.get("column2")
            operator = comparison_info.get("operator")
            if column1 and column2 and operator:
                self.add_rule(
                    CrossColumnComparisonRule(
                        column1=column1,
                        column2=column2,
                        comparison=operator,
                        level=ValidationLevel.ERROR,
                        max_invalid_fraction=comparison_info.get("max_invalid_fraction", 0.0),
                    )
                )
        
        # Create rules for row count
        row_count = self.config.get("row_count", {})
        min_rows = row_count.get("min")
        max_rows = row_count.get("max")
        if min_rows is not None or max_rows is not None:
            self.add_rule(
                RowCountRule(
                    min_rows=min_rows,
                    max_rows=max_rows,
                    level=ValidationLevel.ERROR,
                )
            )
        
        # Create rules for column count
        column_count = self.config.get("column_count", {})
        min_columns = column_count.get("min")
        max_columns = column_count.get("max")
        if min_columns is not None or max_columns is not None:
            self.add_rule(
                ColumnCountRule(
                    min_columns=min_columns,
                    max_columns=max_columns,
                    level=ValidationLevel.ERROR,
                )
            )


class BasicColumnValidator(BaseValidator, ColumnValidator):
    """
    Validator for a single column in a DataFrame.
    
    This validator applies rules to a specific column.
    """
    
    def __init__(
        self,
        column: str,
        rules: Optional[List[ValidationRule]] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a column validator.
        
        Args:
            column: Column name to validate.
            rules: Initial list of validation rules.
            name: Name of the validator. If None, uses the column name.
        """
        super().__init__(name or f"ColumnValidator({column})")
        self.column = column
        
        # Add initial rules
        if rules:
            for rule in rules:
                self.add_rule(rule)
    
    def validate_column(self, df: pd.DataFrame, column: Optional[str] = None) -> ValidationReport:
        """
        Validate a specific column in a DataFrame.
        
        Args:
            df: DataFrame to validate.
            column: Column name to validate. If None, uses the column specified in the constructor.
            
        Returns:
            Validation report.
        """
        column_to_validate = column or self.column
        
        # Check if the column exists
        if column_to_validate not in df.columns:
            report = ValidationReport()
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Column '{column_to_validate}' does not exist",
                    context={"column": column_to_validate}
                )
            )
            return report
        
        # Extract the column as a Series
        column_data = df[column_to_validate]
        
        # Create a single-column DataFrame for validation
        column_df = pd.DataFrame({column_to_validate: column_data})
        
        # Validate using the base validator
        return super().validate(column_df)


class BasicRowValidator(BaseValidator, RowValidator):
    """
    Validator for rows in a DataFrame.
    
    This validator applies rules to individual rows.
    """
    
    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        name: str = "RowValidator",
    ):
        """
        Initialize a row validator.
        
        Args:
            rules: Initial list of validation rules.
            name: Name of the validator.
        """
        super().__init__(name)
        
        # Add initial rules
        if rules:
            for rule in rules:
                self.add_rule(rule)
    
    def validate_row(self, row: pd.Series) -> ValidationReport:
        """
        Validate a single row in a DataFrame.
        
        Args:
            row: Row to validate.
            
        Returns:
            Validation report.
        """
        # Convert the row to a single-row DataFrame
        row_df = pd.DataFrame([row])
        
        # Validate using the base validator
        return super().validate(row_df)


class BasicDataFrameValidator(BaseValidator, DataFrameValidator):
    """
    Validator for an entire DataFrame.
    
    This validator applies rules to the entire DataFrame.
    """
    
    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        name: str = "DataFrameValidator",
    ):
        """
        Initialize a DataFrame validator.
        
        Args:
            rules: Initial list of validation rules.
            name: Name of the validator.
        """
        super().__init__(name)
        
        # Add initial rules
        if rules:
            for rule in rules:
                self.add_rule(rule)
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate an entire DataFrame.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Validation report.
        """
        # Validate using the base validator
        return super().validate(df)


class CompositeValidator(BaseValidator):
    """
    Validator that combines multiple validators.
    
    This validator applies multiple validators to the same data.
    """
    
    def __init__(
        self,
        validators: Optional[List[DataValidator]] = None,
        name: str = "CompositeValidator",
    ):
        """
        Initialize a composite validator.
        
        Args:
            validators: List of validators to apply.
            name: Name of the validator.
        """
        super().__init__(name)
        self.validators = validators or []
    
    def validate(self, data: Any) -> ValidationReport:
        """
        Validate the data using all validators.
        
        Args:
            data: Data to validate.
            
        Returns:
            Validation report.
        """
        report = ValidationReport()
        
        # Apply each validator
        for validator in self.validators:
            validator_report = validator.validate(data)
            
            # Add results from this validator to the composite report
            for result in validator_report.results:
                report.add_result(result)
        
        # Also apply rules directly added to this validator
        for rule in self.rules:
            result = rule.validate(data)
            report.add_result(result)
        
        return report
    
    def add_validator(self, validator: DataValidator) -> None:
        """
        Add a validator to this composite validator.
        
        Args:
            validator: Validator to add.
        """
        self.validators.append(validator)
    
    def get_validators(self) -> List[DataValidator]:
        """
        Get all validators in this composite validator.
        
        Returns:
            List of validators.
        """
        return self.validators


class SchemaValidator(BaseValidator, DataFrameValidator):
    """
    Validator that validates a DataFrame against a schema.
    
    This validator checks that a DataFrame has the expected columns with the expected types.
    """
    
    def __init__(
        self,
        schema: Dict[str, Union[str, type]],
        required_columns: Optional[Set[str]] = None,
        name: str = "SchemaValidator",
    ):
        """
        Initialize a schema validator.
        
        Args:
            schema: Dictionary mapping column names to expected types.
            required_columns: Set of column names that are required. If None, all columns in the schema are required.
            name: Name of the validator.
        """
        super().__init__(name)
        self.schema = schema
        self.required_columns = required_columns or set(schema.keys())
        
        # Create rules for each column in the schema
        for column, expected_type in schema.items():
            # Add column existence rule if the column is required
            if column in self.required_columns:
                self.add_rule(
                    ColumnExistenceRule(
                        column=column,
                        level=ValidationLevel.ERROR,
                        required=True,
                    )
                )
            else:
                self.add_rule(
                    ColumnExistenceRule(
                        column=column,
                        level=ValidationLevel.WARNING,
                        required=False,
                    )
                )
            
            # Add column type rule
            self.add_rule(
                ColumnTypeRule(
                    column=column,
                    expected_type=expected_type,
                    level=ValidationLevel.ERROR,
                )
            )
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationReport:
        """
        Validate a DataFrame against the schema.
        
        Args:
            df: DataFrame to validate.
            
        Returns:
            Validation report.
        """
        # Validate using the base validator
        return super().validate(df)