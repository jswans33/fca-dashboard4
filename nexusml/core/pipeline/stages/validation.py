"""
Validation Stage Module

This module provides implementations of the ValidationStage interface for
validating data against requirements.
"""

from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd

from nexusml.config.manager import ConfigurationManager
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import BaseValidationStage
from nexusml.core.validation.interfaces import DataValidator, ColumnValidator, DataFrameValidator
from nexusml.core.validation.validators import (
    BasicColumnValidator,
    BasicDataFrameValidator,
    ConfigDrivenValidator,
)


class ConfigDrivenValidationStage(BaseValidationStage):
    """
    Implementation of ValidationStage that uses configuration for validation rules.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenValidation",
        description: str = "Validates data against configuration-defined rules",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
        validator: Optional[DataValidator] = None,
    ):
        """
        Initialize the configuration-driven validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading validation configuration.
            validator: Data validator to use. If None, creates a ConfigDrivenValidator.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        # Convert config_manager to a dictionary for ConfigDrivenValidator
        config_dict = {"config_name": self.config.get("config_name", "production_data_config")}
        self.validator = validator or ConfigDrivenValidator(config=config_dict)

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the input data using configuration-driven rules.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.
        """
        # Get the configuration name from kwargs or config
        config_name = kwargs.get(
            "config_name", self.config.get("config_name", "production_data_config")
        )

        # Validate the data
        validation_report = self.validator.validate(data)
        
        # Convert ValidationReport to dictionary
        return {
            "valid": validation_report.is_valid(),
            "issues": [str(result) for result in validation_report.results if not result.valid]
        }


class ColumnValidationStage(BaseValidationStage):
    """
    Implementation of ValidationStage that validates specific columns.
    """

    def __init__(
        self,
        name: str = "ColumnValidation",
        description: str = "Validates specific columns in the data",
        config: Optional[Dict[str, Any]] = None,
        required_columns: Optional[List[str]] = None,
        column_validator: Optional[ColumnValidator] = None,
    ):
        """
        Initialize the column validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            required_columns: List of required column names.
            column_validator: Column validator to use. If None, creates a new one.
        """
        super().__init__(name, description, config)
        self.required_columns = required_columns or []
        self.column_validator = column_validator or BasicColumnValidator(column="")

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate specific columns in the input data.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.
        """
        # Get required columns from kwargs, config, or instance variable
        required_columns = kwargs.get(
            "required_columns",
            self.config.get("required_columns", self.required_columns),
        )

        # Validate the columns
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            return {
                "valid": False,
                "issues": [f"Missing required columns: {', '.join(missing_columns)}"],
            }

        # Check for missing values in critical columns
        critical_columns = kwargs.get(
            "critical_columns",
            self.config.get(
                "critical_columns", ["equipment_tag", "category_name", "mcaa_system_category"]
            ),
        )

        missing_values = {}
        for col in critical_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    missing_values[col] = missing_count

        if missing_values:
            issues = [
                f"Missing values in {col}: {count}" for col, count in missing_values.items()
            ]
            return {"valid": False, "issues": issues}

        # All checks passed
        return {"valid": True, "issues": []}


class DataTypeValidationStage(BaseValidationStage):
    """
    Implementation of ValidationStage that validates data types.
    """

    def __init__(
        self,
        name: str = "DataTypeValidation",
        description: str = "Validates data types in the data",
        config: Optional[Dict[str, Any]] = None,
        column_types: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the data type validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            column_types: Dictionary mapping column names to expected data types.
        """
        super().__init__(name, description, config)
        self.column_types = column_types or {}

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate data types in the input data.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.
        """
        # Get column types from kwargs, config, or instance variable
        column_types = kwargs.get(
            "column_types", self.config.get("column_types", self.column_types)
        )

        # Validate data types
        type_issues = []
        for col, expected_type in column_types.items():
            if col not in data.columns:
                continue

            # Check if the column can be converted to the expected type
            try:
                if expected_type == "int":
                    pd.to_numeric(data[col], errors="raise", downcast="integer")
                elif expected_type == "float":
                    pd.to_numeric(data[col], errors="raise", downcast="float")
                elif expected_type == "str":
                    data[col].astype(str)
                elif expected_type == "bool":
                    data[col].astype(bool)
                elif expected_type == "datetime":
                    pd.to_datetime(data[col], errors="raise")
                else:
                    # Unknown type, skip validation
                    continue
            except Exception as e:
                type_issues.append(f"Column '{col}' cannot be converted to {expected_type}: {str(e)}")

        if type_issues:
            return {"valid": False, "issues": type_issues}

        # All checks passed
        return {"valid": True, "issues": []}


class CompositeValidationStage(BaseValidationStage):
    """
    Implementation of ValidationStage that combines multiple validators.
    """

    def __init__(
        self,
        name: str = "CompositeValidation",
        description: str = "Combines multiple validators",
        config: Optional[Dict[str, Any]] = None,
        validators: Optional[List[BaseValidationStage]] = None,
    ):
        """
        Initialize the composite validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            validators: List of validation stages to use.
        """
        super().__init__(name, description, config)
        self.validators = validators or []

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the input data using multiple validators.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.
        """
        all_issues = []
        valid = True

        # Run all validators
        for validator in self.validators:
            result = validator.validate_data(data, **kwargs)
            if not result.get("valid", True):
                valid = False
                all_issues.extend(result.get("issues", []))

        return {"valid": valid, "issues": all_issues}


class DataFrameValidationStage(BaseValidationStage):
    """
    Implementation of ValidationStage that validates the entire DataFrame.
    """

    def __init__(
        self,
        name: str = "DataFrameValidation",
        description: str = "Validates the entire DataFrame",
        config: Optional[Dict[str, Any]] = None,
        validator: Optional[DataFrameValidator] = None,
    ):
        """
        Initialize the DataFrame validation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            validator: DataFrame validator to use. If None, creates a new one.
        """
        super().__init__(name, description, config)
        self.validator = validator or BasicDataFrameValidator()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("data")

    def validate_data(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Validate the entire DataFrame.

        Args:
            data: Input DataFrame to validate.
            **kwargs: Additional arguments for validation.

        Returns:
            Dictionary with validation results.
        """
        # Validate the DataFrame
        min_rows = kwargs.get("min_rows", self.config.get("min_rows", 1))
        max_rows = kwargs.get("max_rows", self.config.get("max_rows", None))
        min_columns = kwargs.get("min_columns", self.config.get("min_columns", 1))
        max_columns = kwargs.get("max_columns", self.config.get("max_columns", None))

        issues = []

        # Check row count
        if len(data) < min_rows:
            issues.append(f"DataFrame has fewer than {min_rows} rows")
        if max_rows is not None and len(data) > max_rows:
            issues.append(f"DataFrame has more than {max_rows} rows")

        # Check column count
        if len(data.columns) < min_columns:
            issues.append(f"DataFrame has fewer than {min_columns} columns")
        if max_columns is not None and len(data.columns) > max_columns:
            issues.append(f"DataFrame has more than {max_columns} columns")

        # Check for duplicate rows
        if kwargs.get("check_duplicates", self.config.get("check_duplicates", True)):
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"DataFrame contains {duplicate_count} duplicate rows")

        if issues:
            return {"valid": False, "issues": issues}

        # All checks passed
        return {"valid": True, "issues": []}