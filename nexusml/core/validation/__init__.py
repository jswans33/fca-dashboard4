"""
Validation Package

This package provides a comprehensive validation system for data in the NexusML suite.
It follows SOLID principles, particularly the Single Responsibility Principle (SRP)
and the Interface Segregation Principle (ISP).
"""

# Import interfaces
from nexusml.core.validation.interfaces import (
    ValidationLevel,
    ValidationResult,
    ValidationReport,
    ValidationRule,
    DataValidator,
    ColumnValidator,
    RowValidator,
    DataFrameValidator,
)

# Import rules
from nexusml.core.validation.rules import (
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
)

# Import validators
from nexusml.core.validation.validators import (
    BaseValidator,
    ConfigDrivenValidator,
    BasicColumnValidator,
    BasicRowValidator,
    BasicDataFrameValidator,
    CompositeValidator,
    SchemaValidator,
)

# Import adapters
from nexusml.core.validation.adapters import (
    ReferenceDataValidator,
    LegacyDataFrameValidator,
)

# Define __all__ to control what gets imported with "from nexusml.core.validation import *"
__all__ = [
    # Interfaces
    "ValidationLevel",
    "ValidationResult",
    "ValidationReport",
    "ValidationRule",
    "DataValidator",
    "ColumnValidator",
    "RowValidator",
    "DataFrameValidator",
    
    # Rules
    "ColumnExistenceRule",
    "ColumnTypeRule",
    "NonNullRule",
    "ValueRangeRule",
    "UniqueValuesRule",
    "AllowedValuesRule",
    "RegexPatternRule",
    "CrossColumnComparisonRule",
    "RowCountRule",
    "ColumnCountRule",
    
    # Validators
    "BaseValidator",
    "ConfigDrivenValidator",
    "BasicColumnValidator",
    "BasicRowValidator",
    "BasicDataFrameValidator",
    "CompositeValidator",
    "SchemaValidator",
    
    # Adapters
    "ReferenceDataValidator",
    "LegacyDataFrameValidator",
]

# Convenience function to create a validator from a configuration
def create_validator_from_config(config, name=None):
    """
    Create a validator from a configuration dictionary.
    
    Args:
        config: Configuration dictionary.
        name: Name of the validator. If None, uses "ConfigDrivenValidator".
        
    Returns:
        ConfigDrivenValidator instance.
    """
    return ConfigDrivenValidator(config, name=name or "ConfigDrivenValidator")

# Convenience function to validate a DataFrame
def validate_dataframe(df, config=None, validator=None):
    """
    Validate a DataFrame using a validator or configuration.
    
    Args:
        df: DataFrame to validate.
        config: Configuration dictionary. Used if validator is None.
        validator: Validator to use. If None, creates a ConfigDrivenValidator from config.
        
    Returns:
        ValidationReport instance.
    """
    if validator is None:
        if config is None:
            raise ValueError("Either validator or config must be provided")
        validator = create_validator_from_config(config)
    
    return validator.validate(df)

# Convenience function to validate a column
def validate_column(df, column, config=None, validator=None):
    """
    Validate a column in a DataFrame using a validator or configuration.
    
    Args:
        df: DataFrame containing the column.
        column: Column name to validate.
        config: Configuration dictionary. Used if validator is None.
        validator: Validator to use. If None, creates a BasicColumnValidator from config.
        
    Returns:
        ValidationReport instance.
    """
    if validator is None:
        if config is None:
            # Create a basic column validator with default rules
            validator = BasicColumnValidator(column)
        else:
            # Create a column validator from config
            rules = []
            if "type" in config:
                rules.append(ColumnTypeRule(column, config["type"]))
            if config.get("required", False):
                rules.append(NonNullRule(column, max_null_fraction=config.get("max_null_fraction", 0.0)))
            if "min_value" in config or "max_value" in config:
                rules.append(ValueRangeRule(column, config.get("min_value"), config.get("max_value")))
            if "allowed_values" in config:
                rules.append(AllowedValuesRule(column, config["allowed_values"]))
            if "pattern" in config:
                rules.append(RegexPatternRule(column, config["pattern"]))
            if config.get("unique", False):
                rules.append(UniqueValuesRule(column))
            
            validator = BasicColumnValidator(column, rules)
    
    # If the validator is a ColumnValidator, use validate_column
    if isinstance(validator, ColumnValidator):
        return validator.validate_column(df, column)
    
    # Otherwise, use the regular validate method
    return validator.validate(df)