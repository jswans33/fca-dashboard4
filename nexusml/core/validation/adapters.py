"""
Validation Adapters Module

This module provides adapters that convert existing validation functions to the new validation interfaces.
These adapters follow the Adapter Pattern to allow legacy code to work with the new validation system.
"""

from typing import Any, Dict, List, Optional, Union, cast

import pandas as pd

from nexusml.core.reference.validation import (
    validate_classification_data,
    validate_equipment_taxonomy_data,
    validate_glossary_data,
    validate_manufacturer_data,
    validate_service_life_data,
)
from nexusml.core.reference.base import ReferenceDataSource
from nexusml.core.reference.classification import ClassificationDataSource
from nexusml.core.reference.equipment import EquipmentTaxonomyDataSource
from nexusml.core.reference.glossary import GlossaryDataSource
from nexusml.core.reference.manufacturer import ManufacturerDataSource
from nexusml.core.reference.service_life import ServiceLifeDataSource

from nexusml.core.validation.interfaces import (
    DataValidator,
    ValidationLevel,
    ValidationReport,
    ValidationResult,
)


class ReferenceDataValidator(DataValidator):
    """
    Adapter for reference data validation functions.
    
    This adapter converts the existing reference data validation functions to the new validation interface.
    """
    
    def __init__(self, name: str = "ReferenceDataValidator"):
        """
        Initialize a reference data validator.
        
        Args:
            name: Name of the validator.
        """
        self.name = name
        self.rules = []
    
    def validate(self, data: ReferenceDataSource) -> ValidationReport:
        """
        Validate reference data using the appropriate validation function.
        
        Args:
            data: Reference data source to validate.
            
        Returns:
            Validation report.
        """
        report = ValidationReport()
        
        if not isinstance(data, ReferenceDataSource):
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Expected ReferenceDataSource, got {type(data).__name__}",
                    context={}
                )
            )
            return report
        
        # Call the appropriate validation function based on the type of reference data
        if isinstance(data, ClassificationDataSource):
            # Get the source type from the data source if available
            source_type = getattr(data, "source_type", "unknown")
            # Get the config from the data source if available
            config = getattr(data, "config", {})
            
            # Call the validation function
            result = validate_classification_data(data, source_type, config)
            
            # Convert the result to a ValidationResult
            self._add_legacy_result_to_report(report, result)
        
        elif isinstance(data, GlossaryDataSource):
            result = validate_glossary_data(data)
            self._add_legacy_result_to_report(report, result)
        
        elif isinstance(data, ManufacturerDataSource):
            result = validate_manufacturer_data(data)
            self._add_legacy_result_to_report(report, result)
        
        elif isinstance(data, ServiceLifeDataSource):
            result = validate_service_life_data(data)
            self._add_legacy_result_to_report(report, result)
        
        elif isinstance(data, EquipmentTaxonomyDataSource):
            result = validate_equipment_taxonomy_data(data)
            self._add_legacy_result_to_report(report, result)
        
        else:
            # Unknown reference data type
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Unsupported reference data type: {type(data).__name__}",
                    context={"data_type": type(data).__name__}
                )
            )
        
        return report
    
    def _add_legacy_result_to_report(self, report: ValidationReport, legacy_result: Dict[str, Any]) -> None:
        """
        Convert a legacy validation result to ValidationResults and add them to the report.
        
        Args:
            report: Validation report to add results to.
            legacy_result: Legacy validation result from the old validation functions.
        """
        # Check if the data was loaded
        loaded = legacy_result.get("loaded", False)
        if not loaded:
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message="Data not loaded",
                    context={"legacy_result": legacy_result}
                )
            )
            return
        
        # Add a result for each issue
        issues = legacy_result.get("issues", [])
        for issue in issues:
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=issue,
                    context={"legacy_result": legacy_result}
                )
            )
        
        # If there are no issues, add a success result
        if not issues:
            report.add_result(
                ValidationResult(
                    valid=True,
                    level=ValidationLevel.INFO,
                    message="Validation passed",
                    context={"legacy_result": legacy_result}
                )
            )
        
        # Add results for statistics
        stats = legacy_result.get("stats", {})
        for key, value in stats.items():
            report.add_result(
                ValidationResult(
                    valid=True,
                    level=ValidationLevel.INFO,
                    message=f"Statistic: {key} = {value}",
                    context={"key": key, "value": value, "legacy_result": legacy_result}
                )
            )
    
    def add_rule(self, rule: Any) -> None:
        """
        Add a validation rule to this validator.
        
        This method is not used for this adapter, as it uses the legacy validation functions.
        
        Args:
            rule: Validation rule to add.
        """
        # This adapter doesn't use rules, as it uses the legacy validation functions
        pass
    
    def get_rules(self) -> List[Any]:
        """
        Get all validation rules in this validator.
        
        This method is not used for this adapter, as it uses the legacy validation functions.
        
        Returns:
            Empty list, as this adapter doesn't use rules.
        """
        # This adapter doesn't use rules, as it uses the legacy validation functions
        return []


class LegacyDataFrameValidator(DataValidator):
    """
    Adapter for legacy DataFrame validation functions.
    
    This adapter converts legacy validation functions that take a DataFrame and return a dictionary
    to the new validation interface.
    """
    
    def __init__(
        self,
        validation_func: callable,
        name: Optional[str] = None,
    ):
        """
        Initialize a legacy DataFrame validator.
        
        Args:
            validation_func: Legacy validation function that takes a DataFrame and returns a dictionary.
            name: Name of the validator. If None, uses the function name.
        """
        self.validation_func = validation_func
        self.name = name or validation_func.__name__
        self.rules = []
    
    def validate(self, data: pd.DataFrame) -> ValidationReport:
        """
        Validate a DataFrame using the legacy validation function.
        
        Args:
            data: DataFrame to validate.
            
        Returns:
            Validation report.
        """
        report = ValidationReport()
        
        if not isinstance(data, pd.DataFrame):
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Expected DataFrame, got {type(data).__name__}",
                    context={}
                )
            )
            return report
        
        try:
            # Call the legacy validation function
            result = self.validation_func(data)
            
            # Convert the result to ValidationResults
            self._add_legacy_result_to_report(report, result)
        except Exception as e:
            # Handle exceptions from the legacy validation function
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Error in legacy validation function: {str(e)}",
                    context={"error": str(e)}
                )
            )
        
        return report
    
    def _add_legacy_result_to_report(self, report: ValidationReport, legacy_result: Dict[str, Any]) -> None:
        """
        Convert a legacy validation result to ValidationResults and add them to the report.
        
        Args:
            report: Validation report to add results to.
            legacy_result: Legacy validation result from the old validation function.
        """
        # Check if the validation passed
        valid = legacy_result.get("valid", False)
        
        # Add a result for each issue
        issues = legacy_result.get("issues", [])
        for issue in issues:
            report.add_result(
                ValidationResult(
                    valid=False,
                    level=ValidationLevel.ERROR,
                    message=issue,
                    context={"legacy_result": legacy_result}
                )
            )
        
        # If there are no issues, add a success result
        if not issues and valid:
            report.add_result(
                ValidationResult(
                    valid=True,
                    level=ValidationLevel.INFO,
                    message="Validation passed",
                    context={"legacy_result": legacy_result}
                )
            )
        
        # Add results for any other keys in the legacy result
        for key, value in legacy_result.items():
            if key not in ["valid", "issues"]:
                report.add_result(
                    ValidationResult(
                        valid=True,
                        level=ValidationLevel.INFO,
                        message=f"Additional info: {key} = {value}",
                        context={"key": key, "value": value, "legacy_result": legacy_result}
                    )
                )
    
    def add_rule(self, rule: Any) -> None:
        """
        Add a validation rule to this validator.
        
        This method is not used for this adapter, as it uses a legacy validation function.
        
        Args:
            rule: Validation rule to add.
        """
        # This adapter doesn't use rules, as it uses a legacy validation function
        pass
    
    def get_rules(self) -> List[Any]:
        """
        Get all validation rules in this validator.
        
        This method is not used for this adapter, as it uses a legacy validation function.
        
        Returns:
            Empty list, as this adapter doesn't use rules.
        """
        # This adapter doesn't use rules, as it uses a legacy validation function
        return []