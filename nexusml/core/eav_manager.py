"""
Equipment Attribute-Value (EAV) Manager

This module manages the EAV structure for equipment attributes, providing functionality to:
1. Load attribute templates for different equipment types
2. Validate equipment attributes against templates
3. Generate attribute templates for equipment based on ML predictions
4. Fill in missing attributes using ML predictions and rules
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.config import get_project_root


class EAVManager:
    """
    Manages the Entity-Attribute-Value (EAV) structure for equipment attributes.
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the EAV Manager with templates.

        Args:
            templates_path: Path to the JSON file containing attribute templates.
                           If None, uses the default path.
        """
        self.templates_path = templates_path
        self.templates = {}
        self.load_templates()

    def load_templates(self) -> None:
        """Load attribute templates from the JSON file."""
        templates_path = self.templates_path
        if templates_path is None:
            # Use default path
            root = get_project_root()
            templates_path = root / "config" / "eav" / "equipment_attributes.json"

        try:
            with open(templates_path, "r") as f:
                self.templates = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load attribute templates: {e}")
            self.templates = {}

    def get_equipment_template(self, equipment_type: str) -> Dict[str, Any]:
        """
        Get the attribute template for a specific equipment type.

        Args:
            equipment_type: The type of equipment (e.g., "Chiller", "Air Handler")

        Returns:
            Dict containing the attribute template, or an empty dict if not found
        """
        # Try exact match first
        if equipment_type in self.templates:
            return self.templates[equipment_type]

        # Try case-insensitive match
        for template_name, template in self.templates.items():
            if template_name.lower() == equipment_type.lower():
                return template

        # Try partial match (e.g., "Centrifugal Chiller" should match "Chiller")
        for template_name, template in self.templates.items():
            if (
                template_name.lower() in equipment_type.lower()
                or equipment_type.lower() in template_name.lower()
            ):
                return template

        # Return empty template if no match found
        return {}

    def get_required_attributes(self, equipment_type: str) -> List[str]:
        """
        Get required attributes for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of required attribute names
        """
        template = self.get_equipment_template(equipment_type)
        return template.get("required_attributes", [])

    def get_optional_attributes(self, equipment_type: str) -> List[str]:
        """
        Get optional attributes for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of optional attribute names
        """
        template = self.get_equipment_template(equipment_type)
        return template.get("optional_attributes", [])

    def get_all_attributes(self, equipment_type: str) -> List[str]:
        """
        Get all attributes (required and optional) for a given equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            List of all attribute names
        """
        template = self.get_equipment_template(equipment_type)
        required = template.get("required_attributes", [])
        optional = template.get("optional_attributes", [])
        return required + optional

    def get_attribute_unit(self, equipment_type: str, attribute: str) -> str:
        """
        Get the unit for a specific attribute of an equipment type.

        Args:
            equipment_type: The type of equipment
            attribute: The attribute name

        Returns:
            Unit string, or empty string if not found
        """
        template = self.get_equipment_template(equipment_type)
        units = template.get("units", {})
        return units.get(attribute, "")

    def get_classification_ids(self, equipment_type: str) -> Dict[str, str]:
        """
        Get the classification IDs (OmniClass, MasterFormat, Uniformat) for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with classification IDs
        """
        template = self.get_equipment_template(equipment_type)
        return {
            "omniclass_id": template.get("omniclass_id", ""),
            "masterformat_id": template.get("masterformat_id", ""),
            "uniformat_id": template.get("uniformat_id", ""),
        }

    def get_performance_fields(self, equipment_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get the performance fields for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with performance fields
        """
        template = self.get_equipment_template(equipment_type)
        return template.get("performance_fields", {})

    def validate_attributes(
        self, equipment_type: str, attributes: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate attributes against the template for an equipment type.

        Args:
            equipment_type: The type of equipment
            attributes: Dictionary of attribute name-value pairs

        Returns:
            Dictionary with validation results:
            {
                "missing_required": List of missing required attributes,
                "unknown": List of attributes not in the template
            }
        """
        template = self.get_equipment_template(equipment_type)
        required = set(template.get("required_attributes", []))
        optional = set(template.get("optional_attributes", []))
        all_valid = required.union(optional)

        # Check for missing required attributes
        provided = set(attributes.keys())
        missing_required = required - provided

        # Check for unknown attributes
        unknown = provided - all_valid

        return {"missing_required": list(missing_required), "unknown": list(unknown)}

    def generate_attribute_template(self, equipment_type: str) -> Dict[str, Any]:
        """
        Generate an attribute template for an equipment type.

        Args:
            equipment_type: The type of equipment

        Returns:
            Dictionary with attribute template
        """
        template = self.get_equipment_template(equipment_type)
        if not template:
            return {"error": f"No template found for equipment type: {equipment_type}"}

        result = {
            "equipment_type": equipment_type,
            "classification": self.get_classification_ids(equipment_type),
            "required_attributes": {},
            "optional_attributes": {},
            "performance_fields": self.get_performance_fields(equipment_type),
        }

        # Add required attributes with units
        for attr in template.get("required_attributes", []):
            unit = self.get_attribute_unit(equipment_type, attr)
            result["required_attributes"][attr] = {"value": None, "unit": unit}

        # Add optional attributes with units
        for attr in template.get("optional_attributes", []):
            unit = self.get_attribute_unit(equipment_type, attr)
            result["optional_attributes"][attr] = {"value": None, "unit": unit}

        return result

    def fill_missing_attributes(
        self,
        equipment_type: str,
        attributes: Dict[str, Any],
        description: str,
        model=None,
    ) -> Dict[str, Any]:
        """
        Fill in missing attributes using ML predictions and rules.

        Args:
            equipment_type: The type of equipment
            attributes: Dictionary of existing attribute name-value pairs
            description: Text description of the equipment
            model: Optional ML model to use for predictions

        Returns:
            Dictionary with filled attributes
        """
        result = attributes.copy()
        template = self.get_equipment_template(equipment_type)

        # Get all attributes that should be present
        all_attrs = self.get_all_attributes(equipment_type)

        # Identify missing attributes
        missing_attrs = [
            attr
            for attr in all_attrs
            if attr not in attributes or attributes[attr] is None
        ]

        if not missing_attrs:
            return result  # No missing attributes to fill

        # Fill in performance fields from template defaults
        perf_fields = self.get_performance_fields(equipment_type)
        for field, info in perf_fields.items():
            if field not in result or result[field] is None:
                result[field] = info.get("default")

        # If we have a model, use it to predict missing attributes
        if model and hasattr(model, "predict_attributes"):
            predictions = model.predict_attributes(equipment_type, description)
            for attr, value in predictions.items():
                if attr in missing_attrs:
                    result[attr] = value

        return result


class EAVTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that adds EAV attributes to the feature set.
    """

    def __init__(self, eav_manager: Optional[EAVManager] = None):
        """
        Initialize the EAV Transformer.

        Args:
            eav_manager: EAVManager instance. If None, creates a new one.
        """
        self.eav_manager = eav_manager or EAVManager()

    def fit(self, X, y=None):
        """Fit method (does nothing but is required for the transformer interface)."""
        return self

    def transform(self, X):
        """
        Transform the input DataFrame by adding EAV attributes.

        Args:
            X: Input DataFrame with at least 'Equipment_Category' column

        Returns:
            Transformed DataFrame with EAV attributes
        """
        X = X.copy()

        # Check if Equipment_Category column exists
        if "Equipment_Category" not in X.columns:
            print(
                "Warning: 'Equipment_Category' column not found in EAVTransformer. Adding empty EAV attributes."
            )
            # Add empty columns for EAV attributes
            X["omniclass_id"] = ""
            X["masterformat_id"] = ""
            X["uniformat_id"] = ""
            X["default_service_life"] = 0
            X["maintenance_interval"] = 0
            X["required_attribute_count"] = 0
            return X

        # Add classification IDs
        X["omniclass_id"] = X["Equipment_Category"].apply(
            lambda x: self.eav_manager.get_classification_ids(x).get("omniclass_id", "")
        )
        X["masterformat_id"] = X["Equipment_Category"].apply(
            lambda x: self.eav_manager.get_classification_ids(x).get(
                "masterformat_id", ""
            )
        )
        X["uniformat_id"] = X["Equipment_Category"].apply(
            lambda x: self.eav_manager.get_classification_ids(x).get("uniformat_id", "")
        )

        # Add performance fields
        X["default_service_life"] = X["Equipment_Category"].apply(
            lambda x: self.eav_manager.get_performance_fields(x)
            .get("service_life", {})
            .get("default", 0)
        )
        X["maintenance_interval"] = X["Equipment_Category"].apply(
            lambda x: self.eav_manager.get_performance_fields(x)
            .get("maintenance_interval", {})
            .get("default", 0)
        )

        # Create a feature indicating how many required attributes are typically needed
        X["required_attribute_count"] = X["Equipment_Category"].apply(
            lambda x: len(self.eav_manager.get_required_attributes(x))
        )

        return X


def get_eav_manager() -> EAVManager:
    """
    Get an instance of the EAVManager.

    Returns:
        EAVManager instance
    """
    return EAVManager()
