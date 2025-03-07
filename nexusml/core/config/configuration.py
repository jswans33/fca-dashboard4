"""
Configuration models for NexusML.

This module contains Pydantic models for validating and managing NexusML configuration.
It provides a unified interface for all configuration settings used throughout the system.

Note: The legacy configuration files are maintained for backward compatibility
and are planned for removal in future work chunks. Once all code is updated to
use the new unified configuration system, these files will be removed.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, RootModel, root_validator, validator


class TextCombination(BaseModel):
    """Configuration for text field combinations."""

    name: str = Field(..., description="Name of the combined field")
    columns: List[str] = Field(..., description="List of columns to combine")
    separator: str = Field(" ", description="Separator to use between combined fields")


class NumericColumn(BaseModel):
    """Configuration for numeric column processing."""

    name: str = Field(..., description="Original column name")
    new_name: Optional[str] = Field(None, description="New column name (if renaming)")
    fill_value: Union[int, float] = Field(
        0, description="Value to use for missing data"
    )
    dtype: str = Field("float", description="Data type for the column")


class Hierarchy(BaseModel):
    """Configuration for hierarchical field creation."""

    new_col: str = Field(..., description="Name of the new hierarchical column")
    parents: List[str] = Field(
        ..., description="List of parent columns in hierarchy order"
    )
    separator: str = Field("-", description="Separator to use between hierarchy levels")


class ColumnMapping(BaseModel):
    """Configuration for column mapping."""

    source: str = Field(..., description="Source column name")
    target: str = Field(..., description="Target column name")


class ClassificationSystem(BaseModel):
    """Configuration for classification system mapping."""

    name: str = Field(..., description="Name of the classification system")
    source_column: str = Field(
        ..., description="Source column containing classification codes"
    )
    target_column: str = Field(
        ..., description="Target column for mapped classifications"
    )
    mapping_type: str = Field(
        "direct", description="Type of mapping (direct, function, eav)"
    )


class EAVIntegration(BaseModel):
    """Configuration for Entity-Attribute-Value integration."""

    enabled: bool = Field(False, description="Whether EAV integration is enabled")


class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering."""

    text_combinations: List[TextCombination] = Field(
        default_factory=list, description="Text field combinations"
    )
    numeric_columns: List[NumericColumn] = Field(
        default_factory=list, description="Numeric column configurations"
    )
    hierarchies: List[Hierarchy] = Field(
        default_factory=list, description="Hierarchical field configurations"
    )
    column_mappings: List[ColumnMapping] = Field(
        default_factory=list, description="Column mapping configurations"
    )
    classification_systems: List[ClassificationSystem] = Field(
        default_factory=list, description="Classification system configurations"
    )
    direct_mappings: List[ColumnMapping] = Field(
        default_factory=list, description="Direct column mapping configurations"
    )
    eav_integration: EAVIntegration = Field(
        default_factory=lambda: EAVIntegration(enabled=False),
        description="EAV integration configuration",
    )


class ClassificationTarget(BaseModel):
    """Configuration for a classification target."""

    name: str = Field(..., description="Name of the classification target")
    description: str = Field("", description="Description of the classification target")
    required: bool = Field(False, description="Whether this classification is required")
    master_db: Optional[Dict[str, str]] = Field(
        None, description="Master database configuration for this target"
    )


class InputFieldMapping(BaseModel):
    """Configuration for input field mapping."""

    target: str = Field(..., description="Target standardized field name")
    patterns: List[str] = Field(..., description="Patterns to match in input data")


class ClassificationConfig(BaseModel):
    """Configuration for classification."""

    classification_targets: List[ClassificationTarget] = Field(
        default_factory=list, description="Classification targets"
    )
    input_field_mappings: List[InputFieldMapping] = Field(
        default_factory=list, description="Input field mapping configurations"
    )


class RequiredColumn(BaseModel):
    """Configuration for a required column."""

    name: str = Field(..., description="Column name")
    default_value: Any = Field(None, description="Default value if column is missing")
    data_type: str = Field("str", description="Data type for the column")


class TrainingDataConfig(BaseModel):
    """Configuration for training data."""

    default_path: str = Field(
        "nexusml/data/training_data/fake_training_data.csv",
        description="Default path to training data",
    )
    encoding: str = Field("utf-8", description="File encoding")
    fallback_encoding: str = Field(
        "latin1", description="Fallback encoding if primary fails"
    )


class DataConfig(BaseModel):
    """Configuration for data preprocessing."""

    required_columns: List[RequiredColumn] = Field(
        default_factory=list, description="Required columns configuration"
    )
    training_data: TrainingDataConfig = Field(
        default_factory=lambda: TrainingDataConfig(
            default_path="nexusml/data/training_data/fake_training_data.csv",
            encoding="utf-8",
            fallback_encoding="latin1",
        ),
        description="Training data configuration",
    )


class PathConfig(BaseModel):
    """Configuration for reference data paths."""

    omniclass: str = Field(
        "nexusml/ingest/reference/omniclass",
        description="Path to OmniClass reference data",
    )
    uniformat: str = Field(
        "nexusml/ingest/reference/uniformat",
        description="Path to UniFormat reference data",
    )
    masterformat: str = Field(
        "nexusml/ingest/reference/masterformat",
        description="Path to MasterFormat reference data",
    )
    mcaa_glossary: str = Field(
        "nexusml/ingest/reference/mcaa-glossary", description="Path to MCAA glossary"
    )
    mcaa_abbreviations: str = Field(
        "nexusml/ingest/reference/mcaa-glossary",
        description="Path to MCAA abbreviations",
    )
    smacna: str = Field(
        "nexusml/ingest/reference/smacna-manufacturers",
        description="Path to SMACNA data",
    )
    ashrae: str = Field(
        "nexusml/ingest/reference/service-life/ashrae",
        description="Path to ASHRAE data",
    )
    energize_denver: str = Field(
        "nexusml/ingest/reference/service-life/energize-denver",
        description="Path to Energize Denver data",
    )
    equipment_taxonomy: str = Field(
        "nexusml/ingest/reference/equipment-taxonomy",
        description="Path to equipment taxonomy data",
    )


class FilePatternConfig(BaseModel):
    """Configuration for reference data file patterns."""

    omniclass: str = Field("*.csv", description="File pattern for OmniClass data")
    uniformat: str = Field("*.csv", description="File pattern for UniFormat data")
    masterformat: str = Field("*.csv", description="File pattern for MasterFormat data")
    mcaa_glossary: str = Field(
        "Glossary.csv", description="File pattern for MCAA glossary"
    )
    mcaa_abbreviations: str = Field(
        "Abbreviations.csv", description="File pattern for MCAA abbreviations"
    )
    smacna: str = Field("*.json", description="File pattern for SMACNA data")
    ashrae: str = Field("*.csv", description="File pattern for ASHRAE data")
    energize_denver: str = Field(
        "*.csv", description="File pattern for Energize Denver data"
    )
    equipment_taxonomy: str = Field(
        "*.csv", description="File pattern for equipment taxonomy data"
    )


class ColumnMappingGroup(BaseModel):
    """Configuration for a group of column mappings."""

    code: str = Field(..., description="Column name for code")
    name: str = Field(..., description="Column name for name")
    description: str = Field(..., description="Column name for description")


class ServiceLifeMapping(BaseModel):
    """Configuration for service life mapping."""

    equipment_type: str = Field(..., description="Column name for equipment type")
    median_years: str = Field(..., description="Column name for median years")
    min_years: str = Field(..., description="Column name for minimum years")
    max_years: str = Field(..., description="Column name for maximum years")
    source: str = Field(..., description="Column name for source")


class EquipmentTaxonomyMapping(BaseModel):
    """Configuration for equipment taxonomy mapping."""

    asset_category: str = Field(..., description="Column name for asset category")
    equipment_id: str = Field(..., description="Column name for equipment ID")
    trade: str = Field(..., description="Column name for trade")
    title: str = Field(..., description="Column name for title")
    drawing_abbreviation: str = Field(
        ..., description="Column name for drawing abbreviation"
    )
    precon_tag: str = Field(..., description="Column name for precon tag")
    system_type_id: str = Field(..., description="Column name for system type ID")
    sub_system_type: str = Field(..., description="Column name for sub-system type")
    sub_system_id: str = Field(..., description="Column name for sub-system ID")
    sub_system_class: str = Field(..., description="Column name for sub-system class")
    class_id: str = Field(..., description="Column name for class ID")
    equipment_size: str = Field(..., description="Column name for equipment size")
    unit: str = Field(..., description="Column name for unit")
    service_maintenance_hrs: str = Field(
        ..., description="Column name for service maintenance hours"
    )
    service_life: str = Field(..., description="Column name for service life")


class ReferenceColumnMappings(BaseModel):
    """Configuration for reference data column mappings."""

    omniclass: ColumnMappingGroup = Field(
        ..., description="Column mappings for OmniClass data"
    )
    uniformat: ColumnMappingGroup = Field(
        ..., description="Column mappings for UniFormat data"
    )
    masterformat: ColumnMappingGroup = Field(
        ..., description="Column mappings for MasterFormat data"
    )
    service_life: ServiceLifeMapping = Field(
        ..., description="Column mappings for service life data"
    )
    equipment_taxonomy: EquipmentTaxonomyMapping = Field(
        ..., description="Column mappings for equipment taxonomy data"
    )


class HierarchyConfig(BaseModel):
    """Configuration for hierarchy."""

    separator: str = Field("", description="Separator for hierarchy levels")
    levels: int = Field(1, description="Number of hierarchy levels")


class HierarchiesConfig(BaseModel):
    """Configuration for hierarchies."""

    omniclass: HierarchyConfig = Field(
        ..., description="Hierarchy configuration for OmniClass"
    )
    uniformat: HierarchyConfig = Field(
        ..., description="Hierarchy configuration for UniFormat"
    )
    masterformat: HierarchyConfig = Field(
        ..., description="Hierarchy configuration for MasterFormat"
    )


class DefaultsConfig(BaseModel):
    """Configuration for default values."""

    service_life: float = Field(15.0, description="Default service life in years")
    confidence: float = Field(0.5, description="Default confidence level")


class ReferenceConfig(BaseModel):
    """Configuration for reference data."""

    paths: PathConfig = Field(
        default_factory=lambda: PathConfig(
            omniclass="nexusml/ingest/reference/omniclass",
            uniformat="nexusml/ingest/reference/uniformat",
            masterformat="nexusml/ingest/reference/masterformat",
            mcaa_glossary="nexusml/ingest/reference/mcaa-glossary",
            mcaa_abbreviations="nexusml/ingest/reference/mcaa-glossary",
            smacna="nexusml/ingest/reference/smacna-manufacturers",
            ashrae="nexusml/ingest/reference/service-life/ashrae",
            energize_denver="nexusml/ingest/reference/service-life/energize-denver",
            equipment_taxonomy="nexusml/ingest/reference/equipment-taxonomy",
        ),
        description="Reference data paths",
    )
    file_patterns: FilePatternConfig = Field(
        default_factory=lambda: FilePatternConfig(
            omniclass="*.csv",
            uniformat="*.csv",
            masterformat="*.csv",
            mcaa_glossary="Glossary.csv",
            mcaa_abbreviations="Abbreviations.csv",
            smacna="*.json",
            ashrae="*.csv",
            energize_denver="*.csv",
            equipment_taxonomy="*.csv",
        ),
        description="Reference data file patterns",
    )
    column_mappings: ReferenceColumnMappings = Field(
        ..., description="Reference data column mappings"
    )
    hierarchies: HierarchiesConfig = Field(
        ..., description="Reference data hierarchy configurations"
    )
    defaults: DefaultsConfig = Field(
        default_factory=lambda: DefaultsConfig(service_life=15.0, confidence=0.5),
        description="Default values",
    )


class EquipmentAttribute(BaseModel):
    """Configuration for equipment attributes."""

    omniclass_id: str = Field(..., description="OmniClass ID")
    masterformat_id: str = Field(..., description="MasterFormat ID")
    uniformat_id: str = Field(..., description="UniFormat ID")
    required_attributes: List[str] = Field(
        default_factory=list, description="Required attributes"
    )
    optional_attributes: List[str] = Field(
        default_factory=list, description="Optional attributes"
    )
    units: Dict[str, str] = Field(
        default_factory=dict, description="Units for attributes"
    )
    performance_fields: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Performance field configurations"
    )


class MasterFormatMapping(RootModel):
    """Configuration for MasterFormat mappings."""

    root: Dict[str, Dict[str, str]] = Field(
        ..., description="MasterFormat mappings by system type"
    )


class EquipmentMasterFormatMapping(RootModel):
    """Configuration for equipment-specific MasterFormat mappings."""

    root: Dict[str, str] = Field(
        ..., description="MasterFormat mappings by equipment type"
    )


class NexusMLConfig(BaseModel):
    """Main configuration class for NexusML."""

    feature_engineering: FeatureEngineeringConfig = Field(
        default_factory=FeatureEngineeringConfig,
        description="Feature engineering configuration",
    )
    classification: ClassificationConfig = Field(
        default_factory=ClassificationConfig,
        description="Classification configuration",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data preprocessing configuration",
    )
    reference: Optional[ReferenceConfig] = Field(
        None,
        description="Reference data configuration",
    )
    equipment_attributes: Dict[str, EquipmentAttribute] = Field(
        default_factory=dict,
        description="Equipment attributes configuration",
    )
    masterformat_primary: Optional[MasterFormatMapping] = Field(
        None,
        description="Primary MasterFormat mappings",
    )
    masterformat_equipment: Optional[EquipmentMasterFormatMapping] = Field(
        None,
        description="Equipment-specific MasterFormat mappings",
    )

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "NexusMLConfig":
        """
        Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML configuration file

        Returns:
            NexusMLConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return cls.model_validate(config_dict)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            file_path: Path to save the YAML configuration file

        Raises:
            IOError: If the file cannot be written
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_env(cls) -> "NexusMLConfig":
        """
        Load configuration from the path specified in the NEXUSML_CONFIG environment variable.

        Returns:
            NexusMLConfig: Loaded and validated configuration

        Raises:
            ValueError: If the NEXUSML_CONFIG environment variable is not set
            FileNotFoundError: If the configuration file doesn't exist
        """
        config_path = os.environ.get("NEXUSML_CONFIG")
        if not config_path:
            raise ValueError(
                "NEXUSML_CONFIG environment variable not set. "
                "Please set it to the path of your configuration file."
            )
        return cls.from_yaml(config_path)

    @classmethod
    def default_config_path(cls) -> Path:
        """
        Get the default configuration file path.

        Returns:
            Path: Default configuration file path
        """
        return Path("nexusml/config/nexusml_config.yml")
