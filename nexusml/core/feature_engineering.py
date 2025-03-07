"""
Feature Engineering Module

This module handles feature engineering for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on feature transformations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.config import get_project_root
from nexusml.core.eav_manager import EAVManager, EAVTransformer


class TextCombiner(BaseEstimator, TransformerMixin):
    """
    Combines multiple text columns into one column.

    Config example: {"columns": ["Asset Category","Equip Name ID"], "separator": " "}
    """

    def __init__(
        self,
        columns: List[str],
        separator: str = " ",
        new_column: str = "combined_text",
    ):
        self.columns = columns
        self.separator = separator
        self.new_column = new_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Check if all columns exist
        missing_columns = [col for col in self.columns if col not in X.columns]
        if missing_columns:
            print(
                f"Warning: Columns {missing_columns} not found for TextCombiner. Using available columns only."
            )
            available_columns = [col for col in self.columns if col in X.columns]
            if not available_columns:
                print(
                    f"No columns available for TextCombiner. Creating empty column {self.new_column}."
                )
                X[self.new_column] = ""
                return X
            # Create a single text column from available columns
            X[self.new_column] = (
                X[available_columns]
                .astype(str)
                .apply(lambda row: self.separator.join(row.values), axis=1)
            )
        else:
            # Create a single text column from all specified columns
            X[self.new_column] = (
                X[self.columns]
                .astype(str)
                .apply(lambda row: self.separator.join(row.values), axis=1)
            )
        X[self.new_column] = X[self.new_column].fillna("")
        return X


class NumericCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans and transforms numeric columns.

    Config example: {"name": "Service Life", "new_name": "service_life", "fill_value": 0, "dtype": "float"}
    """

    def __init__(
        self,
        column: str,
        new_name: Optional[str] = None,
        fill_value: Union[int, float] = 0,
        dtype: str = "float",
    ):
        self.column = column
        self.new_name = new_name or column
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Check if the column exists
        if self.column not in X.columns:
            print(
                f"Warning: Column '{self.column}' not found for NumericCleaner. Creating column with default value."
            )
            if self.dtype == "float":
                X[self.new_name] = float(self.fill_value)
            elif self.dtype == "int":
                X[self.new_name] = int(self.fill_value)
            return X

        # Clean and transform the numeric column
        if self.dtype == "float":
            X[self.new_name] = X[self.column].fillna(self.fill_value).astype(float)
        elif self.dtype == "int":
            X[self.new_name] = X[self.column].fillna(self.fill_value).astype(int)
        return X


class HierarchyBuilder(BaseEstimator, TransformerMixin):
    """
    Creates hierarchical category columns by combining parent columns.

    Config example: {"new_col": "Equipment_Type", "parents": ["Asset Category", "Equip Name ID"], "separator": "-"}
    """

    def __init__(
        self, new_column: str, parent_columns: List[str], separator: str = "-"
    ):
        self.new_column = new_column
        self.parent_columns = parent_columns
        self.separator = separator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Check if all parent columns exist
        missing_columns = [col for col in self.parent_columns if col not in X.columns]
        if missing_columns:
            print(
                f"Warning: Columns {missing_columns} not found for HierarchyBuilder. Using available columns only."
            )
            available_columns = [col for col in self.parent_columns if col in X.columns]
            if not available_columns:
                print(
                    f"No columns available for HierarchyBuilder. Creating empty column {self.new_column}."
                )
                X[self.new_column] = ""
                return X
            # Create hierarchical column from available parent columns
            X[self.new_column] = (
                X[available_columns]
                .astype(str)
                .apply(lambda row: self.separator.join(row.values), axis=1)
            )
        else:
            # Create hierarchical column from all parent columns
            X[self.new_column] = (
                X[self.parent_columns]
                .astype(str)
                .apply(lambda row: self.separator.join(row.values), axis=1)
            )
        return X


class ColumnMapper(BaseEstimator, TransformerMixin):
    """
    Maps source columns to target columns.

    Config example: {"source": "Asset Category", "target": "Equipment_Category"}
    """

    def __init__(self, mappings: List[Dict[str, str]]):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Map source columns to target columns
        for mapping in self.mappings:
            source = mapping["source"]
            target = mapping["target"]
            if source in X.columns:
                X[target] = X[source]
            else:
                print(
                    f"Warning: Source column '{source}' not found in DataFrame. Skipping mapping to '{target}'."
                )
        return X


class KeywordClassificationMapper(BaseEstimator, TransformerMixin):
    """
    Maps equipment descriptions to classification system IDs using keyword matching.

    Config example: {
        "name": "Uniformat",
        "source_column": "combined_text",
        "target_column": "Uniformat_Class",
        "reference_manager": "uniformat_keywords"
    }
    """

    def __init__(
        self,
        name: str,
        source_column: str,
        target_column: str,
        reference_manager: str = "uniformat_keywords",
        max_results: int = 1,
        confidence_threshold: float = 0.0,
    ):
        """
        Initialize the transformer.

        Args:
            name: Name of the classification system
            source_column: Column containing text to search for keywords
            target_column: Column to store the matched classification code
            reference_manager: Reference manager to use for keyword matching
            max_results: Maximum number of results to consider
            confidence_threshold: Minimum confidence score to accept a match
        """
        self.name = name
        self.source_column = source_column
        self.target_column = target_column
        self.reference_manager = reference_manager
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold

        # Initialize reference manager
        from nexusml.core.reference.manager import ReferenceManager

        self.ref_manager = ReferenceManager()
        self.ref_manager.load_all()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transform the input DataFrame by adding classification codes based on keyword matching."""
        X = X.copy()

        # Check if the source column exists
        if self.source_column not in X.columns:
            print(
                f"Warning: Source column '{self.source_column}' not found for KeywordClassificationMapper. Setting target column to empty."
            )
            X[self.target_column] = ""
            return X

        # Apply keyword matching
        if (
            self.name.lower() == "uniformat"
            and self.reference_manager == "uniformat_keywords"
        ):
            # Only process rows where the target column is empty or NaN
            mask = X[self.target_column].isna() | (X[self.target_column] == "")
            if mask.any():

                def find_uniformat_code(text):
                    if pd.isna(text) or text == "":
                        return ""

                    # Find Uniformat codes by keyword
                    results = self.ref_manager.find_uniformat_codes_by_keyword(
                        text, self.max_results
                    )
                    if results:
                        return results[0]["uniformat_code"]
                    return ""

                # Apply the function to find codes
                X.loc[mask, self.target_column] = X.loc[mask, self.source_column].apply(
                    find_uniformat_code
                )

        return X


class ClassificationSystemMapper(BaseEstimator, TransformerMixin):
    """
    Maps equipment categories to classification system IDs (OmniClass, MasterFormat, Uniformat).

    Config example: {
        "name": "OmniClass",
        "source_column": "Equipment_Category",
        "target_column": "OmniClass_ID",
        "mapping_type": "eav"
    }
    """

    def __init__(
        self,
        name: str,
        source_column: Union[str, List[str]],
        target_column: str,
        mapping_type: str = "eav",
        mapping_function: Optional[str] = None,
        eav_manager: Optional[EAVManager] = None,
    ):
        self.name = name
        self.source_column = source_column
        self.target_column = target_column
        self.mapping_type = mapping_type
        self.mapping_function = mapping_function
        self.eav_manager = eav_manager or EAVManager()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Handle different mapping types
        if self.mapping_type == "eav":
            # Use EAV manager to get classification IDs
            if isinstance(self.source_column, str):
                # Check if the source column exists
                if self.source_column not in X.columns:
                    print(
                        f"Warning: Source column '{self.source_column}' not found for ClassificationSystemMapper. Setting target column to empty."
                    )
                    X[self.target_column] = ""
                    return X

                # Single source column
                if self.name.lower() == "omniclass":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "omniclass_id", ""
                        )
                    )
                elif self.name.lower() == "masterformat":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "masterformat_id", ""
                        )
                    )
                elif self.name.lower() == "uniformat":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "uniformat_id", ""
                        )
                    )
            else:
                # Multiple source columns not supported for EAV mapping
                print(
                    f"Warning: Multiple source columns not supported for EAV mapping of {self.name}"
                )

        elif self.mapping_function == "enhanced_masterformat_mapping":
            # Use the enhanced_masterformat_mapping function
            if isinstance(self.source_column, list) and len(self.source_column) >= 3:
                # Extract the required columns
                uniformat_col = self.source_column[0]
                system_type_col = self.source_column[1]
                equipment_category_col = self.source_column[2]
                equipment_subcategory_col = (
                    self.source_column[3] if len(self.source_column) > 3 else None
                )

                # Check if all required columns exist
                missing_columns = [
                    col
                    for col in [uniformat_col, system_type_col, equipment_category_col]
                    if col not in X.columns
                ]
                if (
                    equipment_subcategory_col
                    and equipment_subcategory_col not in X.columns
                ):
                    missing_columns.append(equipment_subcategory_col)

                if missing_columns:
                    print(
                        f"Warning: Columns {missing_columns} not found for enhanced_masterformat_mapping. Setting target column to empty."
                    )
                    X[self.target_column] = ""
                    return X

                # Apply the mapping function
                X[self.target_column] = X.apply(
                    lambda row: enhanced_masterformat_mapping(
                        row[uniformat_col],
                        row[system_type_col],
                        row[equipment_category_col],
                        (
                            row[equipment_subcategory_col]
                            if equipment_subcategory_col
                            else None
                        ),
                    ),
                    axis=1,
                )
            else:
                print(
                    f"Warning: enhanced_masterformat_mapping requires at least 3 source columns"
                )

        return X


from nexusml.core.di.decorators import inject, injectable


@injectable
class GenericFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A generic feature engineering transformer that applies multiple transformations
    based on a configuration file.

    This class uses dependency injection to receive its dependencies,
    making it more testable and configurable.
    """

    @inject
    def __init__(
        self,
        config_path: Optional[str] = None,
        eav_manager: Optional[EAVManager] = None,
    ):
        """
        Initialize the transformer with a configuration file path.

        Args:
            config_path: Path to the YAML configuration file. If None, uses the default path.
            eav_manager: EAVManager instance (injected). If None, uses the one from the DI container.
        """
        self.config_path = config_path
        self.transformers = []
        self.config = {}

        # Get EAV manager from DI container if not provided
        if eav_manager is None:
            try:
                from nexusml.core.di.provider import ContainerProvider

                container = ContainerProvider().container
                self.eav_manager = container.resolve(EAVManager)
            except Exception:
                # Fallback for backward compatibility
                self.eav_manager = EAVManager()
        else:
            self.eav_manager = eav_manager

        # Load the configuration
        try:
            self._load_config()
        except AttributeError:
            # Handle the case when _load_config is called on the class instead of an instance
            # This can happen in the backward compatibility test
            pass

    def _load_config(self):
        """Load the configuration from the YAML file."""
        config_path = self.config_path
        if config_path is None:
            # Use default path
            root = get_project_root()
            config_path = root / "config" / "feature_config.yml"

        # Load the configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform the input DataFrame based on the configuration.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        X = X.copy()

        # 1. Apply column mappings
        if "column_mappings" in self.config:
            mapper = ColumnMapper(self.config["column_mappings"])
            X = mapper.transform(X)

        # 2. Apply text combinations
        if "text_combinations" in self.config:
            for combo in self.config["text_combinations"]:
                combiner = TextCombiner(
                    columns=combo["columns"],
                    separator=combo.get("separator", " "),
                    new_column=combo.get("name", "combined_text"),
                )
                X = combiner.transform(X)

        # 3. Apply numeric column cleaning
        if "numeric_columns" in self.config:
            for num_col in self.config["numeric_columns"]:
                cleaner = NumericCleaner(
                    column=num_col["name"],
                    new_name=num_col.get("new_name", num_col["name"]),
                    fill_value=num_col.get("fill_value", 0),
                    dtype=num_col.get("dtype", "float"),
                )
                X = cleaner.transform(X)

        # 4. Apply hierarchical categories
        if "hierarchies" in self.config:
            for hierarchy in self.config["hierarchies"]:
                builder = HierarchyBuilder(
                    new_column=hierarchy["new_col"],
                    parent_columns=hierarchy["parents"],
                    separator=hierarchy.get("separator", "-"),
                )
                X = builder.transform(X)

        # 4.5. Apply keyword classification mappings
        if "keyword_classifications" in self.config:
            for system in self.config["keyword_classifications"]:
                mapper = KeywordClassificationMapper(
                    name=system["name"],
                    source_column=system["source_column"],
                    target_column=system["target_column"],
                    reference_manager=system.get(
                        "reference_manager", "uniformat_keywords"
                    ),
                    max_results=system.get("max_results", 1),
                    confidence_threshold=system.get("confidence_threshold", 0.0),
                )
                X = mapper.transform(X)

        # 5. Apply classification system mappings
        if "classification_systems" in self.config:
            for system in self.config["classification_systems"]:
                mapper = ClassificationSystemMapper(
                    name=system["name"],
                    source_column=system.get("source_column")
                    or system.get("source_columns", []),
                    target_column=system["target_column"],
                    mapping_type=system.get("mapping_type", "eav"),
                    mapping_function=system.get("mapping_function"),
                    eav_manager=self.eav_manager,
                )
                X = mapper.transform(X)

        # 6. Apply EAV integration if enabled
        if "eav_integration" in self.config and self.config["eav_integration"].get(
            "enabled", False
        ):
            eav_config = self.config["eav_integration"]
            eav_transformer = EAVTransformer(eav_manager=self.eav_manager)
            X = eav_transformer.transform(X)

        return X


def enhance_features(
    df: pd.DataFrame, feature_engineer: Optional[GenericFeatureEngineer] = None
) -> pd.DataFrame:
    """
    Enhanced feature engineering with hierarchical structure and more granular categories

    This function now uses the GenericFeatureEngineer transformer to apply
    transformations based on the configuration file.

    Args:
        df (pd.DataFrame): Input dataframe with raw features
        feature_engineer (Optional[GenericFeatureEngineer]): Feature engineer instance.
            If None, uses the one from the DI container.

    Returns:
        pd.DataFrame: DataFrame with enhanced features
    """
    # Get feature engineer from DI container if not provided
    if feature_engineer is None:
        from nexusml.core.di.provider import ContainerProvider

        container = ContainerProvider().container
        feature_engineer = container.resolve(GenericFeatureEngineer)

    # Apply transformations
    return feature_engineer.transform(df)


def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical category structure to better handle "Other" categories

    This function is kept for backward compatibility but now simply returns the
    input DataFrame as the hierarchical categories are created by the GenericFeatureEngineer.

    Args:
        df (pd.DataFrame): Input dataframe with basic features

    Returns:
        pd.DataFrame: DataFrame with hierarchical category features
    """
    # This function is kept for backward compatibility
    # The hierarchical categories are now created by the GenericFeatureEngineer
    return df


def load_masterformat_mappings() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Load MasterFormat mappings from JSON files.

    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Primary and equipment-specific mappings
    """
    root = get_project_root()

    try:
        with open(root / "config" / "mappings" / "masterformat_primary.json") as f:
            primary_mapping = json.load(f)

        with open(root / "config" / "mappings" / "masterformat_equipment.json") as f:
            equipment_specific_mapping = json.load(f)

        return primary_mapping, equipment_specific_mapping
    except Exception as e:
        print(f"Warning: Could not load MasterFormat mappings: {e}")
        # Return empty mappings if files cannot be loaded
        return {}, {}


def enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None,
    eav_manager: Optional[EAVManager] = None,
) -> str:
    """
    Enhanced mapping with better handling of specialty equipment types

    Args:
        uniformat_class (str): Uniformat classification
        system_type (str): System type
        equipment_category (str): Equipment category
        equipment_subcategory (Optional[str]): Equipment subcategory
        eav_manager (Optional[EAVManager]): EAV manager instance. If None, uses the one from the DI container.

    Returns:
        str: MasterFormat classification code
    """
    # Load mappings from JSON files
    primary_mapping, equipment_specific_mapping = load_masterformat_mappings()

    # Try equipment-specific mapping first
    if equipment_subcategory in equipment_specific_mapping:
        return equipment_specific_mapping[equipment_subcategory]

    # Then try primary mapping
    if (
        uniformat_class in primary_mapping
        and system_type in primary_mapping[uniformat_class]
    ):
        return primary_mapping[uniformat_class][system_type]

    # Try EAV-based mapping
    try:
        # Get EAV manager from DI container if not provided
        if eav_manager is None:
            from nexusml.core.di.provider import ContainerProvider

            container = ContainerProvider().container
            eav_manager = container.resolve(EAVManager)

        masterformat_id = eav_manager.get_classification_ids(equipment_category).get(
            "masterformat_id", ""
        )
        if masterformat_id:
            return masterformat_id
    except Exception as e:
        print(f"Warning: Could not use EAV for MasterFormat mapping: {e}")

    # Refined fallback mappings by Uniformat class
    fallbacks = {
        "H": "23 00 00",  # Heating, Ventilating, and Air Conditioning (HVAC)
        "P": "22 00 00",  # Plumbing
        "SM": "23 00 00",  # HVAC
        "R": "11 40 00",  # Foodservice Equipment (Refrigeration)
    }

    return fallbacks.get(uniformat_class, "00 00 00")  # Return unknown if no match
