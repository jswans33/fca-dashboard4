"""
Categorical Transformers Module

This module provides transformers for categorical features in the NexusML suite.
Each transformer follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a specific categorical transformation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from nexusml.core.feature_engineering.base import BaseColumnTransformer, BaseConfigurableTransformer
from nexusml.core.feature_engineering.registry import register_transformer


class ColumnMapper(BaseColumnTransformer):
    """
    Maps source columns to target columns.
    
    This transformer copies values from source columns to target columns,
    allowing for column renaming and reorganization.
    
    Config example: {"mappings": [{"source": "Asset Category", "target": "Equipment_Category"}]}
    """
    
    def __init__(
        self,
        mappings: List[Dict[str, str]],
        name: str = "ColumnMapper",
    ):
        """
        Initialize the column mapper.
        
        Args:
            mappings: List of mappings from source columns to target columns.
                Each mapping should be a dictionary with "source" and "target" keys.
            name: Name of the transformer.
        """
        source_columns = [mapping["source"] for mapping in mappings]
        target_columns = [mapping["target"] for mapping in mappings]
        super().__init__(source_columns, target_columns, name)
        self.mappings = mappings
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map source columns to target columns.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with mapped columns.
        """
        # Map source columns to target columns
        for mapping in self.mappings:
            source = mapping["source"]
            target = mapping["target"]
            X[target] = X[source]
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If some source columns are missing, skip the corresponding mappings.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Map only the available source columns
        for mapping in self.mappings:
            source = mapping["source"]
            target = mapping["target"]
            if source in X.columns:
                X[target] = X[source]
        
        return X


class OneHotEncoder(BaseColumnTransformer):
    """
    Performs one-hot encoding on categorical columns.
    
    This transformer converts categorical columns into one-hot encoded columns,
    creating a binary column for each category.
    
    Config example: {
        "column": "color",
        "prefix": "color_",
        "drop_first": true,
        "handle_unknown": "ignore"
    }
    """
    
    def __init__(
        self,
        column: str,
        prefix: Optional[str] = None,
        drop_first: bool = False,
        handle_unknown: str = "ignore",
        name: str = "OneHotEncoder",
    ):
        """
        Initialize the one-hot encoder.
        
        Args:
            column: Name of the categorical column to encode.
            prefix: Prefix to add to the encoded column names. If None, uses the column name.
            drop_first: Whether to drop the first category to avoid multicollinearity.
            handle_unknown: How to handle unknown categories. One of: "error", "ignore".
            name: Name of the transformer.
        """
        super().__init__([column], [], name)
        self.column = column
        self.prefix = prefix or f"{column}_"
        self.drop_first = drop_first
        self.handle_unknown = handle_unknown
        self.encoder = None
        self.categories = []
        self.output_columns = []
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OneHotEncoder':
        """
        Fit the one-hot encoder to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the input column is missing.
        """
        # Check if the input column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame")
        
        # Create the encoder
        from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
        self.encoder = SklearnOneHotEncoder(
            sparse=False,
            drop="first" if self.drop_first else None,
            handle_unknown=self.handle_unknown,
        )
        
        # Fit the encoder
        self.encoder.fit(X[[self.column]])
        
        # Get the categories
        self.categories = self.encoder.categories_[0].tolist()
        
        # Generate output column names
        if self.drop_first:
            categories = self.categories[1:]
        else:
            categories = self.categories
        
        self.output_columns = [f"{self.prefix}{category}" for category in categories]
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform one-hot encoding on the categorical column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with one-hot encoded columns.
        """
        # Encode the column
        encoded = self.encoder.transform(X[[self.column]])
        
        # Add the encoded columns to the DataFrame
        for i, col in enumerate(self.output_columns):
            X[col] = encoded[:, i]
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create empty encoded columns.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create empty encoded columns
        for col in self.output_columns:
            X[col] = 0
        
        return X
    
    def get_output_columns(self) -> List[str]:
        """
        Get the names of the output columns produced by this transformer.
        
        Returns:
            List of output column names.
        """
        return self.output_columns


class LabelEncoder(BaseColumnTransformer):
    """
    Performs label encoding on categorical columns.
    
    This transformer converts categorical values into numeric indices.
    
    Config example: {
        "column": "color",
        "new_column": "color_encoded",
        "unknown_value": -1
    }
    """
    
    def __init__(
        self,
        column: str,
        new_column: Optional[str] = None,
        unknown_value: int = -1,
        name: str = "LabelEncoder",
    ):
        """
        Initialize the label encoder.
        
        Args:
            column: Name of the categorical column to encode.
            new_column: Name of the new column to create. If None, uses "{column}_encoded".
            unknown_value: Value to use for unknown categories.
            name: Name of the transformer.
        """
        output_column = new_column or f"{column}_encoded"
        super().__init__([column], [output_column], name)
        self.column = column
        self.new_column = output_column
        self.unknown_value = unknown_value
        self.categories = []
        self.category_map = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LabelEncoder':
        """
        Fit the label encoder to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the input column is missing.
        """
        # Check if the input column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame")
        
        # Get unique categories
        self.categories = X[self.column].dropna().unique().tolist()
        
        # Create a mapping from categories to indices
        self.category_map = {category: i for i, category in enumerate(self.categories)}
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform label encoding on the categorical column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with label encoded column.
        """
        # Encode the column
        X[self.new_column] = X[self.column].map(self.category_map).fillna(self.unknown_value).astype(int)
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create an empty encoded column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create an empty encoded column
        X[self.new_column] = self.unknown_value
        
        return X


class ClassificationSystemMapper(BaseColumnTransformer):
    """
    Maps equipment categories to classification system IDs.
    
    This transformer maps equipment categories to classification system IDs
    (OmniClass, MasterFormat, Uniformat) using the EAV manager.
    
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
        eav_manager: Optional[Any] = None,
        name_prefix: str = "ClassificationSystemMapper",
    ):
        """
        Initialize the classification system mapper.
        
        Args:
            name: Name of the classification system (OmniClass, MasterFormat, Uniformat).
            source_column: Name of the source column or list of source columns.
            target_column: Name of the target column to create.
            mapping_type: Type of mapping to use. One of: "eav".
            mapping_function: Name of the mapping function to use.
            eav_manager: EAV manager instance. If None, creates a new one.
            name_prefix: Prefix for the transformer name.
        """
        input_columns = [source_column] if isinstance(source_column, str) else source_column
        super().__init__(input_columns, [target_column], f"{name_prefix}_{name}")
        self.system_name = name
        self.source_column = source_column
        self.target_column = target_column
        self.mapping_type = mapping_type
        self.mapping_function = mapping_function
        
        # Initialize EAV manager if needed
        if eav_manager is None:
            from nexusml.core.eav_manager import EAVManager
            self.eav_manager = EAVManager()
        else:
            self.eav_manager = eav_manager
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map equipment categories to classification system IDs.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with classification system IDs.
        """
        # Handle different mapping types
        if self.mapping_type == "eav":
            # Use EAV manager to get classification IDs
            if isinstance(self.source_column, str):
                # Single source column
                if self.system_name.lower() == "omniclass":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "omniclass_id", ""
                        )
                    )
                elif self.system_name.lower() == "masterformat":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "masterformat_id", ""
                        )
                    )
                elif self.system_name.lower() == "uniformat":
                    X[self.target_column] = X[self.source_column].apply(
                        lambda x: self.eav_manager.get_classification_ids(x).get(
                            "uniformat_id", ""
                        )
                    )
            else:
                # Multiple source columns not supported for EAV mapping
                X[self.target_column] = ""
        
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
                
                # Import the mapping function
                from nexusml.core.feature_engineering.transformers.mapping import enhanced_masterformat_mapping
                
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
                        self.eav_manager,
                    ),
                    axis=1,
                )
            else:
                # Not enough source columns
                X[self.target_column] = ""
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If some input columns are missing, create an empty target column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create an empty target column
        X[self.target_column] = ""
        
        return X


class KeywordClassificationMapper(BaseColumnTransformer):
    """
    Maps equipment descriptions to classification system IDs using keyword matching.
    
    This transformer maps equipment descriptions to classification system IDs
    using keyword matching through the reference manager.
    
    Config example: {
        "name": "Uniformat",
        "source_column": "combined_text",
        "target_column": "Uniformat_Class",
        "reference_manager": "uniformat_keywords",
        "max_results": 1,
        "confidence_threshold": 0.0
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
        name_prefix: str = "KeywordClassificationMapper",
    ):
        """
        Initialize the keyword classification mapper.
        
        Args:
            name: Name of the classification system (Uniformat, MasterFormat, OmniClass).
            source_column: Name of the source column containing text to search for keywords.
            target_column: Name of the target column to create.
            reference_manager: Reference manager to use for keyword matching.
            max_results: Maximum number of results to consider.
            confidence_threshold: Minimum confidence score to accept a match.
            name_prefix: Prefix for the transformer name.
        """
        super().__init__([source_column], [target_column], f"{name_prefix}_{name}")
        self.system_name = name
        self.source_column = source_column
        self.target_column = target_column
        self.reference_manager_name = reference_manager
        self.max_results = max_results
        self.confidence_threshold = confidence_threshold
        
        # Initialize reference manager
        from nexusml.core.reference.manager import ReferenceManager
        self.ref_manager = ReferenceManager()
        self.ref_manager.load_all()
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map equipment descriptions to classification system IDs using keyword matching.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with classification system IDs.
        """
        # Apply keyword matching
        if (
            self.system_name.lower() == "uniformat"
            and self.reference_manager_name == "uniformat_keywords"
        ):
            # Only process rows where the target column is empty or NaN
            if self.target_column not in X.columns:
                X[self.target_column] = ""
            
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
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the source column is missing, create an empty target column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create an empty target column
        X[self.target_column] = ""
        
        return X


# Register transformers with the registry
register_transformer("column_mapper", ColumnMapper)
register_transformer("one_hot_encoder", OneHotEncoder)
register_transformer("label_encoder", LabelEncoder)
register_transformer("classification_system_mapper", ClassificationSystemMapper)
register_transformer("keyword_classification_mapper", KeywordClassificationMapper)