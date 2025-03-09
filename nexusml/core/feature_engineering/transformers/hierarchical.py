"""
Hierarchical Transformers Module

This module provides transformers for hierarchical features in the NexusML suite.
Each transformer follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a specific hierarchical transformation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd

from nexusml.core.feature_engineering.base import BaseColumnTransformer, BaseConfigurableTransformer
from nexusml.core.feature_engineering.registry import register_transformer


class HierarchyBuilder(BaseColumnTransformer):
    """
    Creates hierarchical category columns by combining parent columns.
    
    This transformer takes multiple parent columns and combines them into a single
    hierarchical column using a specified separator.
    
    Config example: {"new_col": "Equipment_Type", "parents": ["Asset Category", "Equip Name ID"], "separator": "-"}
    """
    
    def __init__(
        self,
        parent_columns: List[str],
        new_column: str,
        separator: str = "-",
        name: str = "HierarchyBuilder",
    ):
        """
        Initialize the hierarchy builder.
        
        Args:
            parent_columns: Names of the parent columns to combine.
            new_column: Name of the new hierarchical column to create.
            separator: Separator to use between parent values.
            name: Name of the transformer.
        """
        super().__init__(parent_columns, [new_column], name)
        self.parent_columns = parent_columns
        self.new_column = new_column
        self.separator = separator
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create a hierarchical column by combining parent columns.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the hierarchical column.
        """
        # Create hierarchical column from all parent columns
        X[self.new_column] = (
            X[self.parent_columns]
            .astype(str)
            .apply(lambda row: self.separator.join(row.values), axis=1)
        )
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If some parent columns are missing, use only the available ones.
        If all parent columns are missing, create an empty hierarchical column.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Find available parent columns
        available_columns = [col for col in self.parent_columns if col in X.columns]
        
        if not available_columns:
            # If no parent columns are available, create an empty hierarchical column
            X[self.new_column] = ""
            return X
        
        # Create hierarchical column from available parent columns
        X[self.new_column] = (
            X[available_columns]
            .astype(str)
            .apply(lambda row: self.separator.join(row.values), axis=1)
        )
        
        return X


class HierarchyExpander(BaseColumnTransformer):
    """
    Expands a hierarchical column into its component parts.
    
    This transformer takes a hierarchical column and splits it into multiple
    columns, one for each level of the hierarchy.
    
    Config example: {
        "column": "Equipment_Type",
        "separator": "-",
        "level_names": ["Category", "Subcategory", "Type"],
        "prefix": "Equipment_"
    }
    """
    
    def __init__(
        self,
        column: str,
        separator: str = "-",
        level_names: Optional[List[str]] = None,
        prefix: str = "",
        name: str = "HierarchyExpander",
    ):
        """
        Initialize the hierarchy expander.
        
        Args:
            column: Name of the hierarchical column to expand.
            separator: Separator used between hierarchy levels.
            level_names: Names to use for the expanded columns. If None, uses "Level_1", "Level_2", etc.
            prefix: Prefix to add to the expanded column names.
            name: Name of the transformer.
        """
        self.column = column
        self.separator = separator
        self.level_names = level_names
        self.prefix = prefix
        self.output_columns: List[str] = []
        super().__init__([column], self.output_columns, name)
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'HierarchyExpander':
        """
        Fit the hierarchy expander to the data.
        
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
        
        # Determine the maximum number of levels in the hierarchy
        max_levels = X[self.column].str.split(self.separator).map(len).max()
        
        # Generate output column names
        if self.level_names is None:
            self.level_names = [f"Level_{i+1}" for i in range(max_levels)]
        elif len(self.level_names) < max_levels:
            # If level_names is provided but too short, extend it
            self.level_names.extend([f"Level_{i+1}" for i in range(len(self.level_names), max_levels)])
        
        # Set output columns
        self.output_columns = [f"{self.prefix}{name}" for name in self.level_names[:max_levels]]
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the hierarchical column into its component parts.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the expanded hierarchical column.
        """
        # Split the hierarchical column into its component parts
        split_values = X[self.column].str.split(self.separator, expand=True)
        
        # Rename the columns
        split_values.columns = self.output_columns[:len(split_values.columns)]
        
        # Add the expanded columns to the DataFrame
        for col in split_values.columns:
            X[col] = split_values[col]
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create empty expanded columns.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create empty expanded columns
        for col in self.output_columns:
            X[col] = ""
        
        return X
    
    def get_output_columns(self) -> List[str]:
        """
        Get the names of the output columns produced by this transformer.
        
        Returns:
            List of output column names.
        """
        return self.output_columns


class HierarchyFilter(BaseColumnTransformer):
    """
    Filters rows based on hierarchical column values.
    
    This transformer filters rows based on the values in a hierarchical column,
    allowing for filtering at different levels of the hierarchy.
    
    Config example: {
        "column": "Equipment_Type",
        "include": ["HVAC-*", "Plumbing-Fixtures-*"],
        "exclude": ["*-Other"],
        "separator": "-",
        "case_sensitive": false
    }
    """
    
    def __init__(
        self,
        column: str,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        separator: str = "-",
        case_sensitive: bool = False,
        name: str = "HierarchyFilter",
    ):
        """
        Initialize the hierarchy filter.
        
        Args:
            column: Name of the hierarchical column to filter on.
            include: Patterns to include. Rows with hierarchical values matching any of these patterns will be kept.
                Wildcards (*) can be used to match any sequence of characters.
                If None, all rows are included by default.
            exclude: Patterns to exclude. Rows with hierarchical values matching any of these patterns will be removed.
                Wildcards (*) can be used to match any sequence of characters.
                If None, no rows are excluded by default.
            separator: Separator used between hierarchy levels.
            case_sensitive: Whether pattern matching should be case-sensitive.
            name: Name of the transformer.
        """
        super().__init__([column], [], name)
        self.column = column
        self.include = include or ["*"]
        self.exclude = exclude or []
        self.separator = separator
        self.case_sensitive = case_sensitive
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filter rows based on hierarchical column values.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with filtered rows.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        X_filtered = X.copy()
        
        # Convert patterns to regular expressions
        import re
        
        def pattern_to_regex(pattern: str) -> str:
            # Escape special characters except *
            pattern = re.escape(pattern).replace("\\*", ".*")
            return f"^{pattern}$"
        
        include_regexes = [re.compile(pattern_to_regex(p), 0 if self.case_sensitive else re.IGNORECASE) for p in self.include]
        exclude_regexes = [re.compile(pattern_to_regex(p), 0 if self.case_sensitive else re.IGNORECASE) for p in self.exclude]
        
        # Create a mask for rows to keep
        include_mask = pd.Series(False, index=X_filtered.index)
        for regex in include_regexes:
            include_mask |= X_filtered[self.column].str.match(regex, na=False)
        
        # Create a mask for rows to exclude
        exclude_mask = pd.Series(False, index=X_filtered.index)
        for regex in exclude_regexes:
            exclude_mask |= X_filtered[self.column].str.match(regex, na=False)
        
        # Apply the filters
        return X_filtered[include_mask & ~exclude_mask]
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, return an empty DataFrame with the same columns.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Return an empty DataFrame with the same columns
        return X.iloc[0:0]


# Register transformers with the registry
register_transformer("hierarchy_builder", HierarchyBuilder)
register_transformer("hierarchy_expander", HierarchyExpander)
register_transformer("hierarchy_filter", HierarchyFilter)