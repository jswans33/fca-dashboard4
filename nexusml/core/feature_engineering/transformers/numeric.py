"""
Numeric Transformers Module

This module provides transformers for numeric features in the NexusML suite.
Each transformer follows the Single Responsibility Principle (SRP) from SOLID,
focusing on a specific numeric transformation.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nexusml.core.feature_engineering.base import BaseColumnTransformer, BaseConfigurableTransformer
from nexusml.core.feature_engineering.registry import register_transformer


class NumericCleaner(BaseColumnTransformer):
    """
    Cleans and transforms numeric columns.
    
    This transformer handles missing values, converts to the specified data type,
    and applies basic cleaning to numeric columns.
    
    Config example: {"name": "Service Life", "new_name": "service_life", "fill_value": 0, "dtype": "float"}
    """
    
    def __init__(
        self,
        column: str,
        new_name: Optional[str] = None,
        fill_value: Union[int, float] = 0,
        dtype: str = "float",
        name: str = "NumericCleaner",
    ):
        """
        Initialize the numeric cleaner.
        
        Args:
            column: Name of the column to clean.
            new_name: Name of the new column to create. If None, uses the input column name.
            fill_value: Value to use for filling missing values.
            dtype: Data type to convert the column to. One of: "float", "int".
            name: Name of the transformer.
        """
        output_column = new_name or column
        super().__init__([column], [output_column], name)
        self.column = column
        self.new_name = output_column
        self.fill_value = fill_value
        self.dtype = dtype
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform the numeric column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the cleaned numeric column.
        """
        # Clean and transform the numeric column
        if self.dtype == "float":
            X[self.new_name] = pd.to_numeric(X[self.column], errors="coerce").fillna(self.fill_value).astype(float)
        elif self.dtype == "int":
            X[self.new_name] = pd.to_numeric(X[self.column], errors="coerce").fillna(self.fill_value).astype(int)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}. Must be one of: 'float', 'int'")
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, create a new column with the default value.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
        """
        # Create a new column with the default value
        if self.dtype == "float":
            X[self.new_name] = float(self.fill_value)
        elif self.dtype == "int":
            X[self.new_name] = int(self.fill_value)
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}. Must be one of: 'float', 'int'")
        
        return X


class NumericScaler(BaseColumnTransformer):
    """
    Scales numeric columns.
    
    This transformer applies scaling to numeric columns using various scaling methods.
    
    Config example: {
        "column": "age",
        "new_name": "scaled_age",
        "method": "standard",
        "with_mean": true,
        "with_std": true
    }
    """
    
    def __init__(
        self,
        column: str,
        new_name: Optional[str] = None,
        method: str = "standard",
        with_mean: bool = True,
        with_std: bool = True,
        feature_range: Tuple[float, float] = (0, 1),
        name: str = "NumericScaler",
    ):
        """
        Initialize the numeric scaler.
        
        Args:
            column: Name of the column to scale.
            new_name: Name of the new column to create. If None, uses the input column name.
            method: Scaling method to use. One of: "standard", "minmax".
            with_mean: Whether to center the data before scaling (for standard scaling).
            with_std: Whether to scale the data to unit variance (for standard scaling).
            feature_range: Range of the scaled data (for minmax scaling).
            name: Name of the transformer.
        """
        output_column = new_name or column
        super().__init__([column], [output_column], name)
        self.column = column
        self.new_name = output_column
        self.method = method
        self.with_mean = with_mean
        self.with_std = with_std
        self.feature_range = feature_range
        self.scaler = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'NumericScaler':
        """
        Fit the scaler to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the input column is missing or the method is unsupported.
        """
        # Check if the input column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame")
        
        # Create the appropriate scaler
        if self.method == "standard":
            self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        elif self.method == "minmax":
            self.scaler = MinMaxScaler(feature_range=self.feature_range)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}. Must be one of: 'standard', 'minmax'")
        
        # Fit the scaler
        values = X[self.column].values.reshape(-1, 1)
        self.scaler.fit(values)
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the numeric column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with the scaled numeric column.
        """
        # Scale the column
        values = X[self.column].values.reshape(-1, 1)
        scaled_values = self.scaler.transform(values)
        X[self.new_name] = scaled_values.flatten()
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, raise a ValueError since we can't scale without data.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the input column is missing.
        """
        raise ValueError(f"Column '{self.column}' not found in input DataFrame. Cannot scale without data.")


class MissingValueHandler(BaseColumnTransformer):
    """
    Handles missing values in numeric columns.
    
    This transformer provides various strategies for handling missing values in numeric columns.
    
    Config example: {
        "column": "age",
        "strategy": "mean",
        "fill_value": 0
    }
    """
    
    def __init__(
        self,
        column: str,
        strategy: str = "mean",
        fill_value: Optional[Union[int, float]] = None,
        name: str = "MissingValueHandler",
    ):
        """
        Initialize the missing value handler.
        
        Args:
            column: Name of the column to handle missing values in.
            strategy: Strategy to use for handling missing values.
                One of: "mean", "median", "mode", "constant", "forward_fill", "backward_fill".
            fill_value: Value to use for filling missing values when strategy is "constant".
            name: Name of the transformer.
        """
        super().__init__([column], [column], name)
        self.column = column
        self.strategy = strategy
        self.fill_value = fill_value
        self.fill_value_computed = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MissingValueHandler':
        """
        Fit the missing value handler to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the input column is missing or the strategy is unsupported.
        """
        # Check if the input column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame")
        
        # Compute the fill value based on the strategy
        if self.strategy == "mean":
            self.fill_value_computed = X[self.column].mean()
        elif self.strategy == "median":
            self.fill_value_computed = X[self.column].median()
        elif self.strategy == "mode":
            self.fill_value_computed = X[self.column].mode().iloc[0]
        elif self.strategy == "constant":
            if self.fill_value is None:
                raise ValueError("fill_value must be provided when strategy is 'constant'")
            self.fill_value_computed = self.fill_value
        elif self.strategy in ["forward_fill", "backward_fill"]:
            # No need to compute a fill value for these strategies
            pass
        else:
            raise ValueError(
                f"Unsupported strategy: {self.strategy}. "
                "Must be one of: 'mean', 'median', 'mode', 'constant', 'forward_fill', 'backward_fill'"
            )
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the numeric column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with missing values handled.
        """
        # Handle missing values based on the strategy
        if self.strategy in ["mean", "median", "mode", "constant"]:
            X[self.column] = X[self.column].fillna(self.fill_value_computed)
        elif self.strategy == "forward_fill":
            X[self.column] = X[self.column].ffill()
        elif self.strategy == "backward_fill":
            X[self.column] = X[self.column].bfill()
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, raise a ValueError since we can't handle missing values without data.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the input column is missing.
        """
        raise ValueError(f"Column '{self.column}' not found in input DataFrame. Cannot handle missing values without data.")


class OutlierDetector(BaseColumnTransformer):
    """
    Detects and handles outliers in numeric columns.
    
    This transformer provides various methods for detecting and handling outliers in numeric columns.
    
    Config example: {
        "column": "age",
        "method": "zscore",
        "threshold": 3.0,
        "strategy": "clip"
    }
    """
    
    def __init__(
        self,
        column: str,
        method: str = "zscore",
        threshold: float = 3.0,
        strategy: str = "clip",
        name: str = "OutlierDetector",
    ):
        """
        Initialize the outlier detector.
        
        Args:
            column: Name of the column to detect outliers in.
            method: Method to use for detecting outliers. One of: "zscore", "iqr".
            threshold: Threshold for outlier detection. For zscore, values with absolute z-score
                greater than this are considered outliers. For IQR, values outside
                Q1 - threshold * IQR and Q3 + threshold * IQR are considered outliers.
            strategy: Strategy to use for handling outliers. One of: "clip", "remove", "flag".
            name: Name of the transformer.
        """
        output_columns = [column]
        if strategy == "flag":
            output_columns.append(f"{column}_is_outlier")
        
        super().__init__([column], output_columns, name)
        self.column = column
        self.method = method
        self.threshold = threshold
        self.strategy = strategy
        self.stats = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'OutlierDetector':
        """
        Fit the outlier detector to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the input column is missing or the method/strategy is unsupported.
        """
        # Check if the input column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame")
        
        # Validate method and strategy
        if self.method not in ["zscore", "iqr"]:
            raise ValueError(f"Unsupported method: {self.method}. Must be one of: 'zscore', 'iqr'")
        
        if self.strategy not in ["clip", "remove", "flag"]:
            raise ValueError(f"Unsupported strategy: {self.strategy}. Must be one of: 'clip', 'remove', 'flag'")
        
        # Compute statistics for outlier detection
        if self.method == "zscore":
            self.stats["mean"] = X[self.column].mean()
            self.stats["std"] = X[self.column].std()
        elif self.method == "iqr":
            q1 = X[self.column].quantile(0.25)
            q3 = X[self.column].quantile(0.75)
            iqr = q3 - q1
            self.stats["q1"] = q1
            self.stats["q3"] = q3
            self.stats["iqr"] = iqr
            self.stats["lower_bound"] = q1 - self.threshold * iqr
            self.stats["upper_bound"] = q3 + self.threshold * iqr
        
        self._is_fitted = True
        return self
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers in the numeric column.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with outliers handled.
        """
        # Detect outliers
        if self.method == "zscore":
            z_scores = (X[self.column] - self.stats["mean"]) / self.stats["std"]
            is_outlier = abs(z_scores) > self.threshold
        elif self.method == "iqr":
            is_outlier = (X[self.column] < self.stats["lower_bound"]) | (X[self.column] > self.stats["upper_bound"])
        
        # Handle outliers based on the strategy
        if self.strategy == "clip":
            if self.method == "zscore":
                lower_bound = self.stats["mean"] - self.threshold * self.stats["std"]
                upper_bound = self.stats["mean"] + self.threshold * self.stats["std"]
            else:  # iqr
                lower_bound = self.stats["lower_bound"]
                upper_bound = self.stats["upper_bound"]
            
            X[self.column] = X[self.column].clip(lower=lower_bound, upper=upper_bound)
        
        elif self.strategy == "remove":
            # Create a copy of the column with outliers set to NaN
            X[self.column] = X[self.column].mask(is_outlier)
        
        elif self.strategy == "flag":
            # Add a new column indicating whether each value is an outlier
            X[f"{self.column}_is_outlier"] = is_outlier
        
        return X
    
    def _handle_missing_columns(self, X: pd.DataFrame, missing_columns: List[str]) -> pd.DataFrame:
        """
        Handle missing input columns.
        
        If the input column is missing, raise a ValueError since we can't detect outliers without data.
        
        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the input column is missing.
        """
        raise ValueError(f"Column '{self.column}' not found in input DataFrame. Cannot detect outliers without data.")


# Register transformers with the registry
register_transformer("numeric_cleaner", NumericCleaner)
register_transformer("numeric_scaler", NumericScaler)
register_transformer("missing_value_handler", MissingValueHandler)
register_transformer("outlier_detector", OutlierDetector)