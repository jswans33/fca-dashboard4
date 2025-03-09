"""
Type definitions for the feature engineering interfaces.

This module provides type hints for the feature engineering interfaces to improve type safety.
"""

import abc
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Type, TypeVar, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Type variable for generic types
T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


class FeatureTransformer(Protocol):
    """
    Interface for feature transformers.
    
    A feature transformer is responsible for transforming raw data into features
    suitable for model training. It follows the scikit-learn transformer interface
    with fit, transform, and fit_transform methods.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer': ...
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame: ...
    
    def get_feature_names(self) -> List[str]: ...


class ColumnTransformer(FeatureTransformer, Protocol):
    """
    Interface for column-specific transformers.
    
    A column transformer is a feature transformer that operates on specific columns
    in a DataFrame. It knows which columns it needs to transform and can handle
    missing columns gracefully.
    """
    
    def get_input_columns(self) -> List[str]: ...
    
    def get_output_columns(self) -> List[str]: ...


class ConfigurableTransformer(FeatureTransformer, Protocol):
    """
    Interface for configurable transformers.
    
    A configurable transformer is a feature transformer that can be configured
    using a dictionary of parameters. This allows for dynamic configuration
    without changing the code.
    """
    
    def get_config(self) -> Dict[str, Any]: ...
    
    def set_config(self, config: Dict[str, Any]) -> None: ...


class TransformerRegistry(Protocol):
    """
    Interface for transformer registries.
    
    A transformer registry maintains a collection of transformers and provides
    methods for registering, retrieving, and creating transformers.
    """
    
    def register_transformer(self, name: str, transformer_class: Type[FeatureTransformer]) -> None: ...
    
    def get_transformer_class(self, name: str) -> Type[FeatureTransformer]: ...
    
    def create_transformer(self, name: str, **kwargs: Any) -> FeatureTransformer: ...
    
    def get_registered_transformers(self) -> Dict[str, Type[FeatureTransformer]]: ...


class FeatureEngineer(Protocol):
    """
    Interface for feature engineers.
    
    A feature engineer is responsible for coordinating the application of multiple
    transformers to engineer features from raw data. It manages the transformer
    pipeline and provides methods for fitting and transforming data.
    """
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer': ...
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame: ...
    
    def get_transformers(self) -> List[FeatureTransformer]: ...
    
    def add_transformer(self, transformer: FeatureTransformer) -> None: ...
    
    def get_feature_names(self) -> List[str]: ...


class ConfigDrivenFeatureEngineer(FeatureEngineer, Protocol):
    """
    Interface for configuration-driven feature engineers.
    
    A configuration-driven feature engineer is a feature engineer that can be configured
    using a dictionary of parameters. This allows for dynamic configuration without
    changing the code.
    """
    
    def get_config(self) -> Dict[str, Any]: ...
    
    def set_config(self, config: Dict[str, Any]) -> None: ...
    
    def create_transformers_from_config(self) -> List[FeatureTransformer]: ...