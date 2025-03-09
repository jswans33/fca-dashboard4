"""
Feature Engineering Interfaces Module

This module defines the interfaces for feature engineering components in the NexusML suite.
Each interface follows the Interface Segregation Principle (ISP) from SOLID,
defining a minimal set of methods that components must implement.
"""

import abc
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureTransformer(abc.ABC):
    """
    Interface for feature transformers.
    
    A feature transformer is responsible for transforming raw data into features
    suitable for model training. It follows the scikit-learn transformer interface
    with fit, transform, and fit_transform methods.
    """
    
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer':
        """
        Fit the transformer to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the transformer cannot be fit to the data.
        """
        pass
    
    @abc.abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted transformer.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the transformer has not been fitted or the data cannot be transformed.
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the transformer to the data and transform it.
        
        This method is provided for convenience and follows the scikit-learn convention.
        The default implementation calls fit and then transform, but subclasses can
        override this method for efficiency.
        
        Args:
            X: Input DataFrame to fit and transform.
            y: Target values (optional).
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the transformer cannot be fit or the data cannot be transformed.
        """
        return self.fit(X, y).transform(X)
    
    @abc.abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features produced by this transformer.
        
        Returns:
            List of feature names.
        """
        pass


class ColumnTransformer(FeatureTransformer, abc.ABC):
    """
    Interface for column-specific transformers.
    
    A column transformer is a feature transformer that operates on specific columns
    in a DataFrame. It knows which columns it needs to transform and can handle
    missing columns gracefully.
    """
    
    @abc.abstractmethod
    def get_input_columns(self) -> List[str]:
        """
        Get the names of the input columns required by this transformer.
        
        Returns:
            List of input column names.
        """
        pass
    
    @abc.abstractmethod
    def get_output_columns(self) -> List[str]:
        """
        Get the names of the output columns produced by this transformer.
        
        Returns:
            List of output column names.
        """
        pass


class ConfigurableTransformer(FeatureTransformer, abc.ABC):
    """
    Interface for configurable transformers.
    
    A configurable transformer is a feature transformer that can be configured
    using a dictionary of parameters. This allows for dynamic configuration
    without changing the code.
    """
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this transformer.
        
        Returns:
            Dictionary containing the configuration.
        """
        pass
    
    @abc.abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration of this transformer.
        
        Args:
            config: Dictionary containing the configuration.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        pass


class TransformerRegistry(abc.ABC):
    """
    Interface for transformer registries.
    
    A transformer registry maintains a collection of transformers and provides
    methods for registering, retrieving, and creating transformers.
    """
    
    @abc.abstractmethod
    def register_transformer(self, name: str, transformer_class: type) -> None:
        """
        Register a transformer class with the registry.
        
        Args:
            name: Name to register the transformer under.
            transformer_class: Transformer class to register.
            
        Raises:
            ValueError: If the name is already registered or the class is not a transformer.
        """
        pass
    
    @abc.abstractmethod
    def get_transformer_class(self, name: str) -> type:
        """
        Get a transformer class from the registry.
        
        Args:
            name: Name of the transformer class to get.
            
        Returns:
            Transformer class.
            
        Raises:
            KeyError: If the name is not registered.
        """
        pass
    
    @abc.abstractmethod
    def create_transformer(self, name: str, **kwargs) -> FeatureTransformer:
        """
        Create a transformer instance from the registry.
        
        Args:
            name: Name of the transformer class to create.
            **kwargs: Arguments to pass to the transformer constructor.
            
        Returns:
            Transformer instance.
            
        Raises:
            KeyError: If the name is not registered.
            ValueError: If the transformer cannot be created with the given arguments.
        """
        pass
    
    @abc.abstractmethod
    def get_registered_transformers(self) -> Dict[str, type]:
        """
        Get all registered transformers.
        
        Returns:
            Dictionary mapping transformer names to transformer classes.
        """
        pass


class FeatureEngineer(abc.ABC):
    """
    Interface for feature engineers.
    
    A feature engineer is responsible for coordinating the application of multiple
    transformers to engineer features from raw data. It manages the transformer
    pipeline and provides methods for fitting and transforming data.
    """
    
    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If the feature engineer cannot be fit to the data.
        """
        pass
    
    @abc.abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the feature engineer has not been fitted or the data cannot be transformed.
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the feature engineer to the data and transform it.
        
        This method is provided for convenience and follows the scikit-learn convention.
        The default implementation calls fit and then transform, but subclasses can
        override this method for efficiency.
        
        Args:
            X: Input DataFrame to fit and transform.
            y: Target values (optional).
            
        Returns:
            Transformed DataFrame.
            
        Raises:
            ValueError: If the feature engineer cannot be fit or the data cannot be transformed.
        """
        return self.fit(X, y).transform(X)
    
    @abc.abstractmethod
    def get_transformers(self) -> List[FeatureTransformer]:
        """
        Get the transformers used by this feature engineer.
        
        Returns:
            List of transformers.
        """
        pass
    
    @abc.abstractmethod
    def add_transformer(self, transformer: FeatureTransformer) -> None:
        """
        Add a transformer to this feature engineer.
        
        Args:
            transformer: Transformer to add.
            
        Raises:
            ValueError: If the transformer is not compatible with this feature engineer.
        """
        pass
    
    @abc.abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features produced by this feature engineer.
        
        Returns:
            List of feature names.
        """
        pass


class ConfigDrivenFeatureEngineer(FeatureEngineer, abc.ABC):
    """
    Interface for configuration-driven feature engineers.
    
    A configuration-driven feature engineer is a feature engineer that can be configured
    using a dictionary of parameters. This allows for dynamic configuration without
    changing the code.
    """
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this feature engineer.
        
        Returns:
            Dictionary containing the configuration.
        """
        pass
    
    @abc.abstractmethod
    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration of this feature engineer.
        
        Args:
            config: Dictionary containing the configuration.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        pass
    
    @abc.abstractmethod
    def create_transformers_from_config(self) -> List[FeatureTransformer]:
        """
        Create transformers from the configuration.
        
        Returns:
            List of transformers created from the configuration.
            
        Raises:
            ValueError: If the configuration is invalid or transformers cannot be created.
        """
        pass