"""
Compatibility Module

This module provides backward compatibility with the old feature engineering API.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.config import get_project_root
from nexusml.core.feature_engineering.config_driven import ConfigDrivenFeatureEngineer


class GenericFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A generic feature engineering transformer that applies multiple transformations
    based on a configuration file.
    
    This class is provided for backward compatibility with the old API.
    New code should use ConfigDrivenFeatureEngineer instead.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        eav_manager: Optional[Any] = None,
    ):
        """
        Initialize the transformer with a configuration file path.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses the default path.
            eav_manager: EAVManager instance. If None, uses the one from the DI container.
        """
        self.config_path = config_path
        self.eav_manager = eav_manager
        
        # Create a ConfigDrivenFeatureEngineer instance
        self.feature_engineer = ConfigDrivenFeatureEngineer(
            config_path=config_path,
            name="GenericFeatureEngineer",
        )
    
    def fit(self, X, y=None):
        """
        Fit the transformer to the data.
        
        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).
            
        Returns:
            Self for method chaining.
        """
        self.feature_engineer.fit(X, y)
        return self
    
    def transform(self, X):
        """
        Transform the input data using the fitted transformer.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame.
        """
        return self.feature_engineer.transform(X)