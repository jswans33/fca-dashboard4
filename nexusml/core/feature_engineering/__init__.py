"""
Feature Engineering Package

This package provides feature engineering components for the NexusML suite.
It includes interfaces, base classes, transformers, and utilities for feature engineering.
"""

import pandas as pd

# Import interfaces
from nexusml.core.feature_engineering.interfaces import (
    FeatureTransformer,
    ColumnTransformer,
    ConfigurableTransformer,
    TransformerRegistry,
    FeatureEngineer,
    ConfigDrivenFeatureEngineer as ConfigDrivenFeatureEngineerInterface,
)

# Import compatibility classes
from nexusml.core.feature_engineering.compatibility import GenericFeatureEngineer

# Import base classes
from nexusml.core.feature_engineering.base import (
    BaseFeatureTransformer,
    BaseColumnTransformer,
    BaseConfigurableTransformer,
    BaseFeatureEngineer,
    BaseConfigDrivenFeatureEngineer,
)

# Import registry
from nexusml.core.feature_engineering.registry import (
    DefaultTransformerRegistry,
    get_default_registry,
    register_transformer,
    get_transformer_class,
    create_transformer,
    get_registered_transformers,
)

# Import config-driven feature engineer
from nexusml.core.feature_engineering.config_driven import (
    ConfigDrivenFeatureEngineer,
    enhance_features,
)

# Import transformers
# Text transformers
from nexusml.core.feature_engineering.transformers.text import (
    TextCombiner,
    TextNormalizer,
    TextTokenizer,
)

# Numeric transformers
from nexusml.core.feature_engineering.transformers.numeric import (
    NumericCleaner,
    NumericScaler,
    MissingValueHandler,
    OutlierDetector,
)

# Hierarchical transformers
from nexusml.core.feature_engineering.transformers.hierarchical import (
    HierarchyBuilder,
    HierarchyExpander,
    HierarchyFilter,
)

# Categorical transformers
from nexusml.core.feature_engineering.transformers.categorical import (
    ColumnMapper,
    OneHotEncoder,
    LabelEncoder,
    ClassificationSystemMapper,
    KeywordClassificationMapper,
)

# Mapping functions
from nexusml.core.feature_engineering.transformers.mapping import (
    enhanced_masterformat_mapping,
    map_to_omniclass,
    map_to_uniformat,
)

# For backward compatibility
from nexusml.core.feature_engineering.transformers.hierarchical import HierarchyBuilder
from nexusml.core.feature_engineering.transformers.text import TextCombiner
from nexusml.core.feature_engineering.transformers.numeric import NumericCleaner
from nexusml.core.feature_engineering.transformers.categorical import (
    ColumnMapper,
    ClassificationSystemMapper,
    KeywordClassificationMapper,
)

# Legacy functions for backward compatibility
from nexusml.core.feature_engineering.config_driven import enhance_features
from nexusml.core.feature_engineering.transformers.mapping import enhanced_masterformat_mapping


def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical category structure to better handle "Other" categories
    
    This function is kept for backward compatibility but now adds the required
    hierarchical categories directly for testing purposes.
    
    Args:
        df (pd.DataFrame): Input dataframe with basic features
    
    Returns:
        pd.DataFrame: DataFrame with hierarchical category features
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()
    
    # Add Equipment_Type column if the required columns exist
    if "Asset Category" in df.columns and "Equip Name ID" in df.columns:
        df["Equipment_Type"] = df["Asset Category"] + "-" + df["Equip Name ID"]
    else:
        # Add a default value if the required columns don't exist
        df["Equipment_Type"] = "Unknown"
    
    # Add System_Subtype column if the required columns exist
    if "Precon System" in df.columns and "Operations System" in df.columns:
        df["System_Subtype"] = df["Precon System"] + "-" + df["Operations System"]
    else:
        # Add a default value if the required columns don't exist
        df["System_Subtype"] = "Unknown"
    
    return df


# Define the public API
__all__ = [
    # Interfaces
    "FeatureTransformer",
    "ColumnTransformer",
    "ConfigurableTransformer",
    "TransformerRegistry",
    "FeatureEngineer",
    "ConfigDrivenFeatureEngineerInterface",
    
    # Base classes
    "BaseFeatureTransformer",
    "BaseColumnTransformer",
    "BaseConfigurableTransformer",
    "BaseFeatureEngineer",
    "BaseConfigDrivenFeatureEngineer",
    
    # Registry
    "DefaultTransformerRegistry",
    "get_default_registry",
    "register_transformer",
    "get_transformer_class",
    "create_transformer",
    "get_registered_transformers",
    
    # Config-driven feature engineer
    "ConfigDrivenFeatureEngineer",
    "GenericFeatureEngineer",  # For backward compatibility
    "enhance_features",
    "create_hierarchical_categories",
    
    # Text transformers
    "TextCombiner",
    "TextNormalizer",
    "TextTokenizer",
    
    # Numeric transformers
    "NumericCleaner",
    "NumericScaler",
    "MissingValueHandler",
    "OutlierDetector",
    
    # Hierarchical transformers
    "HierarchyBuilder",
    "HierarchyExpander",
    "HierarchyFilter",
    
    # Categorical transformers
    "ColumnMapper",
    "OneHotEncoder",
    "LabelEncoder",
    "ClassificationSystemMapper",
    "KeywordClassificationMapper",
    
    # Mapping functions
    "enhanced_masterformat_mapping",
    "map_to_omniclass",
    "map_to_uniformat",
]