"""
Configuration-Driven Feature Engineering Module

This module provides a configuration-driven approach to feature engineering in the NexusML suite.
It allows for dynamic creation of transformers based on a configuration file or dictionary.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd

from nexusml.config import get_project_root
from nexusml.core.feature_engineering.base import BaseConfigDrivenFeatureEngineer
from nexusml.core.feature_engineering.interfaces import FeatureTransformer
from nexusml.core.feature_engineering.registry import (
    create_transformer,
    get_registered_transformers,
    register_transformer,
)


class ConfigDrivenFeatureEngineer(BaseConfigDrivenFeatureEngineer):
    """
    A feature engineer that creates and applies transformers based on a configuration.
    
    This class uses the transformer registry to create transformers from a configuration
    file or dictionary, and applies them in sequence to transform the input data.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        name: str = "ConfigDrivenFeatureEngineer",
    ):
        """
        Initialize the configuration-driven feature engineer.
        
        Args:
            config_path: Path to the YAML configuration file. If None, uses the default path.
            config: Configuration dictionary. If provided, overrides config_path.
            name: Name of the feature engineer.
        """
        self.config_path = config_path
        
        # Load the configuration from the file if provided
        if config is None and config_path is not None:
            config = self._load_config_from_file(config_path)
        elif config is None:
            # Use default path
            root = get_project_root()
            default_config_path = root / "config" / "feature_config.yml"
            config = self._load_config_from_file(default_config_path)
        
        # Initialize the base class
        super().__init__(config, name)
    
    def _load_config_from_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load the configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            Configuration dictionary.
            
        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If the configuration file is not valid YAML.
        """
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration.
        
        Args:
            config: Configuration to validate.
            
        Raises:
            ValueError: If the configuration is invalid.
        """
        # Check if the configuration is a dictionary
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Check if the configuration contains any of the expected sections
        expected_sections = [
            "column_mappings",
            "text_combinations",
            "numeric_columns",
            "hierarchies",
            "keyword_classifications",
            "classification_systems",
            "eav_integration",
            "transformers",
        ]
        
        if not any(section in config for section in expected_sections):
            raise ValueError(
                f"Configuration must contain at least one of the following sections: {expected_sections}"
            )
    
    def create_transformers_from_config(self) -> List[FeatureTransformer]:
        """
        Create transformers from the configuration.
        
        Returns:
            List of transformers created from the configuration.
            
        Raises:
            ValueError: If the configuration is invalid or transformers cannot be created.
        """
        transformers = []
        
        # Create transformers from the "transformers" section if it exists
        if "transformers" in self.config:
            for transformer_config in self.config["transformers"]:
                # Get the transformer type
                transformer_type = transformer_config.pop("type")
                
                # Create the transformer
                transformer = create_transformer(transformer_type, **transformer_config)
                
                # Add the transformer to the list
                transformers.append(transformer)
        
        # Create transformers from the legacy sections for backward compatibility
        
        # 1. Column mappings
        if "column_mappings" in self.config:
            transformer = create_transformer("column_mapper", mappings=self.config["column_mappings"])
            transformers.append(transformer)
        
        # 2. Text combinations
        if "text_combinations" in self.config:
            for combo in self.config["text_combinations"]:
                transformer = create_transformer(
                    "text_combiner",
                    columns=combo["columns"],
                    separator=combo.get("separator", " "),
                    new_column=combo.get("name", "combined_text"),
                )
                transformers.append(transformer)
        
        # 3. Numeric columns
        if "numeric_columns" in self.config:
            for num_col in self.config["numeric_columns"]:
                transformer = create_transformer(
                    "numeric_cleaner",
                    column=num_col["name"],
                    new_name=num_col.get("new_name", num_col["name"]),
                    fill_value=num_col.get("fill_value", 0),
                    dtype=num_col.get("dtype", "float"),
                )
                transformers.append(transformer)
        
        # 4. Hierarchies
        if "hierarchies" in self.config:
            for hierarchy in self.config["hierarchies"]:
                transformer = create_transformer(
                    "hierarchy_builder",
                    parent_columns=hierarchy["parents"],
                    new_column=hierarchy["new_col"],
                    separator=hierarchy.get("separator", "-"),
                )
                transformers.append(transformer)
        
        # 5. Keyword classifications
        if "keyword_classifications" in self.config:
            for system in self.config["keyword_classifications"]:
                transformer = create_transformer(
                    "keyword_classification_mapper",
                    name=system["name"],
                    source_column=system["source_column"],
                    target_column=system["target_column"],
                    reference_manager=system.get("reference_manager", "uniformat_keywords"),
                    max_results=system.get("max_results", 1),
                    confidence_threshold=system.get("confidence_threshold", 0.0),
                )
                transformers.append(transformer)
        
        # 6. Classification systems
        if "classification_systems" in self.config:
            for system in self.config["classification_systems"]:
                transformer = create_transformer(
                    "classification_system_mapper",
                    name=system["name"],
                    source_column=system.get("source_column") or system.get("source_columns", []),
                    target_column=system["target_column"],
                    mapping_type=system.get("mapping_type", "eav"),
                    mapping_function=system.get("mapping_function"),
                )
                transformers.append(transformer)
        
        # 7. EAV integration
        if "eav_integration" in self.config and self.config["eav_integration"].get("enabled", False):
            from nexusml.core.eav_manager import EAVTransformer
            
            transformer = EAVTransformer()
            transformers.append(transformer)
        
        return transformers


def enhance_features(
    df: pd.DataFrame, feature_engineer: Optional[ConfigDrivenFeatureEngineer] = None
) -> pd.DataFrame:
    """
    Enhanced feature engineering with hierarchical structure and more granular categories.
    
    This function uses the ConfigDrivenFeatureEngineer to apply transformations
    based on the configuration file.
    
    Args:
        df: Input dataframe with raw features.
        feature_engineer: Feature engineer instance. If None, creates a new one.
    
    Returns:
        DataFrame with enhanced features.
    """
    # Create a feature engineer if not provided
    if feature_engineer is None:
        try:
            # Try to get the feature engineer from the DI container
            from nexusml.core.di.provider import ContainerProvider
            
            container = ContainerProvider().container
            feature_engineer = container.resolve(ConfigDrivenFeatureEngineer)
        except Exception:
            # Create a new feature engineer if not available in the container
            feature_engineer = ConfigDrivenFeatureEngineer()
    
    # Apply transformations
    return feature_engineer.transform(df)