"""
Gradient Boosting Model Builder Module

This module provides a GradientBoostingBuilder implementation that builds
Gradient Boosting models for classification tasks.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.model_building.base import BaseConfigurableModelBuilder

# Set up logging
logger = logging.getLogger(__name__)


@injectable
class GradientBoostingBuilder(BaseConfigurableModelBuilder):
    """
    Implementation of the ModelBuilder interface for Gradient Boosting models.
    
    This class builds Gradient Boosting models based on configuration provided by the
    ConfigurationProvider. It supports both text and numeric features and provides
    hyperparameter optimization.
    """
    
    def __init__(
        self,
        name: str = "GradientBoostingBuilder",
        description: str = "Gradient Boosting model builder using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the GradientBoostingBuilder.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        logger.info(f"Initialized {name}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the Gradient Boosting model.
        
        Returns:
            Dictionary of default parameters.
        """
        return {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
                "use_idf": True,
                "sublinear_tf": True,
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "subsample": 1.0,
                "random_state": 42,
            },
        }
    
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Get the parameter grid for hyperparameter optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of values to try.
        """
        return {
            "preprocessor__text__tfidf__max_features": [3000, 5000, 7000],
            "preprocessor__text__tfidf__ngram_range": [(1, 2), (1, 3)],
            "clf__estimator__n_estimators": [50, 100, 200],
            "clf__estimator__learning_rate": [0.01, 0.1, 0.2],
            "clf__estimator__max_depth": [3, 5, 7],
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the model builder configuration.
        
        Args:
            config: Configuration to validate.
            
        Returns:
            True if the configuration is valid, False otherwise.
        """
        # Check if the required sections exist
        if "tfidf" not in config:
            logger.warning("Missing 'tfidf' section in configuration")
            return False
        
        if "gradient_boosting" not in config:
            logger.warning("Missing 'gradient_boosting' section in configuration")
            return False
        
        # Check if the required parameters exist in the tfidf section
        tfidf_required_params = ["max_features", "ngram_range", "min_df", "max_df"]
        for param in tfidf_required_params:
            if param not in config["tfidf"]:
                logger.warning(f"Missing '{param}' parameter in 'tfidf' section")
                return False
        
        # Check if the required parameters exist in the gradient_boosting section
        gb_required_params = ["n_estimators", "learning_rate", "max_depth", "random_state"]
        for param in gb_required_params:
            if param not in config["gradient_boosting"]:
                logger.warning(f"Missing '{param}' parameter in 'gradient_boosting' section")
                return False
        
        return True
    
    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a Gradient Boosting model.
        
        This method creates a pipeline with a preprocessor for text and numeric features
        and a Gradient Boosting classifier.
        
        Args:
            **kwargs: Configuration parameters for the model. These override the
                    configuration from the provider.
            
        Returns:
            Configured model pipeline.
            
        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        try:
            logger.info("Building Gradient Boosting model")
            
            # Update config with kwargs
            if kwargs:
                for key, value in kwargs.items():
                    if key in self.config:
                        self.config[key] = value
                        logger.debug(f"Updated config parameter {key} with value {value}")
            
            # Extract TF-IDF settings
            tfidf_settings = self.config.get("tfidf", {})
            max_features = tfidf_settings.get("max_features", 5000)
            ngram_range = tuple(tfidf_settings.get("ngram_range", [1, 3]))
            min_df = tfidf_settings.get("min_df", 2)
            max_df = tfidf_settings.get("max_df", 0.9)
            use_idf = tfidf_settings.get("use_idf", True)
            sublinear_tf = tfidf_settings.get("sublinear_tf", True)
            
            # Extract Gradient Boosting settings
            gb_settings = self.config.get("gradient_boosting", {})
            n_estimators = gb_settings.get("n_estimators", 100)
            learning_rate = gb_settings.get("learning_rate", 0.1)
            max_depth = gb_settings.get("max_depth", 3)
            min_samples_split = gb_settings.get("min_samples_split", 2)
            min_samples_leaf = gb_settings.get("min_samples_leaf", 1)
            subsample = gb_settings.get("subsample", 1.0)
            random_state = gb_settings.get("random_state", 42)
            
            # Text feature processing
            text_features = Pipeline(
                [
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=max_features,
                            ngram_range=ngram_range,
                            min_df=min_df,
                            max_df=max_df,
                            use_idf=use_idf,
                            sublinear_tf=sublinear_tf,
                        ),
                    )
                ]
            )
            
            # Numeric feature processing
            numeric_features = Pipeline([("scaler", StandardScaler())])
            
            # Combine text and numeric features
            preprocessor = ColumnTransformer(
                transformers=[
                    ("text", text_features, "combined_text"),
                    ("numeric", numeric_features, ["service_life"]),
                ],
                remainder="drop",
            )
            
            # Complete pipeline with feature processing and classifier
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "clf",
                        MultiOutputClassifier(
                            GradientBoostingClassifier(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                subsample=subsample,
                                random_state=random_state,
                            )
                        ),
                    ),
                ]
            )
            
            logger.info("Gradient Boosting model built successfully")
            return pipeline
        
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ValueError(f"Error building model: {str(e)}") from e