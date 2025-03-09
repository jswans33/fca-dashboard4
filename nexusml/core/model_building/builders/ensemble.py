"""
Ensemble Model Builder Module

This module provides an EnsembleBuilder implementation that builds
ensemble models combining multiple base classifiers for classification tasks.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
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
class EnsembleBuilder(BaseConfigurableModelBuilder):
    """
    Implementation of the ModelBuilder interface for Ensemble models.
    
    This class builds ensemble models that combine multiple base classifiers
    based on configuration provided by the ConfigurationProvider.
    It supports both text and numeric features and provides hyperparameter optimization.
    """
    
    def __init__(
        self,
        name: str = "EnsembleBuilder",
        description: str = "Ensemble model builder using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the EnsembleBuilder.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        logger.info(f"Initialized {name}")
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for the Ensemble model.
        
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
            "ensemble": {
                "voting": "soft",
                "weights": [1, 1, 1],  # Equal weights for all classifiers
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": "balanced_subsample",
                "random_state": 42,
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
            "clf__estimator__weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
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
        
        if "ensemble" not in config:
            logger.warning("Missing 'ensemble' section in configuration")
            return False
        
        if "random_forest" not in config:
            logger.warning("Missing 'random_forest' section in configuration")
            return False
        
        if "gradient_boosting" not in config:
            logger.warning("Missing 'gradient_boosting' section in configuration")
            return False
        
        # Check if the required parameters exist in the ensemble section
        ensemble_required_params = ["voting", "weights", "random_state"]
        for param in ensemble_required_params:
            if param not in config["ensemble"]:
                logger.warning(f"Missing '{param}' parameter in 'ensemble' section")
                return False
        
        return True
    
    def build_model(self, **kwargs) -> Pipeline:
        """
        Build an Ensemble model.
        
        This method creates a pipeline with a preprocessor for text and numeric features
        and an ensemble classifier that combines multiple base classifiers.
        
        Args:
            **kwargs: Configuration parameters for the model. These override the
                    configuration from the provider.
            
        Returns:
            Configured model pipeline.
            
        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        try:
            logger.info("Building Ensemble model")
            
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
            
            # Extract Ensemble settings
            ensemble_settings = self.config.get("ensemble", {})
            voting = ensemble_settings.get("voting", "soft")
            weights = ensemble_settings.get("weights", [1, 1, 1])
            random_state = ensemble_settings.get("random_state", 42)
            
            # Extract Random Forest settings
            rf_settings = self.config.get("random_forest", {})
            rf_n_estimators = rf_settings.get("n_estimators", 200)
            rf_max_depth = rf_settings.get("max_depth", None)
            rf_min_samples_split = rf_settings.get("min_samples_split", 2)
            rf_min_samples_leaf = rf_settings.get("min_samples_leaf", 1)
            rf_class_weight = rf_settings.get("class_weight", "balanced_subsample")
            rf_random_state = rf_settings.get("random_state", 42)
            
            # Extract Gradient Boosting settings
            gb_settings = self.config.get("gradient_boosting", {})
            gb_n_estimators = gb_settings.get("n_estimators", 100)
            gb_learning_rate = gb_settings.get("learning_rate", 0.1)
            gb_max_depth = gb_settings.get("max_depth", 3)
            gb_min_samples_split = gb_settings.get("min_samples_split", 2)
            gb_min_samples_leaf = gb_settings.get("min_samples_leaf", 1)
            gb_subsample = gb_settings.get("subsample", 1.0)
            gb_random_state = gb_settings.get("random_state", 42)
            
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
                    ("text", text_features, "combined_features"),
                    ("numeric", numeric_features, ["service_life"]),
                ],
                remainder="drop",
            )
            
            # Create base classifiers
            rf_classifier = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_split=rf_min_samples_split,
                min_samples_leaf=rf_min_samples_leaf,
                class_weight=rf_class_weight,
                random_state=rf_random_state,
            )
            
            gb_classifier = GradientBoostingClassifier(
                n_estimators=gb_n_estimators,
                learning_rate=gb_learning_rate,
                max_depth=gb_max_depth,
                min_samples_split=gb_min_samples_split,
                min_samples_leaf=gb_min_samples_leaf,
                subsample=gb_subsample,
                random_state=gb_random_state,
            )
            
            # Create a voting classifier that combines the base classifiers
            voting_classifier = VotingClassifier(
                estimators=[
                    ("rf", rf_classifier),
                    ("gb", gb_classifier),
                ],
                voting=voting,
                weights=weights[:2],  # Use only the first two weights
            )
            
            # Complete pipeline with feature processing and classifier
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "clf",
                        MultiOutputClassifier(voting_classifier),
                    ),
                ]
            )
            
            logger.info("Ensemble model built successfully")
            return pipeline
        
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ValueError(f"Error building model: {str(e)}") from e