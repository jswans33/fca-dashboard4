"""
Model Builder Component

This module provides a standard implementation of the ModelBuilder interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseModelBuilder

# Set up logging
logger = logging.getLogger(__name__)


class RandomForestModelBuilder(BaseModelBuilder):
    """
    Implementation of the ModelBuilder interface for Random Forest models.

    This class builds Random Forest models based on configuration provided by the
    ConfigurationProvider. It supports both text and numeric features and provides
    hyperparameter optimization.
    """

    def __init__(
        self,
        name: str = "RandomForestModelBuilder",
        description: str = "Random Forest model builder using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the RandomForestModelBuilder.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = config_provider or ConfigurationProvider()

        # Create a default model configuration if it doesn't exist in the config
        self.config = {
            "tfidf": {
                "max_features": 5000,
                "ngram_range": [1, 3],
                "min_df": 2,
                "max_df": 0.9,
                "use_idf": True,
                "sublinear_tf": True,
            },
            "model": {
                "random_forest": {
                    "n_estimators": 200,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced_subsample",
                    "random_state": 42,
                }
            },
            "hyperparameter_optimization": {
                "param_grid": {
                    "preprocessor__text__tfidf__max_features": [3000, 5000, 7000],
                    "preprocessor__text__tfidf__ngram_range": [[1, 2], [1, 3]],
                    "clf__estimator__n_estimators": [100, 200, 300],
                    "clf__estimator__min_samples_leaf": [1, 2, 4],
                },
                "cv": 3,
                "scoring": "f1_macro",
                "verbose": 1,
            },
        }

        # Try to update from configuration provider if available
        try:
            # Check if there's a classifier section in the config
            if hasattr(self._config_provider.config, "classification"):
                classifier_config = (
                    self._config_provider.config.classification.model_dump()
                )
                if "model" in classifier_config:
                    self.config.update(classifier_config["model"])
                    logger.info(
                        "Updated model configuration from classification section"
                    )
            logger.debug(f"Using model configuration: {self.config}")
        except Exception as e:
            logger.warning(f"Could not load model configuration: {e}")
            logger.info("Using default model configuration")

        logger.info(f"Initialized {name}")

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        This method creates a pipeline with a preprocessor for text and numeric features
        and a Random Forest classifier.

        Args:
            **kwargs: Configuration parameters for the model. These override the
                     configuration from the provider.

        Returns:
            Configured model pipeline.

        Raises:
            ValueError: If the model cannot be built with the given parameters.
        """
        try:
            logger.info("Building Random Forest model")

            # Update config with kwargs
            if kwargs:
                for key, value in kwargs.items():
                    if key in self.config:
                        self.config[key] = value
                        logger.debug(
                            f"Updated config parameter {key} with value {value}"
                        )

            # Extract TF-IDF settings
            tfidf_settings = self.config.get("tfidf", {})
            max_features = tfidf_settings.get("max_features", 5000)
            ngram_range = tuple(tfidf_settings.get("ngram_range", [1, 3]))
            min_df = tfidf_settings.get("min_df", 2)
            max_df = tfidf_settings.get("max_df", 0.9)
            use_idf = tfidf_settings.get("use_idf", True)
            sublinear_tf = tfidf_settings.get("sublinear_tf", True)

            # Extract Random Forest settings
            rf_settings = self.config.get("model", {}).get("random_forest", {})
            n_estimators = rf_settings.get("n_estimators", 200)
            max_depth = rf_settings.get("max_depth", None)
            min_samples_split = rf_settings.get("min_samples_split", 2)
            min_samples_leaf = rf_settings.get("min_samples_leaf", 1)
            class_weight = rf_settings.get("class_weight", "balanced_subsample")
            random_state = rf_settings.get("random_state", 42)

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

            # Complete pipeline with feature processing and classifier
            pipeline = Pipeline(
                [
                    ("preprocessor", preprocessor),
                    (
                        "clf",
                        MultiOutputClassifier(
                            RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                class_weight=class_weight,
                                random_state=random_state,
                            )
                        ),
                    ),
                ]
            )

            logger.info("Random Forest model built successfully")
            return pipeline

        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise ValueError(f"Error building model: {str(e)}") from e

    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        This method uses GridSearchCV to find the best hyperparameters for the model.

        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for hyperparameter optimization.

        Returns:
            Optimized model pipeline.

        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        try:
            logger.info("Starting hyperparameter optimization")

            # Get hyperparameter optimization settings
            hp_settings = self.config.get("hyperparameter_optimization", {})
            param_grid = kwargs.get(
                "param_grid",
                hp_settings.get(
                    "param_grid",
                    {
                        "preprocessor__text__tfidf__max_features": [3000, 5000, 7000],
                        "preprocessor__text__tfidf__ngram_range": [(1, 2), (1, 3)],
                        "clf__estimator__n_estimators": [100, 200, 300],
                        "clf__estimator__min_samples_leaf": [1, 2, 4],
                    },
                ),
            )
            cv = kwargs.get("cv", hp_settings.get("cv", 3))
            scoring = kwargs.get("scoring", hp_settings.get("scoring", "f1_macro"))
            verbose = kwargs.get("verbose", hp_settings.get("verbose", 1))

            # Use GridSearchCV for hyperparameter optimization
            grid_search = GridSearchCV(
                model, param_grid=param_grid, cv=cv, scoring=scoring, verbose=verbose
            )

            # Fit the grid search to the data
            logger.info(f"Fitting GridSearchCV with {len(param_grid)} parameters")
            grid_search.fit(x_train, y_train)

            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_}")

            return grid_search.best_estimator_

        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise ValueError(f"Error optimizing hyperparameters: {str(e)}") from e
