"""
Model Building Module

This module defines the core model architecture for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on model definition.
"""

import os
from pathlib import Path

import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_enhanced_model(sampling_strategy: str = "random_over", **kwargs) -> Pipeline:
    """
    Build an enhanced model with configurable sampling strategy

    This model incorporates both text features (via TF-IDF) and numeric features
    (like service_life) using a ColumnTransformer to create a more comprehensive
    feature representation.

    Args:
        sampling_strategy: Sampling strategy to use ("random_over", "smote", or "direct")
        **kwargs: Additional parameters for the model

    Returns:
        Pipeline: Scikit-learn pipeline with preprocessor and classifier
    """
    # Try to load settings from configuration file
    try:
        # First try to load from fca_dashboard if available
        try:
            from fca_dashboard.utils.path_util import get_config_path

            settings_path = get_config_path("settings.yml")
        except ImportError:
            # If not running in fca_dashboard context, look for settings in nexusml
            settings_path = (
                Path(__file__).resolve().parent.parent.parent
                / "config"
                / "settings.yml"
            )
            if not settings_path.exists():
                # Fallback to environment variable
                settings_path_str = os.environ.get("NEXUSML_CONFIG", "")
                settings_path = (
                    Path(settings_path_str) if settings_path_str else Path("")
                )
                if not settings_path_str or not settings_path.exists():
                    raise FileNotFoundError("Could not find settings.yml") from None

        with open(settings_path, "r") as file:
            settings = yaml.safe_load(file)

        # Get TF-IDF settings
        tfidf_settings = settings.get("classifier", {}).get("tfidf", {})
        max_features = tfidf_settings.get("max_features", 5000)
        ngram_range = tuple(tfidf_settings.get("ngram_range", [1, 3]))
        min_df = tfidf_settings.get("min_df", 2)
        max_df = tfidf_settings.get("max_df", 0.9)
        use_idf = tfidf_settings.get("use_idf", True)
        sublinear_tf = tfidf_settings.get("sublinear_tf", True)

        # Get Random Forest settings
        rf_settings = (
            settings.get("classifier", {}).get("model", {}).get("random_forest", {})
        )
        n_estimators = rf_settings.get("n_estimators", 200)
        max_depth = rf_settings.get("max_depth", None)
        min_samples_split = rf_settings.get("min_samples_split", 2)
        min_samples_leaf = rf_settings.get("min_samples_leaf", 1)
        class_weight = rf_settings.get("class_weight", "balanced_subsample")
        random_state = rf_settings.get("random_state", 42)
    except Exception as e:
        print(f"Warning: Could not load settings: {e}")
        # Use default values if settings cannot be loaded
        max_features = 5000
        ngram_range = (1, 3)
        min_df = 2
        max_df = 0.9
        use_idf = True
        sublinear_tf = True

        n_estimators = 200
        max_depth = None
        min_samples_split = 2
        min_samples_leaf = 1
        class_weight = "balanced_subsample"
        random_state = 42

    # Text feature processing
    text_features = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,  # Include more n-grams for better feature extraction
                    min_df=min_df,  # Ignore very rare terms
                    max_df=max_df,  # Ignore very common terms
                    use_idf=use_idf,
                    sublinear_tf=sublinear_tf,  # Apply sublinear scaling to term frequencies
                ),
            )
        ]
    )

    # Numeric feature processing - simplified to just use StandardScaler
    # The ColumnTransformer handles column selection
    numeric_features = Pipeline(
        [("scaler", StandardScaler())]  # Scale numeric features
    )

    # Combine text and numeric features
    # Use a list for numeric features to ensure it's treated as a column name, not a Series
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_features, "combined_features"),
            (
                "numeric",
                numeric_features,
                ["service_life"],
            ),  # Use a list to specify column
        ],
        remainder="drop",  # Drop any other columns
    )

    # Complete pipeline with feature processing and classifier
    # Note: We use both RandomOverSampler (applied earlier) and class_weight='balanced_subsample'
    # for a two-pronged approach to handling imbalanced classes
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=n_estimators,  # More trees for better generalization
                        max_depth=max_depth,  # Allow trees to grow deeply
                        min_samples_split=min_samples_split,  # Default value
                        min_samples_leaf=min_samples_leaf,  # Default value
                        class_weight=class_weight,  # Additional protection against imbalance
                        random_state=random_state,
                    )
                ),
            ),
        ]
    )

    return pipeline


def optimize_hyperparameters(pipeline: Pipeline, x_train, y_train) -> Pipeline:
    """
    Optimize hyperparameters for better handling of all classes including "Other"

    This function uses GridSearchCV to find the best hyperparameters for the model.
    It optimizes both the text processing parameters and the classifier parameters.
    The scoring metric has been changed to f1_macro to better handle imbalanced classes.

    Args:
        pipeline (Pipeline): Model pipeline to optimize
        x_train: Training features
        y_train: Training targets

    Returns:
        Pipeline: Optimized pipeline
    """
    from sklearn.model_selection import GridSearchCV

    # Param grid for optimization with updated paths for the new pipeline structure
    param_grid = {
        "preprocessor__text__tfidf__max_features": [3000, 5000, 7000],
        "preprocessor__text__tfidf__ngram_range": [(1, 2), (1, 3)],
        "clf__estimator__n_estimators": [100, 200, 300],
        "clf__estimator__min_samples_leaf": [1, 2, 4],
    }

    # Use GridSearchCV for hyperparameter optimization
    # Changed scoring from 'accuracy' to 'f1_macro' for better handling of imbalanced classes
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1_macro",  # Better for imbalanced classes than accuracy
        verbose=1,
    )

    # Fit the grid search to the data
    # Note: x_train must now be a DataFrame with both 'combined_features' and 'service_life' columns
    grid_search.fit(x_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")

    return grid_search.best_estimator_
