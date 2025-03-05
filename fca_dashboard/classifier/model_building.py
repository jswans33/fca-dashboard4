"""
Model Building Module

This module defines the core model architecture for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on model definition.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def build_enhanced_model() -> Pipeline:
    """
    Build an enhanced model with better handling of "Other" categories
    
    This model incorporates both text features (via TF-IDF) and numeric features
    (like service_life) using a ColumnTransformer to create a more comprehensive
    feature representation.
    
    Returns:
        Pipeline: Scikit-learn pipeline with preprocessor and classifier
    """
    # Text feature processing
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),  # Include more n-grams for better feature extraction
            min_df=2,            # Ignore very rare terms
            max_df=0.9,          # Ignore very common terms
            use_idf=True,
            sublinear_tf=True    # Apply sublinear scaling to term frequencies
        ))
    ])
    
    # Numeric feature processing - simplified to just use StandardScaler
    # The ColumnTransformer handles column selection
    numeric_features = Pipeline([
        ('scaler', StandardScaler())  # Scale numeric features
    ])
    
    # Combine text and numeric features
    # Use a list for numeric features to ensure it's treated as a column name, not a Series
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_features, 'combined_features'),
            ('numeric', numeric_features, ['service_life'])  # Use a list to specify column
        ],
        remainder='drop'  # Drop any other columns
    )
    
    # Complete pipeline with feature processing and classifier
    # Note: We use both RandomOverSampler (applied earlier) and class_weight='balanced_subsample'
    # for a two-pronged approach to handling imbalanced classes
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,    # More trees for better generalization
                max_depth=None,      # Allow trees to grow deeply
                min_samples_split=2, # Default value
                min_samples_leaf=1,  # Default value
                class_weight='balanced_subsample',  # Additional protection against imbalance
                random_state=42
            )
        ))
    ])
    
    return pipeline


def optimize_hyperparameters(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    """
    Optimize hyperparameters for better handling of all classes including "Other"
    
    This function uses GridSearchCV to find the best hyperparameters for the model.
    It optimizes both the text processing parameters and the classifier parameters.
    The scoring metric has been changed to f1_macro to better handle imbalanced classes.
    
    Args:
        pipeline (Pipeline): Model pipeline to optimize
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Pipeline: Optimized pipeline
    """
    from sklearn.model_selection import GridSearchCV
    
    # Param grid for optimization with updated paths for the new pipeline structure
    param_grid = {
        'preprocessor__text__tfidf__max_features': [3000, 5000, 7000],
        'preprocessor__text__tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__min_samples_leaf': [1, 2, 4]
    }
    
    # Use GridSearchCV for hyperparameter optimization
    # Changed scoring from 'accuracy' to 'f1_macro' for better handling of imbalanced classes
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='f1_macro',  # Better for imbalanced classes than accuracy
        verbose=1
    )
    
    # Fit the grid search to the data
    # Note: X_train must now be a DataFrame with both 'combined_features' and 'service_life' columns
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_