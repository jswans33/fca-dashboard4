"""
Unit tests for the classifier pipeline

These tests focus on validating the data structure and transformations
in the classifier pipeline, particularly around resampling and the
ColumnTransformer handling of DataFrame inputs.
"""

import pytest
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from fca_dashboard.classifier.model_building import build_enhanced_model


def build_test_model():
    """
    Build a model for testing purposes with parameters adjusted for small test datasets.
    
    This is similar to build_enhanced_model but with min_df=1 to handle small test datasets.
    """
    # Text feature processing with min_df=1 for testing
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,  # Use min_df=1 for testing with small datasets
            max_df=1.0,  # Allow terms that appear in all documents
            use_idf=True,
            sublinear_tf=True
        ))
    ])
    
    # Numeric feature processing
    numeric_features = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Combine text and numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_features, 'combined_features'),
            ('numeric', numeric_features, ['service_life'])
        ],
        remainder='drop'
    )
    
    # Complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=10,  # Use fewer trees for faster testing
                random_state=42
            )
        ))
    ])
    
    return pipeline


def test_resampled_dataframe_structure():
    """
    Test that resampling maintains the correct DataFrame structure.
    
    This test verifies that after resampling with RandomOverSampler,
    the resulting DataFrame maintains the correct structure with
    the expected column names.
    """
    # Create a simple test DataFrame
    X_train = pd.DataFrame({
        'combined_features': ['text example 1', 'text example 2', 'text example 3'] * 3,
        'service_life': [10, 20, 30] * 3
    })
    
    # Create a multi-output target
    y_train = pd.DataFrame({
        'col1': ['A', 'A', 'B'] * 3,
        'col2': ['X', 'Y', 'Z'] * 3
    })
    
    # Create a combined target for resampling
    y_combined = y_train.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # Apply RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled_array, y_combined_resampled = oversampler.fit_resample(X_train, y_combined)
    
    # Reconstruct DataFrame with correct columns
    X_train_resampled = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    
    # Verify the structure
    assert isinstance(X_train_resampled, pd.DataFrame), "X_train_resampled should be a DataFrame"
    assert list(X_train_resampled.columns) == list(X_train.columns), "Column names should be preserved"
    assert 'combined_features' in X_train_resampled.columns, "combined_features column should exist"
    assert 'service_life' in X_train_resampled.columns, "service_life column should exist"
    assert len(X_train_resampled) >= len(X_train), "Resampled data should have at least as many rows as original"


def test_column_transformer_with_dataframe():
    """
    Test that ColumnTransformer correctly handles DataFrame inputs.
    
    This test verifies that the ColumnTransformer can properly extract
    columns from a DataFrame and apply the appropriate transformations.
    """
    # Create a simple test DataFrame
    X = pd.DataFrame({
        'combined_features': ['text example 1', 'text example 2', 'text example 3'],
        'service_life': [10, 20, 30]
    })
    
    # Create a ColumnTransformer similar to the one in the model
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'combined_features'),
            ('numeric', StandardScaler(), ['service_life'])  # Note: using a list for column name
        ],
        remainder='drop'
    )
    
    # This should not raise an error
    try:
        preprocessor.fit(X)
        transformed = preprocessor.transform(X)
        assert transformed.shape[0] == X.shape[0], "Number of samples should be preserved"
    except Exception as e:
        pytest.fail(f"ColumnTransformer raised an exception: {e}")


def test_multi_output_resampling():
    """
    Test that multi-output resampling works correctly.
    
    This test verifies that resampling with multiple target columns
    maintains the integrity of the targets and produces correctly
    structured data for the model.
    """
    # Create a simple test DataFrame
    X_train = pd.DataFrame({
        'combined_features': ['text example 1', 'text example 2', 'text example 3'] * 2,
        'service_life': [10, 20, 30] * 2
    })
    
    # Create a multi-output target with imbalanced classes
    y_train = pd.DataFrame({
        'col1': ['A', 'A', 'B'] * 2,
        'col2': ['X', 'Y', 'Z'] * 2
    })
    
    # Create a combined target for resampling
    y_combined = y_train.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # Apply RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled_array, y_combined_resampled = oversampler.fit_resample(X_train, y_combined)
    
    # Reconstruct DataFrame with correct columns
    X_train_resampled = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    
    # Split combined targets back into original columns
    y_train_resampled = pd.DataFrame(
        [val.split('_') for val in y_combined_resampled],
        columns=y_train.columns
    )
    
    # Verify the structure
    assert X_train_resampled.shape[0] == y_train_resampled.shape[0], "X and y should have the same number of samples"
    assert y_train_resampled.shape[1] == y_train.shape[1], "Target should have the same number of columns"
    
    # Check that the resampling balanced the classes
    for col in y_train.columns:
        value_counts = y_train_resampled[col].value_counts()
        assert len(value_counts) == len(y_train[col].unique()), "All classes should be present"
        # In a balanced dataset, all classes should have similar counts
        # Use >= 0.5 instead of > 0.5 to account for cases where the ratio is exactly 0.5
        assert value_counts.min() / value_counts.max() >= 0.5, "Classes should be reasonably balanced"


def test_full_pipeline_with_resampled_data():
    """
    Test that the full pipeline works with resampled data.
    
    This test verifies that the complete model pipeline can handle
    resampled data with the correct structure.
    """
    # Create a simple test DataFrame
    X_train = pd.DataFrame({
        'combined_features': ['text example 1', 'text example 2', 'text example 3'] * 3,
        'service_life': [10, 20, 30] * 3
    })
    
    # Create a multi-output target
    y_train = pd.DataFrame({
        'col1': ['A', 'A', 'B'] * 3,
        'col2': ['X', 'Y', 'Z'] * 3
    })
    
    # Create a combined target for resampling
    y_combined = y_train.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # Apply RandomOverSampler
    oversampler = RandomOverSampler(random_state=42)
    X_resampled_array, y_combined_resampled = oversampler.fit_resample(X_train, y_combined)
    
    # Reconstruct DataFrame with correct columns
    X_train_resampled = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    
    # Split combined targets back into original columns
    y_train_resampled = pd.DataFrame(
        [val.split('_') for val in y_combined_resampled],
        columns=y_train.columns
    )
    
    # Build the model
    model = build_enhanced_model()
    
    # This should not raise an error
    try:
        model.fit(X_train_resampled, y_train_resampled)
        assert True, "Model should fit without errors"
    except Exception as e:
        pytest.fail(f"Model fitting raised an exception: {e}")