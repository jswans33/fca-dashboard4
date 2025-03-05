"""
Training Module

This module handles model training and class imbalance for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on training logic.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import from other modules
from fca_dashboard.classifier.data_preprocessing import load_and_preprocess_data
from fca_dashboard.classifier.feature_engineering import enhance_features, create_hierarchical_categories
from fca_dashboard.classifier.model_building import build_enhanced_model
from fca_dashboard.classifier.evaluation import enhanced_evaluation, analyze_other_category_features, analyze_other_misclassifications


def handle_class_imbalance(X: Union[pd.DataFrame, np.ndarray], y: pd.DataFrame) -> Tuple[Union[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    Handle class imbalance to give proper weight to "Other" categories
    
    This function uses RandomOverSampler instead of SMOTE because:
    1. It's more appropriate for text data
    2. It duplicates existing samples rather than creating synthetic samples
    3. The duplicated samples maintain the original text meaning
    
    For numeric-only data, SMOTE might still be preferable, but for text or mixed data,
    RandomOverSampler is generally a better choice.
    
    Args:
        X: Features to resample
        y (pd.DataFrame): Target labels to resample
        
    Returns:
        Tuple: Resampled features and targets
    """
    # Check class distribution
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(y[col].value_counts())
    
    # Use RandomOverSampler to duplicate minority class samples
    # This is more appropriate for text data than SMOTE
    oversample = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    
    print("\nAfter oversampling:")
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(pd.Series(y_resampled[col]).value_counts())
    
    return X_resampled, y_resampled


def train_enhanced_model(data_path: Optional[str] = None) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train and evaluate the enhanced model with better handling of "Other" categories
    
    Args:
        data_path (str, optional): Path to the CSV file. Defaults to None, which uses the standard location.
        
    Returns:
        tuple: (trained model, preprocessed dataframe)
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    
    # 2. Enhanced feature engineering
    print("Enhancing features...")
    df = enhance_features(df)
    
    # 3. Create hierarchical categories
    print("Creating hierarchical categories...")
    df = create_hierarchical_categories(df)
    
    # 4. Prepare training data - now including both text and numeric features
    # Create a DataFrame with both text and numeric features
    X = pd.DataFrame({
        'combined_features': df['combined_features'],
        'service_life': df['service_life']
    })
    
    # Use hierarchical classification targets
    y = df[['Equipment_Category', 'Uniformat_Class', 'System_Type', 'Equipment_Type', 'System_Subtype']]
    
    # 5. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 6. Handle class imbalance using RandomOverSampler with a composite key strategy
    # This allows us to handle multi-output classification properly
    print("Handling class imbalance with RandomOverSampler (multi-output)...")
    
    # For multi-output classification, we need to handle resampling differently
    # We'll use a composite key strategy as recommended
    
    # Create a combined multi-target column for resampling
    y_combined = y_train.apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    
    # Oversample
    oversampler = RandomOverSampler(random_state=42)
    X_resampled_array, y_combined_resampled = oversampler.fit_resample(X_train, y_combined)
    
    # Rebuild DataFrame properly after oversampling
    X_train_resampled = pd.DataFrame(X_resampled_array, columns=X_train.columns)
    
    # Reconstruct original multi-output targets after resampling
    y_train_resampled = pd.DataFrame(
        [val.split('_') for val in y_combined_resampled],
        columns=y_train.columns
    )
    
    # Reset indices explicitly
    X_train_resampled.reset_index(drop=True, inplace=True)
    y_train_resampled.reset_index(drop=True, inplace=True)
    
    # Print statistics about the resampling
    print(f"Original samples: {len(X_train)}, Resampled samples: {len(X_train_resampled)}")
    print(f"Shape of X_train_resampled: {X_train_resampled.shape}, Shape of y_train_resampled: {y_train_resampled.shape}")
    
    # Verify that the shapes match
    assert X_train_resampled.shape[0] == y_train_resampled.shape[0], "Mismatch in sample counts after resampling"
    
    # 7. Build enhanced model
    print("Building enhanced model...")
    model = build_enhanced_model()
    
    # 8. Train the model
    print("Training model...")
    model.fit(X_train_resampled, y_train_resampled)
    
    # 9. Evaluate with focus on "Other" categories
    print("Evaluating model...")
    y_pred_df = enhanced_evaluation(model, X_test, y_test)
    
    # 10. Analyze "Other" category features
    print("Analyzing 'Other' category features...")
    analyze_other_category_features(model, X_test, y_test, y_pred_df)
    
    # 11. Analyze misclassifications for "Other" categories
    print("Analyzing 'Other' category misclassifications...")
    analyze_other_misclassifications(X_test, y_test, y_pred_df)
    
    return model, df


def predict_with_enhanced_model(model: Pipeline, description: str, service_life: float = 0.0) -> dict:
    """
    Make predictions with enhanced detail for "Other" categories
    
    This function has been updated to work with the new pipeline structure that uses
    both text and numeric features.
    
    Args:
        model (Pipeline): Trained model pipeline
        description (str): Text description to classify
        service_life (float, optional): Service life value. Defaults to 0.0.
        
    Returns:
        dict: Prediction results with classifications
    """
    from fca_dashboard.classifier.feature_engineering import enhanced_masterformat_mapping
    
    # Create a DataFrame with the required structure for the pipeline
    input_data = pd.DataFrame({
        'combined_features': [description],
        'service_life': [service_life]
    })
    
    # Predict using the trained pipeline
    pred = model.predict(input_data)[0]
    
    # Extract predictions
    result = {
        'Equipment_Category': pred[0],
        'Uniformat_Class': pred[1],
        'System_Type': pred[2],
        'Equipment_Type': pred[3],
        'System_Subtype': pred[4]
    }
    
    # Add MasterFormat prediction with enhanced mapping
    result['MasterFormat_Class'] = enhanced_masterformat_mapping(
        result['Uniformat_Class'],
        result['System_Type'],
        result['Equipment_Category'],
        # Extract equipment subcategory if available
        result['Equipment_Type'].split('-')[1] if '-' in result['Equipment_Type'] else None
    )
    
    return result