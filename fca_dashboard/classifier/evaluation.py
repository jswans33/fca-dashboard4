"""
Evaluation Module

This module handles model evaluation and analysis of "Other" categories.
It follows the Single Responsibility Principle by focusing solely on model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Union
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline


def enhanced_evaluation(model: Pipeline, X_test: Union[pd.Series, pd.DataFrame], y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the model with focus on "Other" categories performance
    
    This function has been updated to handle both Series and DataFrame inputs for X_test,
    to support the new pipeline structure that uses both text and numeric features.
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test: Test features
        y_test (pd.DataFrame): Test targets
        
    Returns:
        pd.DataFrame: Predictions dataframe
    """
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
    
    # Print overall evaluation metrics
    print("Model Evaluation:")
    for i, col in enumerate(y_test.columns):
        print(f"\n{col} Classification Report:")
        print(classification_report(y_test[col], y_pred_df[col]))
        print(f"{col} Accuracy:", accuracy_score(y_test[col], y_pred_df[col]))
        
        # Specifically examine "Other" category performance
        if "Other" in y_test[col].unique():
            other_indices = y_test[col] == "Other"
            other_accuracy = accuracy_score(
                y_test[col][other_indices], 
                y_pred_df[col][other_indices]
            )
            print(f"'Other' category accuracy for {col}: {other_accuracy:.4f}")
            
            # Calculate confusion metrics for "Other" category
            tp = ((y_test[col] == "Other") & (y_pred_df[col] == "Other")).sum()
            fp = ((y_test[col] != "Other") & (y_pred_df[col] == "Other")).sum()
            fn = ((y_test[col] == "Other") & (y_pred_df[col] != "Other")).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"'Other' category metrics for {col}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
    
    return y_pred_df


def analyze_other_category_features(model: Pipeline, X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None:
    """
    Analyze what features are most important for classifying items as "Other"
    
    This function has been updated to work with the new pipeline structure that uses
    a ColumnTransformer to combine text and numeric features.
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test (pd.Series): Test features
        y_test (pd.DataFrame): Test targets
        y_pred_df (pd.DataFrame): Predictions dataframe
    """
    # Extract the Random Forest model from the pipeline
    rf_model = model.named_steps['clf'].estimators_[0]
    
    # Get feature names from the TF-IDF vectorizer (now nested in preprocessor)
    # Access the text transformer from the ColumnTransformer, then the TF-IDF vectorizer
    tfidf_vectorizer = model.named_steps['preprocessor'].transformers_[0][1].named_steps['tfidf']
    text_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Also include numeric features for a complete analysis
    numeric_feature_names = ['service_life']
    all_feature_names = list(text_feature_names) + numeric_feature_names
    
    # For each classification column
    for col in y_test.columns:
        if "Other" in y_test[col].unique():
            print(f"\nAnalyzing 'Other' category for {col}:")
            
            # Find examples predicted as "Other"
            other_indices = y_pred_df[col] == "Other"
            
            if other_indices.sum() > 0:
                # Create a DataFrame with the required structure for the preprocessor
                if isinstance(X_test, pd.Series):
                    X_test_df = pd.DataFrame({
                        'combined_features': X_test[other_indices],
                        'service_life': np.zeros(other_indices.sum())  # Placeholder
                    })
                    # Transform using the full preprocessor
                    transformed_features = model.named_steps['preprocessor'].transform(X_test_df)
                    
                    # Extract just the text features (first part of the transformed features)
                    text_feature_count = len(text_feature_names)
                    text_features = transformed_features[:, :text_feature_count]
                    
                    # Get the average feature values for text features
                    avg_features = text_features.mean(axis=0)
                    if hasattr(avg_features, 'A1'):  # If it's a sparse matrix
                        avg_features = avg_features.A1
                    
                    # Get the top text features
                    top_indices = np.argsort(avg_features)[-20:]
                    
                    print("Top text features for 'Other' classification:")
                    for idx in top_indices:
                        print(f"  {text_feature_names[idx]}: {avg_features[idx]:.4f}")
                    
                    # Also analyze feature importance from the Random Forest model
                    # This will show the importance of both text and numeric features
                    print("\nFeature importance from Random Forest:")
                    
                    # Get feature importances for this specific estimator (for the current target column)
                    # Find the index of the current column in the target columns
                    col_idx = list(y_test.columns).index(col)
                    rf_estimator = model.named_steps['clf'].estimators_[col_idx]
                    
                    # Get feature importances
                    importances = rf_estimator.feature_importances_
                    
                    # Create a DataFrame to sort and display importances
                    importance_df = pd.DataFrame({
                        'feature': all_feature_names[:len(importances)],
                        'importance': importances
                    })
                    
                    # Sort by importance
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    
                    # Display top 10 features
                    print("Top 10 features by importance:")
                    for i, (feature, importance) in enumerate(zip(importance_df['feature'].head(10),
                                                                importance_df['importance'].head(10))):
                        print(f"  {feature}: {importance:.4f}")
                    
                    # Check if service_life is important
                    service_life_importance = importance_df[importance_df['feature'] == 'service_life']
                    if not service_life_importance.empty:
                        print(f"\nService life importance: {service_life_importance.iloc[0]['importance']:.4f}")
                        print(f"Service life rank: {importance_df['feature'].tolist().index('service_life') + 1} out of {len(importance_df)}")
                else:
                    print("Cannot analyze features: X_test is not a pandas Series")
            else:
                print("No examples predicted as 'Other'")


def analyze_other_misclassifications(X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None:
    """
    Analyze cases where "Other" was incorrectly predicted or missed
    
    Args:
        X_test (pd.Series): Test features
        y_test (pd.DataFrame): Test targets
        y_pred_df (pd.DataFrame): Predictions dataframe
    """
    for col in y_test.columns:
        if "Other" in y_test[col].unique():
            print(f"\nMisclassifications for 'Other' in {col}:")
            
            # False positives: Predicted as "Other" but actually something else
            fp_indices = (y_test[col] != "Other") & (y_pred_df[col] == "Other")
            
            if fp_indices.sum() > 0:
                print(f"\nFalse Positives (predicted as 'Other' but weren't): {fp_indices.sum()} cases")
                fp_examples = X_test[fp_indices].values[:5]  # Show first 5
                fp_actual = y_test[col][fp_indices].values[:5]
                
                for i, (example, actual) in enumerate(zip(fp_examples, fp_actual)):
                    print(f"Example {i+1}:")
                    print(f"  Text: {example[:100]}...")  # Show first 100 chars
                    print(f"  Actual class: {actual}")
            
            # False negatives: Actually "Other" but predicted as something else
            fn_indices = (y_test[col] == "Other") & (y_pred_df[col] != "Other")
            
            if fn_indices.sum() > 0:
                print(f"\nFalse Negatives (were 'Other' but predicted as something else): {fn_indices.sum()} cases")
                fn_examples = X_test[fn_indices].values[:5]  # Show first 5
                fn_predicted = y_pred_df[col][fn_indices].values[:5]
                
                for i, (example, predicted) in enumerate(zip(fn_examples, fn_predicted)):
                    print(f"Example {i+1}:")
                    print(f"  Text: {example[:100]}...")  # Show first 100 chars
                    print(f"  Predicted as: {predicted}")