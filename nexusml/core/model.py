"""
Enhanced Equipment Classification Model

This module implements a machine learning pipeline for classifying equipment based on text descriptions
and numeric features. Key features include:

1. Combined Text and Numeric Features:
   - Uses a ColumnTransformer to incorporate both text features (via TF-IDF) and numeric features
     (like service_life) into a single model.

2. Improved Handling of Imbalanced Classes:
   - Uses RandomOverSampler instead of SMOTE for text data, which duplicates existing samples
     rather than creating synthetic samples that don't correspond to meaningful text.
   - Also uses class_weight='balanced_subsample' in the RandomForestClassifier for additional
     protection against class imbalance.

3. Better Evaluation Metrics:
   - Uses f1_macro scoring for hyperparameter optimization, which is more appropriate for
     imbalanced classes than accuracy.
   - Provides detailed analysis of "Other" category performance.

4. Feature Importance Analysis:
   - Analyzes the importance of both text and numeric features in classifying equipment.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

# Local imports
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.evaluation import (
    analyze_other_category_features,
    analyze_other_misclassifications,
    enhanced_evaluation,
)
from nexusml.core.feature_engineering import (
    create_hierarchical_categories,
    enhance_features,
    enhanced_masterformat_mapping,
)
from nexusml.core.model_building import build_enhanced_model


def handle_class_imbalance(
    X: Union[pd.DataFrame, np.ndarray], y: pd.DataFrame
) -> Tuple[Union[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    Handle class imbalance to give proper weight to "Other" categories

    This function uses RandomOverSampler instead of SMOTE because:
    1. It's more appropriate for text data
    2. It duplicates existing samples rather than creating synthetic samples
    3. The duplicated samples maintain the original text meaning

    For numeric-only data, SMOTE might still be preferable, but for text or mixed data,
    RandomOverSampler is generally a better choice.

    Args:
        X: Features
        y: Target variables

    Returns:
        Tuple: (Resampled features, resampled targets)
    """
    # Check class distribution
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(y[col].value_counts())

    # Use RandomOverSampler to duplicate minority class samples
    # This is more appropriate for text data than SMOTE
    oversample = RandomOverSampler(sampling_strategy="auto", random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)

    print("\nAfter oversampling:")
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(pd.Series(y_resampled[col]).value_counts())

    return X_resampled, y_resampled


def train_enhanced_model(data_path: Optional[str] = None) -> Tuple[Any, pd.DataFrame]:
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
    X = pd.DataFrame({"combined_features": df["combined_features"], "service_life": df["service_life"]})

    # Use hierarchical classification targets
    y = df[["Equipment_Category", "Uniformat_Class", "System_Type", "Equipment_Type", "System_Subtype"]]

    # 5. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 6. Handle class imbalance using RandomOverSampler instead of SMOTE
    # RandomOverSampler is more appropriate for text data as it duplicates existing samples
    # rather than creating synthetic samples that don't correspond to meaningful text
    print("Handling class imbalance with RandomOverSampler...")

    # Apply RandomOverSampler directly to the DataFrame
    # This will duplicate minority class samples rather than creating synthetic samples
    oversampler = RandomOverSampler(random_state=42)

    # We need to convert the DataFrame to a format suitable for RandomOverSampler
    # For this, we'll create a temporary unique ID for each sample
    X_train_with_id = X_train.copy()
    X_train_with_id["temp_id"] = range(len(X_train_with_id))

    # Fit and transform using the oversampler
    # We use the ID column as the feature for oversampling, but the actual resampling
    # is based on the class distribution in y_train
    X_resampled_ids, y_train_resampled = oversampler.fit_resample(X_train_with_id[["temp_id"]], y_train)

    # Map the resampled IDs back to the original DataFrame rows
    # This effectively duplicates rows from the original DataFrame
    X_train_resampled = pd.DataFrame(columns=X_train.columns)
    for idx in X_resampled_ids["temp_id"]:
        X_train_resampled = pd.concat([X_train_resampled, X_train.iloc[[idx]]], ignore_index=True)

    # Print statistics about the resampling
    original_sample_count = X_train.shape[0]
    total_resampled_count = X_train_resampled.shape[0]
    print(f"Original samples: {original_sample_count}, Resampled samples: {total_resampled_count}")
    print(
        f"Shape of X_train_resampled: {X_train_resampled.shape}, Shape of y_train_resampled: {y_train_resampled.shape}"
    )

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


def predict_with_enhanced_model(model: Any, description: str, service_life: float = 0.0) -> dict:
    """
    Make predictions with enhanced detail for "Other" categories

    This function has been updated to work with the new pipeline structure that uses
    both text and numeric features.

    Args:
        model: Trained model pipeline
        description (str): Text description to classify
        service_life (float, optional): Service life value. Defaults to 0.0.

    Returns:
        dict: Prediction results with classifications
    """
    # Create a DataFrame with the required structure for the pipeline
    input_data = pd.DataFrame({"combined_features": [description], "service_life": [service_life]})

    # Predict using the trained pipeline
    pred = model.predict(input_data)[0]

    # Extract predictions
    result = {
        "Equipment_Category": pred[0],
        "Uniformat_Class": pred[1],
        "System_Type": pred[2],
        "Equipment_Type": pred[3],
        "System_Subtype": pred[4],
    }

    # Add MasterFormat prediction with enhanced mapping
    result["MasterFormat_Class"] = enhanced_masterformat_mapping(
        result["Uniformat_Class"],
        result["System_Type"],
        result["Equipment_Category"],
        # Extract equipment subcategory if available
        result["Equipment_Type"].split("-")[1] if "-" in result["Equipment_Type"] else None,
    )

    return result


def visualize_category_distribution(df: pd.DataFrame, output_dir: str = "outputs") -> Tuple[str, str]:
    """
    Visualize the distribution of categories in the dataset

    Args:
        df (pd.DataFrame): DataFrame with category columns
        output_dir (str, optional): Directory to save visualizations. Defaults to "outputs".

    Returns:
        Tuple[str, str]: Paths to the saved visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    equipment_category_file = f"{output_dir}/equipment_category_distribution.png"
    system_type_file = f"{output_dir}/system_type_distribution.png"

    # Generate visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="Equipment_Category")
    plt.title("Equipment Category Distribution")
    plt.tight_layout()
    plt.savefig(equipment_category_file)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="System_Type")
    plt.title("System Type Distribution")
    plt.tight_layout()
    plt.savefig(system_type_file)

    return equipment_category_file, system_type_file


# Example usage
if __name__ == "__main__":
    # Train enhanced model
    model, df = train_enhanced_model()

    # Example prediction with service life
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = 20.0  # Example service life in years
    prediction = predict_with_enhanced_model(model, description, service_life)

    print("\nEnhanced Prediction:")
    for key, value in prediction.items():
        print(f"{key}: {value}")

    # Visualize category distribution
    equipment_category_file, system_type_file = visualize_category_distribution(df)

    print(f"\nVisualizations saved to:")
    print(f"  - {equipment_category_file}")
    print(f"  - {system_type_file}")
