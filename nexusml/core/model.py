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
from typing import Any, Optional, Tuple, Union

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
    GenericFeatureEngineer,
    create_hierarchical_categories,
    enhance_features,
    enhanced_masterformat_mapping,
)
from nexusml.core.model_building import build_enhanced_model


def handle_class_imbalance(
    x: Union[pd.DataFrame, np.ndarray],
    y: pd.DataFrame,
    method: str = "random_over",
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Handle class imbalance with configurable method

    This function supports multiple oversampling strategies:
    - "random_over": Uses RandomOverSampler, which duplicates existing samples
      (better for text data as it preserves original text meaning)
    - "smote": Uses SMOTE to create synthetic samples
      (better for numeric-only data, but can create meaningless text)

    Args:
        x: Features
        y: Target variables
        method: Method to use ("random_over" or "smote")
        **kwargs: Additional parameters for the oversampler

    Returns:
        Tuple: (Resampled features, resampled targets)
    """
    # Check class distribution
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(y[col].value_counts())

    # Set default parameters
    params = {"sampling_strategy": "auto", "random_state": 42}
    params.update(kwargs)

    # Select oversampling method
    if method.lower() == "smote":
        try:
            from imblearn.over_sampling import SMOTE

            oversample = SMOTE(**params)
            print("Using SMOTE for oversampling...")
        except ImportError:
            print("SMOTE not available, falling back to RandomOverSampler...")
            oversample = RandomOverSampler(**params)
    else:  # default to random_over
        oversample = RandomOverSampler(**params)
        print("Using RandomOverSampler for oversampling...")

    # Apply oversampling
    # Handle the case where fit_resample might return 2 or 3 values
    result = oversample.fit_resample(x, y)

    # Extract the first two elements regardless of tuple size
    x_resampled, y_resampled = result[0], result[1]

    print("\nAfter oversampling:")
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(pd.Series(y_resampled[col]).value_counts())

    return x_resampled, y_resampled


def train_enhanced_model(
    data_path: Optional[str] = None,
    sampling_strategy: str = "random_over",
    feature_config_path: Optional[str] = None,
    **kwargs,
) -> Tuple[Any, pd.DataFrame]:
    """
    Train and evaluate the enhanced model with better handling of "Other" categories

    Args:
        data_path: Path to the CSV file. Defaults to None, which uses the standard location.
        sampling_strategy: Strategy for handling class imbalance ("random_over", "smote", or "direct")
        feature_config_path: Path to the feature configuration file. Defaults to None, which uses the standard location.
        **kwargs: Additional parameters for the oversampling method

    Returns:
        tuple: (trained model, preprocessed dataframe)
    """
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)

    # 2. Apply Generic Feature Engineering
    print("Applying Generic Feature Engineering...")
    feature_engineer = GenericFeatureEngineer(config_path=feature_config_path)
    df = feature_engineer.transform(df)

    # 3. Prepare training data - now including both text and numeric features
    # Create a DataFrame with both text and numeric features
    x = pd.DataFrame(
        {
            "combined_features": df["combined_text"],  # Using the name from config
            "service_life": df["service_life"],
        }
    )

    # Use hierarchical classification targets
    y = df[
        [
            "Equipment_Category",
            "Uniformat_Class",
            "System_Type",
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # 5. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # 6. Handle class imbalance using the specified strategy
    print(f"Handling class imbalance with {sampling_strategy}...")

    if sampling_strategy.lower() == "direct":
        # Skip oversampling entirely
        print("Skipping oversampling as requested...")
        x_train_resampled, y_train_resampled = x_train, y_train
    else:
        # For text data, we need a special approach with RandomOverSampler
        # We create a temporary unique ID for each sample
        x_train_with_id = x_train.copy()
        x_train_with_id["temp_id"] = range(len(x_train_with_id))

        # Handle class imbalance with the specified strategy
        if sampling_strategy.lower() == "smote":
            # For SMOTE, we need to apply it directly to the features
            # This might create synthetic text samples that don't make sense
            x_train_resampled, y_train_resampled = handle_class_imbalance(
                x_train, y_train, method="smote", **kwargs
            )
        else:  # default to random_over with ID-based approach
            # Fit and transform using the oversampler
            # We use the ID column as the feature for oversampling
            x_resampled_ids, y_train_resampled = handle_class_imbalance(
                x_train_with_id[["temp_id"]], y_train, method="random_over", **kwargs
            )

            # Map the resampled IDs back to the original DataFrame rows
            x_train_resampled = pd.DataFrame(columns=x_train.columns)
            for idx in x_resampled_ids["temp_id"]:
                x_train_resampled = pd.concat(
                    [x_train_resampled, x_train.iloc[[idx]]], ignore_index=True
                )

    # Print statistics about the resampling
    original_sample_count = x_train.shape[0]
    total_resampled_count = x_train_resampled.shape[0]
    print(
        f"Original samples: {original_sample_count}, Resampled samples: {total_resampled_count}"
    )
    print(
        f"Shape of x_train_resampled: {x_train_resampled.shape}, Shape of y_train_resampled: {y_train_resampled.shape}"
    )

    # Verify that the shapes match
    assert (
        x_train_resampled.shape[0] == y_train_resampled.shape[0]
    ), "Mismatch in sample counts after resampling"

    # 7. Build enhanced model
    print("Building enhanced model...")
    model = build_enhanced_model(sampling_strategy=sampling_strategy, **kwargs)

    # 8. Train the model
    print("Training model...")
    model.fit(x_train_resampled, y_train_resampled)

    # 9. Evaluate with focus on "Other" categories
    print("Evaluating model...")
    y_pred_df = enhanced_evaluation(model, x_test, y_test)

    # 10. Analyze "Other" category features
    print("Analyzing 'Other' category features...")
    analyze_other_category_features(model, x_test, y_test, y_pred_df)

    # 11. Analyze misclassifications for "Other" categories
    print("Analyzing 'Other' category misclassifications...")
    analyze_other_misclassifications(x_test, y_test, y_pred_df)

    return model, df


def predict_with_enhanced_model(
    model: Any, description: str, service_life: float = 0.0
) -> dict:
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
    input_data = pd.DataFrame(
        {"combined_text": [description], "service_life": [service_life]}
    )

    # Rename to match the expected column name in the pipeline
    input_data.rename(columns={"combined_text": "combined_features"}, inplace=True)

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
        (
            result["Equipment_Type"].split("-")[1]
            if "-" in result["Equipment_Type"]
            else None
        ),
    )

    return result


def visualize_category_distribution(
    df: pd.DataFrame, output_dir: str = "outputs"
) -> Tuple[str, str]:
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

    print("\nVisualizations saved to:")
    print(f"  - {equipment_category_file}")
    print(f"  - {system_type_file}")
