# Standard library imports
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union, Any

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# 1. Enhanced Data Preprocessing with Hierarchical Classification
def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file
    
    Args:
        data_path (str, optional): Path to the CSV file. Defaults to the standard location.
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Use default path if none provided
    if data_path is None:
        data_path = "C:/Repos/fca-dashboard4/fca_dashboard/classifier/ingest/eq_ids.csv"
    
    # Read CSV file using pandas
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Try with a different encoding if utf-8 fails
        df = pd.read_csv(data_path, encoding='latin1')
    
    # Clean up column names (remove any leading/trailing whitespace)
    df.columns = [col.strip() for col in df.columns]
    
    # Fill NaN values with empty strings for text columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna('')
    
    return df

# 2. Enhanced Feature Engineering
def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with hierarchical structure and more granular categories
    """
    # Extract primary classification columns
    df['Equipment_Category'] = df['Asset Category']
    df['Uniformat_Class'] = df['System Type ID']
    df['System_Type'] = df['Precon System']
    
    # Create subcategory field for more granular classification
    df['Equipment_Subcategory'] = df['Equip Name ID']
    
    # Combine fields for rich text features
    df['combined_features'] = (
        df['Asset Category'] + ' ' + 
        df['Equip Name ID'] + ' ' + 
        df['Sub System Type'] + ' ' + 
        df['Sub System ID'] + ' ' + 
        df['Title'] + ' ' + 
        df['Precon System'] + ' ' + 
        df['Operations System'] + ' ' +
        df['Sub System Class'] + ' ' +
        df['Drawing Abbreviation']
    )
    
    # Add equipment size and unit as features
    df['size_feature'] = df['Equipment Size'].astype(str) + ' ' + df['Unit'].astype(str)
    
    # Add service life as a feature
    df['service_life'] = df['Service Life'].fillna(0).astype(float)
    
    # Fill NaN values
    df['combined_features'] = df['combined_features'].fillna('')
    
    return df

# 3. Create hierarchical classification structure
def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical category structure to better handle "Other" categories
    """
    # Create Equipment Type - a more detailed category than Equipment_Category
    df['Equipment_Type'] = df['Asset Category'] + '-' + df['Equip Name ID']
    
    # Create System Subtype - a more detailed category than System_Type
    df['System_Subtype'] = df['Precon System'] + '-' + df['Operations System']
    
    # Create target variables for hierarchical classification
    return df

# 4. Balancing classes for better "Other" category recognition
def handle_class_imbalance(X: Union[pd.DataFrame, np.ndarray], y: pd.DataFrame) -> Tuple[Union[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    Handle class imbalance to give proper weight to "Other" categories
    """
    # Check class distribution
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(y[col].value_counts())
    
    # For demonstration, let's use SMOTE to oversample minority classes
    # In a real implementation, you'd need to tune this for your specific data
    oversample = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversample.fit_resample(X, y)
    
    print("\nAfter oversampling:")
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(pd.Series(y_resampled[col]).value_counts())
    
    return X_resampled, y_resampled

# 5. Enhanced model with deeper architecture
def build_enhanced_model() -> Pipeline:
    """
    Build an enhanced model with better handling of "Other" categories
    """
    # Pipeline with TF-IDF vectorizer and Random Forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 3),  # Include more n-grams for better feature extraction
            min_df=2,            # Ignore very rare terms
            max_df=0.9,          # Ignore very common terms
            use_idf=True,
            sublinear_tf=True    # Apply sublinear scaling to term frequencies
        )),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,    # More trees for better generalization
                max_depth=None,      # Allow trees to grow deeply
                min_samples_split=2, # Default value
                min_samples_leaf=1,  # Default value
                class_weight='balanced_subsample',  # Handle imbalanced classes
                random_state=42
            )
        ))
    ])
    
    return pipeline

# 6. Hyperparameter optimization
def optimize_hyperparameters(pipeline: Pipeline, X_train: pd.Series, y_train: pd.DataFrame) -> Pipeline:
    """
    Optimize hyperparameters for better handling of all classes including "Other"
    """
    # Param grid for optimization
    param_grid = {
        'tfidf__max_features': [3000, 5000, 7000],
        'tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__min_samples_leaf': [1, 2, 4]
    }
    
    # Use GridSearchCV for hyperparameter optimization
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=3, 
        scoring='accuracy', 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")
    
    return grid_search.best_estimator_

# 7. Enhanced evaluation with focus on "Other" categories
def enhanced_evaluation(model: Pipeline, X_test: pd.Series, y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the model with focus on "Other" categories performance
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

# 8. Feature importance analysis for "Other" categories
def analyze_other_category_features(model: Pipeline, X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None:
    """
    Analyze what features are most important for classifying items as "Other"
    """
    # Extract the Random Forest model from the pipeline
    rf_model = model.named_steps['clf'].estimators_[0]
    
    # Get feature names from the TF-IDF vectorizer
    feature_names = model.named_steps['tfidf'].get_feature_names_out()
    
    # For each classification column
    for col in y_test.columns:
        if "Other" in y_test[col].unique():
            print(f"\nAnalyzing 'Other' category for {col}:")
            
            # Find examples predicted as "Other"
            other_indices = y_pred_df[col] == "Other"
            
            if other_indices.sum() > 0:
                # Get the feature vectors for these examples
                other_vectors = model.named_steps['tfidf'].transform(X_test[other_indices])
                
                # Get the average feature values
                avg_features = other_vectors.mean(axis=0).A1
                
                # Get the top features
                top_indices = np.argsort(avg_features)[-20:]
                
                print("Top features for 'Other' classification:")
                for idx in top_indices:
                    print(f"  {feature_names[idx]}: {avg_features[idx]:.4f}")
            else:
                print("No examples predicted as 'Other'")

# 9. Analysis of misclassifications specifically for "Other" categories
def analyze_other_misclassifications(X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None:
    """
    Analyze cases where "Other" was incorrectly predicted or missed
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

# 10. MasterFormat mapping enhanced to handle specialty equipment
def enhanced_masterformat_mapping(uniformat_class: str, system_type: str, equipment_category: str, equipment_subcategory: Optional[str] = None) -> str:
    """
    Enhanced mapping with better handling of specialty equipment types
    """
    # Primary mapping
    primary_mapping = {
        'H': {
            'Chiller Plant': '23 64 00',  # Commercial Water Chillers
            'Cooling Tower Plant': '23 65 00',  # Cooling Towers
            'Heating Water Boiler Plant': '23 52 00',  # Heating Boilers
            'Steam Boiler Plant': '23 52 33',  # Steam Heating Boilers
            'Air Handling Units': '23 73 00',  # Indoor Central-Station Air-Handling Units
        },
        'P': {
            'Domestic Water Plant': '22 11 00',  # Facility Water Distribution
            'Medical/Lab Gas Plant': '22 63 00',  # Gas Systems for Laboratory and Healthcare Facilities
            'Sanitary Equipment': '22 13 00',  # Facility Sanitary Sewerage
        },
        'SM': {
            'Air Handling Units': '23 74 00',  # Packaged Outdoor HVAC Equipment
            'SM Accessories': '23 33 00',  # Air Duct Accessories
            'SM Equipment': '23 30 00',  # HVAC Air Distribution
        }
    }
    
    # Secondary mapping for specific equipment types that were in "Other"
    equipment_specific_mapping = {
        'Heat Exchanger': '23 57 00',  # Heat Exchangers for HVAC
        'Water Softener': '22 31 00',  # Domestic Water Softeners
        'Humidifier': '23 84 13',  # Humidifiers
        'Radiant Panel': '23 83 16',  # Radiant-Heating Hydronic Piping
        'Make-up Air Unit': '23 74 23',  # Packaged Outdoor Heating-Only Makeup Air Units
        'Energy Recovery Ventilator': '23 72 00',  # Air-to-Air Energy Recovery Equipment
        'DI/RO Equipment': '22 31 16',  # Deionized-Water Piping
        'Bypass Filter Feeder': '23 25 00',  # HVAC Water Treatment
        'Grease Interceptor': '22 13 23',  # Sanitary Waste Interceptors
        'Heat Trace': '23 05 33',  # Heat Tracing for HVAC Piping
        'Dust Collector': '23 35 16',  # Engine Exhaust Systems
        'Venturi VAV Box': '23 36 00',  # Air Terminal Units
        'Water Treatment Controller': '23 25 13',  # Water Treatment for Closed-Loop Hydronic Systems
        'Polishing System': '23 25 00',  # HVAC Water Treatment
        'Ozone Generator': '22 67 00',  # Processed Water Systems for Laboratory and Healthcare Facilities
    }
    
    # Try equipment-specific mapping first
    if equipment_subcategory in equipment_specific_mapping:
        return equipment_specific_mapping[equipment_subcategory]
    
    # Then try primary mapping
    if uniformat_class in primary_mapping and system_type in primary_mapping[uniformat_class]:
        return primary_mapping[uniformat_class][system_type]
    
    # Refined fallback mappings by Uniformat class
    fallbacks = {
        'H': '23 00 00',  # Heating, Ventilating, and Air Conditioning (HVAC)
        'P': '22 00 00',  # Plumbing
        'SM': '23 00 00',  # HVAC
        'R': '11 40 00',  # Foodservice Equipment (Refrigeration)
    }
    
    return fallbacks.get(uniformat_class, '00 00 00')  # Return unknown if no match

# 11. Main function to train and evaluate the enhanced model
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
    
    # 4. Prepare training data
    X = df['combined_features']
    
    # Use hierarchical classification targets
    y = df[['Equipment_Category', 'Uniformat_Class', 'System_Type', 'Equipment_Type', 'System_Subtype']]
    
    # 5. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Convert to DataFrame for feature engineering
    X_train_vec = pd.DataFrame({'text': X_train})
    X_test_vec = pd.DataFrame({'text': X_test})
    
    # 6. Handle class imbalance (only for demonstration - you'd need to implement SMOTE)
    # In a real implementation, use:
    # X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_vec, y_train)
    # For this demo, we'll skip resampling
    X_train_resampled, y_train_resampled = X_train, y_train
    
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

# 12. Enhanced prediction function
def predict_with_enhanced_model(model: Pipeline, description: str) -> dict:
    """
    Make predictions with enhanced detail for "Other" categories
    
    Args:
        model (Pipeline): Trained model pipeline
        description (str): Text description to classify
        
    Returns:
        dict: Prediction results with classifications
    """
    # Predict using the trained pipeline
    pred = model.predict([description])[0]
    
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

# Example usage
if __name__ == "__main__":
    # Path to the CSV file
    data_path = "C:/Repos/fca-dashboard4/fca_dashboard/classifier/ingest/eq_ids.csv"
    
    # Train enhanced model using the CSV file
    model, df = train_enhanced_model(data_path)
    
    # Example prediction
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    prediction = predict_with_enhanced_model(model, description)
    
    print("\nEnhanced Prediction:")
    for key, value in prediction.items():
        print(f"{key}: {value}")

    # Visualize category distribution to better understand "Other" classes
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='Equipment_Category')
    plt.title('Equipment Category Distribution')
    plt.tight_layout()
    plt.savefig('equipment_category_distribution.png')
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='System_Type')
    plt.title('System Type Distribution')
    plt.tight_layout()
    plt.savefig('system_type_distribution.png')