"""
Model Building Example

This script demonstrates how to use the model building and training components
in the NexusML suite.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

from nexusml.core.model_building import (
    RandomForestBuilder,
    GradientBoostingBuilder,
    EnsembleBuilder,
)
from nexusml.core.model_training import (
    StandardModelTrainer,
    CrossValidationTrainer,
    GridSearchOptimizer,
)
from nexusml.core.model_building.base import BaseModelEvaluator, BaseModelSerializer


def load_sample_data():
    """
    Load sample data for demonstration.
    
    Returns:
        Tuple of (X, y) DataFrames.
    """
    # Try to load from standard locations
    data_paths = [
        "data/sample_data.csv",
        "examples/sample_data.csv",
        "nexusml/data/sample_data.csv",
        "nexusml/examples/sample_data.csv",
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
            break
    else:
        # If no file is found, create synthetic data
        print("Creating synthetic data")
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_df["combined_features"] = X_df.apply(
            lambda row: " ".join([f"{col}:{val:.2f}" for col, val in row.items()]), axis=1
        )
        X_df["service_life"] = np.random.uniform(10, 30, size=X.shape[0])
        
        # Convert y to DataFrame with multiple target columns
        y_df = pd.DataFrame({
            "category_name": [f"Category_{i}" for i in y],
            "uniformat_code": [f"U{i}" for i in y],
            "mcaa_system_category": [f"System_{i}" for i in y],
            "Equipment_Type": [f"Type_{i}" for i in y],
            "System_Subtype": [f"Subtype_{i}" for i in y],
        })
        
        return X_df, y_df
    
    # If we loaded from a file, process it
    # Assume the file has the required columns
    # Extract features and targets
    feature_cols = [col for col in df.columns if col not in [
        "category_name", "uniformat_code", "mcaa_system_category",
        "Equipment_Type", "System_Subtype"
    ]]
    
    X_df = df[feature_cols]
    y_df = df[[
        "category_name", "uniformat_code", "mcaa_system_category",
        "Equipment_Type", "System_Subtype"
    ]]
    
    return X_df, y_df


def main():
    """
    Main function to demonstrate model building and training.
    """
    print("NexusML Model Building Example")
    print("==============================\n")
    
    # Load sample data
    X, y = load_sample_data()
    print(f"Loaded data with {len(X)} samples")
    print(f"X columns: {X.columns.tolist()}")
    print(f"y columns: {y.columns.tolist()}\n")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}\n")
    
    # Example 1: Building and training a Random Forest model
    print("Example 1: Building and training a Random Forest model")
    print("-----------------------------------------------------")
    
    # Create a Random Forest model builder
    rf_builder = RandomForestBuilder()
    
    # Build the model
    rf_model = rf_builder.build_model()
    print("Random Forest model built")
    
    # Create a standard model trainer
    trainer = StandardModelTrainer()
    
    # Train the model
    trained_rf_model = trainer.train(rf_model, X_train, y_train)
    print("Random Forest model trained\n")
    
    # Example 2: Building and training a Gradient Boosting model with standard training
    print("Example 2: Building and training a Gradient Boosting model")
    print("----------------------------------------------------------")
    
    # Create a Gradient Boosting model builder
    gb_builder = GradientBoostingBuilder()
    
    # Build the model
    gb_model = gb_builder.build_model()
    print("Gradient Boosting model built")
    
    # Create a standard model trainer
    gb_trainer = StandardModelTrainer()
    
    # Train the model
    trained_gb_model = gb_trainer.train(gb_model, X_train, y_train)
    print("Gradient Boosting model trained\n")
    
    # Example 3: Building and training an Ensemble model
    print("Example 3: Building and training an Ensemble model")
    print("--------------------------------------------------")
    
    # Create an Ensemble model builder
    ensemble_builder = EnsembleBuilder()
    
    # Build the model
    ensemble_model = ensemble_builder.build_model()
    print("Ensemble model built")
    
    # Create a standard model trainer
    ensemble_trainer = StandardModelTrainer()
    
    # Train the model
    optimized_model = ensemble_trainer.train(ensemble_model, X_train, y_train)
    print("Ensemble model trained\n")
    
    # Example 4: Evaluating models
    print("Example 4: Evaluating models")
    print("---------------------------")
    
    # Create a model evaluator
    evaluator = BaseModelEvaluator()
    
    # Evaluate the Random Forest model
    rf_metrics = evaluator.evaluate(trained_rf_model, X_test, y_test)
    print("Random Forest model evaluated")
    print(f"Overall accuracy: {rf_metrics['overall']['accuracy_mean']:.4f}")
    print(f"Overall F1 score: {rf_metrics['overall']['f1_macro_mean']:.4f}\n")
    
    # Evaluate the Gradient Boosting model
    gb_metrics = evaluator.evaluate(trained_gb_model, X_test, y_test)
    print("Gradient Boosting model evaluated")
    print(f"Overall accuracy: {gb_metrics['overall']['accuracy_mean']:.4f}")
    print(f"Overall F1 score: {gb_metrics['overall']['f1_macro_mean']:.4f}\n")
    
    # Evaluate the Ensemble model
    ensemble_metrics = evaluator.evaluate(optimized_model, X_test, y_test)
    print("Ensemble model evaluated")
    print(f"Overall accuracy: {ensemble_metrics['overall']['accuracy_mean']:.4f}")
    print(f"Overall F1 score: {ensemble_metrics['overall']['f1_macro_mean']:.4f}\n")
    
    # Example 5: Saving and loading models
    print("Example 5: Saving and loading models")
    print("---------------------------------")
    
    # Create a model serializer
    serializer = BaseModelSerializer()
    
    # Create output directory if it doesn't exist
    os.makedirs("nexusml/output/models", exist_ok=True)
    
    # Save the best model
    best_model_path = "nexusml/output/models/best_model.pkl"
    serializer.save_model(optimized_model, best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Load the model
    loaded_model = serializer.load_model(best_model_path)
    print("Best model loaded")
    
    # Evaluate the loaded model
    loaded_metrics = evaluator.evaluate(loaded_model, X_test, y_test)
    print("Loaded model evaluated")
    print(f"Overall accuracy: {loaded_metrics['overall']['accuracy_mean']:.4f}")
    print(f"Overall F1 score: {loaded_metrics['overall']['f1_macro_mean']:.4f}\n")
    
    print("Model Building Example completed successfully!")


if __name__ == "__main__":
    main()