"""
Example Usage of Equipment Classification Package

This script demonstrates how to use the refactored equipment classification package.
It shows the complete workflow from data loading to model training and prediction.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import from the classifier package
from fca_dashboard.classifier import (
    train_enhanced_model,
    predict_with_enhanced_model
)


def main():
    """
    Main function demonstrating the usage of the equipment classification package
    """
    # Path to the CSV file
    data_path = "C:/Repos/fca-dashboard4/fca_dashboard/classifier/ingest/eq_ids.csv"
    
    # Train enhanced model using the CSV file
    print("Training the model...")
    model, df = train_enhanced_model(data_path)
    
    # Example prediction with service life
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = 20.0  # Example service life in years
    
    print("\nMaking a prediction for:")
    print(f"Description: {description}")
    print(f"Service Life: {service_life} years")
    
    prediction = predict_with_enhanced_model(model, description, service_life)
    
    print("\nEnhanced Prediction:")
    for key, value in prediction.items():
        print(f"{key}: {value}")

    # Visualize category distribution to better understand "Other" classes
    print("\nGenerating visualizations...")
    
    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='Equipment_Category')
    plt.title('Equipment Category Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equipment_category_distribution.png'))
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='System_Type')
    plt.title('System Type Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'system_type_distribution.png'))
    
    print(f"Visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()