"""
Equipment Classification Package

This package provides a machine learning pipeline for classifying equipment based on text descriptions
and numeric features. It follows SOLID principles with each module having a single responsibility:

- data_preprocessing.py: Loading and cleaning data
- feature_engineering.py: Feature transformations and engineering
- model_building.py: Core model architecture definition
- evaluation.py: Model evaluation and analysis
- training.py: Training logic and class imbalance handling

The package uses a combination of text features (via TF-IDF) and numeric features (like service_life)
to improve classification accuracy, particularly for "Other" categories.
"""

# Import main functions to expose at the package level
from fca_dashboard.classifier.data_preprocessing import load_and_preprocess_data
from fca_dashboard.classifier.feature_engineering import enhance_features, create_hierarchical_categories, enhanced_masterformat_mapping
from fca_dashboard.classifier.model_building import build_enhanced_model, optimize_hyperparameters
from fca_dashboard.classifier.evaluation import enhanced_evaluation, analyze_other_category_features, analyze_other_misclassifications
from fca_dashboard.classifier.training import train_enhanced_model, predict_with_enhanced_model, handle_class_imbalance

# Define what gets imported with "from fca_dashboard.classifier import *"
__all__ = [
    'load_and_preprocess_data',
    'enhance_features',
    'create_hierarchical_categories',
    'enhanced_masterformat_mapping',
    'build_enhanced_model',
    'optimize_hyperparameters',
    'enhanced_evaluation',
    'analyze_other_category_features',
    'analyze_other_misclassifications',
    'train_enhanced_model',
    'predict_with_enhanced_model',
    'handle_class_imbalance'
]