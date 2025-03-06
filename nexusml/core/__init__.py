"""
Core functionality for NexusML classification engine.
"""

# Import main functions to expose at the package level
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.evaluation import enhanced_evaluation
from nexusml.core.feature_engineering import create_hierarchical_categories, enhance_features
from nexusml.core.model_building import build_enhanced_model, optimize_hyperparameters
from nexusml.core.training import predict_with_enhanced_model, train_enhanced_model

__all__ = [
    "load_and_preprocess_data",
    "enhance_features",
    "create_hierarchical_categories",
    "build_enhanced_model",
    "optimize_hyperparameters",
    "enhanced_evaluation",
    "train_enhanced_model",
    "predict_with_enhanced_model",
]