"""
Integration tests for NexusML.

These tests verify that the different components of NexusML work together correctly.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.feature_engineering import create_hierarchical_categories, enhance_features
from nexusml.core.model import predict_with_enhanced_model
from nexusml.core.model_building import build_enhanced_model


@pytest.mark.skip(reason="This test requires a full pipeline run which takes time")
def test_full_pipeline(sample_data_path, sample_description, sample_service_life, tmp_path):
    """
    Test the full NexusML pipeline from data loading to prediction.
    
    This test is marked as skip by default because it can take a long time to run.
    """
    # Load and preprocess data
    df = load_and_preprocess_data(sample_data_path)
    
    # Enhance features
    df = enhance_features(df)
    
    # Create hierarchical categories
    df = create_hierarchical_categories(df)
    
    # Prepare training data
    X = pd.DataFrame({
        'combined_features': df['combined_features'],
        'service_life': df['service_life']
    })
    
    y = df[['Equipment_Category', 'Uniformat_Class', 'System_Type', 'Equipment_Type', 'System_Subtype']]
    
    # Build model
    model = build_enhanced_model()
    
    # Train model (this would take time)
    model.fit(X, y)
    
    # Make a prediction
    prediction = predict_with_enhanced_model(model, sample_description, sample_service_life)
    
    # Check the prediction
    assert isinstance(prediction, dict)
    assert 'Equipment_Category' in prediction
    assert 'Uniformat_Class' in prediction
    assert 'System_Type' in prediction
    assert 'Equipment_Type' in prediction
    assert 'System_Subtype' in prediction
    assert 'MasterFormat_Class' in prediction
    
    # Test visualization (optional)
    output_dir = str(tmp_path)
    from nexusml.core.model import visualize_category_distribution
    
    equipment_category_file, system_type_file = visualize_category_distribution(df, output_dir)
    
    assert os.path.exists(equipment_category_file)
    assert os.path.exists(system_type_file)


@pytest.mark.skip(reason="This test requires FCA Dashboard integration")
def test_fca_dashboard_integration():
    """
    Test integration with FCA Dashboard.
    
    This test is marked as skip by default because it requires FCA Dashboard to be available.
    """
    try:
        # Try to import from FCA Dashboard
        from fca_dashboard.classifier import predict_with_enhanced_model as fca_predict_model
        from fca_dashboard.classifier import train_enhanced_model as fca_train_model
        
        # If imports succeed, test the integration
        # This would be a more complex test that verifies the integration works
        pass
    except ImportError:
        pytest.skip("FCA Dashboard not available")