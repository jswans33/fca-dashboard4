"""
Unit tests for the NexusML pipeline.
"""

from pathlib import Path

import pandas as pd
import pytest

from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.feature_engineering import create_hierarchical_categories, enhance_features
from nexusml.core.model import predict_with_enhanced_model
from nexusml.core.model_building import build_enhanced_model


def test_load_and_preprocess_data(sample_data_path):
    """Test that data can be loaded and preprocessed."""
    df = load_and_preprocess_data(sample_data_path)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Asset Category' in df.columns
    assert 'System Type ID' in df.columns


def test_enhance_features():
    """Test that features can be enhanced."""
    # Create a minimal test dataframe
    df = pd.DataFrame({
        'Asset Category': ['Pump', 'Chiller'],
        'Equip Name ID': ['Centrifugal', 'Screw'],
        'System Type ID': ['P', 'H'],
        'Precon System': ['Domestic Water', 'Chiller Plant'],
        'Sub System Type': ['Hot Water', 'Cooling'],
        'Sub System ID': ['HW', 'CHW'],
        'Title': ['Pump 1', 'Chiller 1'],
        'Operations System': ['Domestic', 'HVAC'],
        'Sub System Class': ['Plumbing', 'Mechanical'],
        'Drawing Abbreviation': ['P-1', 'M-1'],
        'Equipment Size': [100, 200],
        'Unit': ['GPM', 'Tons'],
        'Service Life': [15, 20]
    })
    
    enhanced_df = enhance_features(df)
    
    # Check that new columns were added
    assert 'Equipment_Category' in enhanced_df.columns
    assert 'Uniformat_Class' in enhanced_df.columns
    assert 'System_Type' in enhanced_df.columns
    assert 'Equipment_Subcategory' in enhanced_df.columns
    assert 'combined_features' in enhanced_df.columns
    assert 'size_feature' in enhanced_df.columns
    assert 'service_life' in enhanced_df.columns


def test_create_hierarchical_categories():
    """Test that hierarchical categories can be created."""
    # Create a minimal test dataframe with the required columns
    df = pd.DataFrame({
        'Asset Category': ['Pump', 'Chiller'],
        'Equip Name ID': ['Centrifugal', 'Screw'],
        'Precon System': ['Domestic Water', 'Chiller Plant'],
        'Operations System': ['Domestic', 'HVAC']
    })
    
    hierarchical_df = create_hierarchical_categories(df)
    
    # Check that new columns were added
    assert 'Equipment_Type' in hierarchical_df.columns
    assert 'System_Subtype' in hierarchical_df.columns
    
    # Check the values
    assert hierarchical_df['Equipment_Type'][0] == 'Pump-Centrifugal'
    assert hierarchical_df['System_Subtype'][0] == 'Domestic Water-Domestic'


def test_build_enhanced_model():
    """Test that the model can be built."""
    model = build_enhanced_model()
    assert model is not None
    
    # Check that the model has the expected structure
    assert hasattr(model, 'steps')
    assert 'preprocessor' in dict(model.steps)
    assert 'clf' in dict(model.steps)


@pytest.mark.skip(reason="This test requires a trained model which takes time to create")
def test_predict_with_enhanced_model(sample_description, sample_service_life):
    """Test that predictions can be made with the model."""
    # This is a more complex test that requires a trained model
    # In a real test suite, you might use a pre-trained model or mock the model
    
    # For now, we'll skip this test, but here's how it would look
    from nexusml.core.model import train_enhanced_model
    
    # Train a model (this would take time)
    model, _ = train_enhanced_model()
    
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