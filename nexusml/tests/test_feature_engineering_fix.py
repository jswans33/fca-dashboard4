"""
Test script to verify the fix for the feature engineering issue.

This script tests the ConfigDrivenFeatureEngineer with configurations that include
the 'name' parameter to ensure it doesn't cause the 'got multiple values for argument name' error.
"""

import pandas as pd
import logging
from pathlib import Path

from nexusml.core.feature_engineering.config_driven import ConfigDrivenFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_transformer_with_name_parameter():
    """Test creating transformers with configurations that include the 'name' parameter."""
    
    # Create a test configuration with 'name' parameters
    config = {
        "transformers": [
            {
                "type": "text_combiner",
                "columns": ["description", "manufacturer"],
                "separator": " ",
                "new_column": "combined_text",
                "name": "description_combiner"  # This was causing the error
            }
        ],
        "classification_systems": [
            {
                "name": "uniformat",  # This was causing the error
                "source_column": "description",
                "target_column": "uniformat_code",
                "mapping_type": "eav"
            }
        ],
        "keyword_classifications": [
            {
                "name": "keyword_classifier",  # This was causing the error
                "source_column": "description",
                "target_column": "category_name",
                "reference_manager": "uniformat_keywords"
            }
        ]
    }
    
    # Create a sample DataFrame
    data = pd.DataFrame({
        "description": ["Air Handling Unit", "Centrifugal Pump", "Chiller"],
        "manufacturer": ["Trane", "Grundfos", "Carrier"],
        "service_life": [20, 15, 25]
    })
    
    try:
        # Create the feature engineer with the test configuration
        logger.info("Creating ConfigDrivenFeatureEngineer with test configuration")
        feature_engineer = ConfigDrivenFeatureEngineer(config=config)
        
        # Create transformers from the configuration
        logger.info("Creating transformers from configuration")
        transformers = feature_engineer.create_transformers_from_config()
        
        # Log the number and types of transformers created
        logger.info(f"Successfully created {len(transformers)} transformers")
        for i, transformer in enumerate(transformers):
            logger.info(f"  Transformer {i+1}: {transformer.__class__.__name__}")
        
        # Fit the feature engineer before transforming
        logger.info("Fitting feature engineer")
        feature_engineer.fit(data)
        
        # Transform the data
        logger.info("Transforming data")
        transformed_data = feature_engineer.transform(data)
        
        # Log the columns in the transformed data
        logger.info(f"Transformed data columns: {transformed_data.columns.tolist()}")
        
        # If we get here without errors, the fix worked
        logger.info("✅ Test passed: No 'got multiple values for argument name' error")
        return True
        
    except Exception as e:
        # If we get an error, the fix didn't work
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Run the test
    success = test_transformer_with_name_parameter()
    
    # Exit with appropriate status code
    import sys
    sys.exit(0 if success else 1)