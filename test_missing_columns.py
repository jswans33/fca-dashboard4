"""
Test script to verify the mapper's handling of DataFrames with missing required columns.
"""

import pandas as pd

from fca_dashboard.mappers.medtronics_mapper import MappingError, MedtronicsMapper
from fca_dashboard.utils.logging_config import get_logger

# Set up logging
logger = get_logger("test_missing_columns")

def test_missing_columns():
    """Test mapping a DataFrame with missing required columns."""
    logger.info("Testing mapper with DataFrame missing required columns")
    
    # Create a mapper
    mapper = MedtronicsMapper()
    
    # Create a DataFrame with missing required columns
    df = pd.DataFrame({
        'asset_tag': ['AHU-001', 'CH-001', 'B-001'],
        # Missing 'asset_name' and 'serial_number'
    })
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    
    # Map the DataFrame - this should raise a ValidationError
    logger.info("Mapping DataFrame with missing required columns")
    try:
        mapped_df = mapper.map_dataframe(df)
        logger.info("Successfully mapped DataFrame")
        return False  # This should not happen
    except MappingError as e:
        logger.info(f"Expected error occurred: {str(e)}")
        if "missing required columns" in str(e):
            return True
        else:
            logger.error(f"Unexpected error message: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error type: {type(e).__name__}: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_missing_columns()
    if success:
        print("Test passed: Missing columns were detected correctly")
    else:
        print("Test failed: Missing columns were not handled correctly")