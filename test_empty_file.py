"""
Test script to verify the mapper's handling of empty files.
"""

import pandas as pd

from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.utils.logging_config import get_logger

# Set up logging
logger = get_logger("test_empty_file")

def test_empty_file():
    """Test mapping an empty DataFrame."""
    logger.info("Testing mapper with empty DataFrame")
    
    # Create a mapper
    mapper = MedtronicsMapper()
    
    # Load the empty Excel file
    logger.info("Loading empty Excel file")
    df = pd.read_excel("empty_test.xlsx")
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"DataFrame is empty: {df.empty}")
    
    # Map the DataFrame
    logger.info("Mapping empty DataFrame")
    try:
        mapped_df = mapper.map_dataframe(df)
        logger.info("Successfully mapped empty DataFrame")
        logger.info(f"Mapped DataFrame shape: {mapped_df.shape}")
        logger.info(f"Mapped DataFrame columns: {list(mapped_df.columns)}")
        logger.info(f"Mapped DataFrame is empty: {mapped_df.empty}")
        return True
    except Exception as e:
        logger.error(f"Error mapping empty DataFrame: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_empty_file()
    if success:
        print("Test passed: Empty file was handled correctly")
    else:
        print("Test failed: Empty file caused an error")