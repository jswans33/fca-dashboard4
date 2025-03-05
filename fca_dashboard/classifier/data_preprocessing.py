"""
Data Preprocessing Module

This module handles loading and preprocessing data for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on data loading and cleaning.
"""

import pandas as pd
from typing import Optional


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