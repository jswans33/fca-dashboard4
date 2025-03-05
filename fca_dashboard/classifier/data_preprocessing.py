"""
Data Preprocessing Module

This module handles loading and preprocessing data for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on data loading and cleaning.
"""

import pandas as pd
from typing import Optional
import yaml


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
        # Try to load from settings
        try:
            from fca_dashboard.utils.path_util import get_config_path, resolve_path
            
            settings_path = get_config_path("settings.yml")
            with open(settings_path, 'r') as file:
                settings = yaml.safe_load(file)
                
            data_path = settings.get('classifier', {}).get('data_paths', {}).get('training_data')
            if not data_path:
                # Fallback to default path
                data_path = "fca_dashboard/classifier/ingest/eq_ids.csv"
            
            # Resolve the path to ensure it exists
            data_path = str(resolve_path(data_path))
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")
            from pathlib import Path
            # Use absolute path as fallback
            data_path = str(Path(__file__).resolve().parent / "ingest" / "eq_ids.csv")
    
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