"""
Data Preprocessing Module

This module handles loading and preprocessing data for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on data loading and cleaning.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd
import yaml
from pandas.io.parsers import TextFileReader


def load_data_config() -> Dict:
    """
    Load the data preprocessing configuration from YAML file.

    Returns:
        Dict: Configuration dictionary
    """
    try:
        # Get the path to the configuration file
        root = Path(__file__).resolve().parent.parent
        config_path = root / "config" / "data_config.yml"

        # Load the configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config
    except Exception as e:
        print(f"Warning: Could not load data configuration: {e}")
        # Return a minimal default configuration
        return {
            "required_columns": [],
            "training_data": {
                "default_path": "ingest/data/eq_ids.csv",
                "encoding": "utf-8",
                "fallback_encoding": "latin1",
            },
        }


def verify_required_columns(
    df: pd.DataFrame, config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Verify that all required columns exist in the DataFrame and create them if they don't.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (Dict, optional): Configuration dictionary. If None, loads from file.

    Returns:
        pd.DataFrame: DataFrame with all required columns
    """
    if config is None:
        config = load_data_config()

    required_columns = config.get("required_columns", [])

    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Check each required column
    for column_info in required_columns:
        column_name = column_info["name"]
        default_value = column_info["default_value"]
        data_type = column_info["data_type"]

        # Check if the column exists
        if column_name not in df.columns:
            print(
                f"Warning: Required column '{column_name}' not found. Creating with default value."
            )

            # Create the column with the default value
            if data_type == "str":
                df[column_name] = default_value
            elif data_type == "float":
                df[column_name] = float(default_value)
            elif data_type == "int":
                df[column_name] = int(default_value)
            else:
                # Default to string if type is unknown
                df[column_name] = default_value

    return df


def load_and_preprocess_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file

    Args:
        data_path (str, optional): Path to the CSV file. Defaults to the standard location.

    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    # Load the configuration
    config = load_data_config()
    training_data_config = config.get("training_data", {})

    # Use default path if none provided
    if data_path is None:
        # Try to load from settings if available
        try:
            # Check if we're running within the fca_dashboard context
            try:
                from fca_dashboard.utils.path_util import get_config_path, resolve_path

                settings_path = get_config_path("settings.yml")
                with open(settings_path, "r") as file:
                    settings = yaml.safe_load(file)

                data_path = (
                    settings.get("classifier", {})
                    .get("data_paths", {})
                    .get("training_data")
                )
                if data_path:
                    # Resolve the path to ensure it exists
                    data_path = str(resolve_path(data_path))
            except ImportError:
                # Not running in fca_dashboard context
                data_path = None

            # If still no data_path, use the default in nexusml
            if not data_path:
                # Use the default path from config
                default_path = training_data_config.get(
                    "default_path", "ingest/data/eq_ids.csv"
                )
                data_path = str(Path(__file__).resolve().parent.parent / default_path)
        except Exception as e:
            print(f"Warning: Could not determine data path: {e}")
            # Use default path from config as fallback
            default_path = training_data_config.get(
                "default_path", "ingest/data/eq_ids.csv"
            )
            data_path = str(Path(__file__).resolve().parent.parent / default_path)

    # Read CSV file using pandas
    encoding = training_data_config.get("encoding", "utf-8")
    fallback_encoding = training_data_config.get("fallback_encoding", "latin1")

    try:
        df = pd.read_csv(data_path, encoding=encoding)
    except UnicodeDecodeError:
        # Try with a different encoding if the primary one fails
        print(
            f"Warning: Failed to read with {encoding} encoding. Trying {fallback_encoding}."
        )
        df = pd.read_csv(data_path, encoding=fallback_encoding)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Data file not found at {data_path}. Please provide a valid path."
        )

    # Clean up column names (remove any leading/trailing whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Fill NaN values with empty strings for text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("")

    # Verify and create required columns
    df = verify_required_columns(df, config)

    return df
