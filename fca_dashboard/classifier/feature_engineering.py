"""
Feature Engineering Module

This module handles feature engineering for the equipment classification model.
It follows the Single Responsibility Principle by focusing solely on feature transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin


def enhance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with hierarchical structure and more granular categories
    
    Args:
        df (pd.DataFrame): Input dataframe with raw features
        
    Returns:
        pd.DataFrame: DataFrame with enhanced features
    """
    # Extract primary classification columns
    df['Equipment_Category'] = df['Asset Category']
    df['Uniformat_Class'] = df['System Type ID']
    df['System_Type'] = df['Precon System']
    
    # Create subcategory field for more granular classification
    df['Equipment_Subcategory'] = df['Equip Name ID']
    
    # Combine fields for rich text features
    df['combined_features'] = (
        df['Asset Category'] + ' ' + 
        df['Equip Name ID'] + ' ' + 
        df['Sub System Type'] + ' ' + 
        df['Sub System ID'] + ' ' + 
        df['Title'] + ' ' + 
        df['Precon System'] + ' ' + 
        df['Operations System'] + ' ' +
        df['Sub System Class'] + ' ' +
        df['Drawing Abbreviation']
    )
    
    # Add equipment size and unit as features
    df['size_feature'] = df['Equipment Size'].astype(str) + ' ' + df['Unit'].astype(str)
    
    # Add service life as a feature
    df['service_life'] = df['Service Life'].fillna(0).astype(float)
    
    # Fill NaN values
    df['combined_features'] = df['combined_features'].fillna('')
    
    return df


def create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create hierarchical category structure to better handle "Other" categories
    
    Args:
        df (pd.DataFrame): Input dataframe with basic features
        
    Returns:
        pd.DataFrame: DataFrame with hierarchical category features
    """
    # Create Equipment Type - a more detailed category than Equipment_Category
    df['Equipment_Type'] = df['Asset Category'] + '-' + df['Equip Name ID']
    
    # Create System Subtype - a more detailed category than System_Type
    df['System_Subtype'] = df['Precon System'] + '-' + df['Operations System']
    
    return df


def enhanced_masterformat_mapping(uniformat_class: str, system_type: str, equipment_category: str, equipment_subcategory: Optional[str] = None) -> str:
    """
    Enhanced mapping with better handling of specialty equipment types
    
    Args:
        uniformat_class (str): Uniformat classification
        system_type (str): System type
        equipment_category (str): Equipment category
        equipment_subcategory (Optional[str]): Equipment subcategory
        
    Returns:
        str: MasterFormat classification code
    """
    # Primary mapping
    primary_mapping = {
        'H': {
            'Chiller Plant': '23 64 00',  # Commercial Water Chillers
            'Cooling Tower Plant': '23 65 00',  # Cooling Towers
            'Heating Water Boiler Plant': '23 52 00',  # Heating Boilers
            'Steam Boiler Plant': '23 52 33',  # Steam Heating Boilers
            'Air Handling Units': '23 73 00',  # Indoor Central-Station Air-Handling Units
        },
        'P': {
            'Domestic Water Plant': '22 11 00',  # Facility Water Distribution
            'Medical/Lab Gas Plant': '22 63 00',  # Gas Systems for Laboratory and Healthcare Facilities
            'Sanitary Equipment': '22 13 00',  # Facility Sanitary Sewerage
        },
        'SM': {
            'Air Handling Units': '23 74 00',  # Packaged Outdoor HVAC Equipment
            'SM Accessories': '23 33 00',  # Air Duct Accessories
            'SM Equipment': '23 30 00',  # HVAC Air Distribution
        }
    }
    
    # Secondary mapping for specific equipment types that were in "Other"
    equipment_specific_mapping = {
        'Heat Exchanger': '23 57 00',  # Heat Exchangers for HVAC
        'Water Softener': '22 31 00',  # Domestic Water Softeners
        'Humidifier': '23 84 13',  # Humidifiers
        'Radiant Panel': '23 83 16',  # Radiant-Heating Hydronic Piping
        'Make-up Air Unit': '23 74 23',  # Packaged Outdoor Heating-Only Makeup Air Units
        'Energy Recovery Ventilator': '23 72 00',  # Air-to-Air Energy Recovery Equipment
        'DI/RO Equipment': '22 31 16',  # Deionized-Water Piping
        'Bypass Filter Feeder': '23 25 00',  # HVAC Water Treatment
        'Grease Interceptor': '22 13 23',  # Sanitary Waste Interceptors
        'Heat Trace': '23 05 33',  # Heat Tracing for HVAC Piping
        'Dust Collector': '23 35 16',  # Engine Exhaust Systems
        'Venturi VAV Box': '23 36 00',  # Air Terminal Units
        'Water Treatment Controller': '23 25 13',  # Water Treatment for Closed-Loop Hydronic Systems
        'Polishing System': '23 25 00',  # HVAC Water Treatment
        'Ozone Generator': '22 67 00',  # Processed Water Systems for Laboratory and Healthcare Facilities
    }
    
    # Try equipment-specific mapping first
    if equipment_subcategory in equipment_specific_mapping:
        return equipment_specific_mapping[equipment_subcategory]
    
    # Then try primary mapping
    if uniformat_class in primary_mapping and system_type in primary_mapping[uniformat_class]:
        return primary_mapping[uniformat_class][system_type]
    
    # Refined fallback mappings by Uniformat class
    fallbacks = {
        'H': '23 00 00',  # Heating, Ventilating, and Air Conditioning (HVAC)
        'P': '22 00 00',  # Plumbing
        'SM': '23 00 00',  # HVAC
        'R': '11 40 00',  # Foodservice Equipment (Refrigeration)
    }
    
    return fallbacks.get(uniformat_class, '00 00 00')  # Return unknown if no match