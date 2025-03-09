"""
Mapping Functions Module

This module provides mapping functions for feature engineering in the NexusML suite.
These functions are used by transformers to map values from one domain to another.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nexusml.config import get_project_root


def load_masterformat_mappings() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """
    Load MasterFormat mappings from JSON files.
    
    Returns:
        Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Primary and equipment-specific mappings
    """
    root = get_project_root()
    
    try:
        with open(root / "config" / "mappings" / "masterformat_primary.json") as f:
            primary_mapping = json.load(f)
        
        with open(root / "config" / "mappings" / "masterformat_equipment.json") as f:
            equipment_specific_mapping = json.load(f)
        
        return primary_mapping, equipment_specific_mapping
    except Exception as e:
        print(f"Warning: Could not load MasterFormat mappings: {e}")
        # Return empty mappings if files cannot be loaded
        return {}, {}


def enhanced_masterformat_mapping(
    uniformat_class: str,
    system_type: str,
    equipment_category: str,
    equipment_subcategory: Optional[str] = None,
    eav_manager: Optional[Any] = None,
) -> str:
    """
    Enhanced mapping with better handling of specialty equipment types.
    
    Args:
        uniformat_class: Uniformat classification
        system_type: System type
        equipment_category: Equipment category
        equipment_subcategory: Equipment subcategory
        eav_manager: EAV manager instance. If None, uses the one from the DI container.
    
    Returns:
        str: MasterFormat classification code
    """
    # Load mappings from JSON files
    primary_mapping, equipment_specific_mapping = load_masterformat_mappings()
    
    # Try equipment-specific mapping first
    if equipment_subcategory in equipment_specific_mapping:
        return equipment_specific_mapping[equipment_subcategory]
    
    # Then try primary mapping
    if (
        uniformat_class in primary_mapping
        and system_type in primary_mapping[uniformat_class]
    ):
        return primary_mapping[uniformat_class][system_type]
    
    # Try EAV-based mapping
    try:
        # Get EAV manager from DI container if not provided
        if eav_manager is None:
            from nexusml.core.di.provider import ContainerProvider
            
            container = ContainerProvider().container
            eav_manager = container.resolve("EAVManager")
        
        masterformat_id = eav_manager.get_classification_ids(equipment_category).get(
            "masterformat_id", ""
        )
        if masterformat_id:
            return masterformat_id
    except Exception as e:
        print(f"Warning: Could not use EAV for MasterFormat mapping: {e}")
    
    # Refined fallback mappings by Uniformat class
    fallbacks = {
        "H": "23 00 00",  # Heating, Ventilating, and Air Conditioning (HVAC)
        "P": "22 00 00",  # Plumbing
        "SM": "23 00 00",  # HVAC
        "R": "11 40 00",  # Foodservice Equipment (Refrigeration)
    }
    
    return fallbacks.get(uniformat_class, "00 00 00")  # Return unknown if no match


def map_to_omniclass(
    equipment_category: str,
    eav_manager: Optional[Any] = None,
) -> str:
    """
    Map equipment category to OmniClass ID.
    
    Args:
        equipment_category: Equipment category
        eav_manager: EAV manager instance. If None, uses the one from the DI container.
    
    Returns:
        str: OmniClass ID
    """
    try:
        # Get EAV manager from DI container if not provided
        if eav_manager is None:
            from nexusml.core.di.provider import ContainerProvider
            
            container = ContainerProvider().container
            eav_manager = container.resolve("EAVManager")
        
        omniclass_id = eav_manager.get_classification_ids(equipment_category).get(
            "omniclass_id", ""
        )
        return omniclass_id
    except Exception as e:
        print(f"Warning: Could not use EAV for OmniClass mapping: {e}")
        return ""


def map_to_uniformat(
    equipment_category: str,
    eav_manager: Optional[Any] = None,
) -> str:
    """
    Map equipment category to Uniformat ID.
    
    Args:
        equipment_category: Equipment category
        eav_manager: EAV manager instance. If None, uses the one from the DI container.
    
    Returns:
        str: Uniformat ID
    """
    try:
        # Get EAV manager from DI container if not provided
        if eav_manager is None:
            from nexusml.core.di.provider import ContainerProvider
            
            container = ContainerProvider().container
            eav_manager = container.resolve("EAVManager")
        
        uniformat_id = eav_manager.get_classification_ids(equipment_category).get(
            "uniformat_id", ""
        )
        return uniformat_id
    except Exception as e:
        print(f"Warning: Could not use EAV for Uniformat mapping: {e}")
        return ""