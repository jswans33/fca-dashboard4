"""
Medtronics mapper module for the FCA Dashboard application.

This module provides a mapper for Medtronics data to the staging schema.
"""

import json
from typing import Any, Dict, Optional

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.mappers.base_mapper import BaseMapper, MappingError


class MedtronicsMapper(BaseMapper):
    """
    Mapper for Medtronics data to the staging schema.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the Medtronics mapper.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
        """
        super().__init__(logger)
        self._column_mapping = self._load_column_mapping()
    
    def _load_column_mapping(self) -> Dict[str, str]:
        """
        Load the column mapping from settings.
        
        Returns:
            A dictionary mapping source columns to destination columns.
            
        Raises:
            MappingError: If the mapping configuration is not found.
        """
        # Try to get mapping from settings
        mappers_config = settings.get("mappers", {})
        medtronics_config = mappers_config.get("medtronics", {})
        mapping_config = medtronics_config.get("column_mapping", {})
        
        # If mapping_config is empty, use default mapping
        if not mapping_config:
            self.logger.warning("Medtronics mapping not found in settings, using default mapping")
            mapping_config = {
                "asset_name": "equipment_type",
                "asset_tag": "equipment_tag",
                "model_number": "model",
                "serial_number": "serial_number",
                "system_category": "category_name",
                "sub_system_type": "mcaa_subsystem_type",
                "sub_system_classification": "mcaa_subsystem_classification",
                "date_installed": "install_date",
                "room_number": "room",
                "size": "capacity",
                "floor": "floor",
                "area": "other_location_info"
            }
        
        return mapping_config
    
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get the column mapping from source to destination.
        
        Returns:
            A dictionary mapping source columns to destination columns.
        """
        return self._column_mapping
    
    def map_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map a DataFrame from Medtronics format to staging format.
        
        Args:
            df: The source DataFrame to map.
            
        Returns:
            The mapped DataFrame.
            
        Raises:
            MappingError: If an error occurs during mapping.
        """
        try:
            # Make a copy of the DataFrame to avoid modifying the original
            staging_df = df.copy()
            
            # Fix column names by replacing spaces with underscores
            staging_df.columns = [col.replace(' ', '_') if isinstance(col, str) else col for col in staging_df.columns]
            self.logger.info(f"Normalized column names: {list(staging_df.columns)}")
            
            # Create a new DataFrame with the mapped columns
            mapped_df = pd.DataFrame()
            
            # Map columns according to the mapping
            for src_col, dest_col in self._column_mapping.items():
                if src_col in staging_df.columns:
                    mapped_df[dest_col] = staging_df[src_col]
            
            # Handle special columns that need to be converted to JSON
            json_columns = ["motor_hp", "estimated_operating_hours", "notes"]
            attributes = {}
            
            for col in json_columns:
                if col in staging_df.columns:
                    # Add to attributes dictionary
                    attributes[col] = staging_df[col].to_dict()
            
            # If we have attributes, add them as a JSON column
            if attributes:
                mapped_df["attributes"] = [json.dumps({"medtronics_attributes": attributes}) for _ in range(len(mapped_df))]
            
            return mapped_df
            
        except Exception as e:
            error_msg = f"Error mapping Medtronics data: {str(e)}"
            self.logger.error(error_msg)
            raise MappingError(error_msg) from e