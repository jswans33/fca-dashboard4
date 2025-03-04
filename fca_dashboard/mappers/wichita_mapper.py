"""
Wichita mapper module for the FCA Dashboard application.

This module provides a mapper for Wichita Animal Shelter data to the staging schema.
"""

import json
from typing import Any, Dict, Optional

import pandas as pd

from fca_dashboard.config.settings import settings
from fca_dashboard.mappers.base_mapper import BaseMapper, MappingError


class WichitaMapper(BaseMapper):
    """
    Mapper for Wichita Animal Shelter data to the staging schema.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the Wichita mapper.
        
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
        mapping_config = settings.get("mappers.wichita.column_mapping")
        
        if not mapping_config:
            # If not found in settings, use default mapping
            self.logger.warning("Wichita mapping not found in settings, using default mapping")
            return {
                "Asset_Name": "equipment_type",
                "Asset_Tag": "equipment_tag",
                "Manufacturer": "manufacturer",
                "Model": "model",
                "Serial_Number": "serial_number",
                "Asset_Category_Name": "category_name",
                "Asset_Type": "equipment_type",
                "Location": "other_location_info",
                "Install_Date": "install_date"
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
        Map a DataFrame from Wichita format to staging format.
        
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
                elif src_col.replace('_', ' ') in df.columns:
                    # Try with spaces instead of underscores
                    original_col = src_col.replace('_', ' ')
                    mapped_df[dest_col] = df[original_col]
            
            # Handle any additional attributes
            attributes = {}
            
            # If we have attributes, add them as a JSON column
            if attributes:
                mapped_df["attributes"] = [json.dumps({"wichita_attributes": attributes}) for _ in range(len(mapped_df))]
            
            return mapped_df
            
        except Exception as e:
            error_msg = f"Error mapping Wichita data: {str(e)}"
            self.logger.error(error_msg)
            raise MappingError(error_msg) from e