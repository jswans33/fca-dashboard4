"""
Mappers module for the FCA Dashboard application.

This module provides mapping functionality between different data formats,
particularly for mapping extracted data to staging and master database schemas.
"""

from fca_dashboard.mappers.base_mapper import BaseMapper, MappingError
from fca_dashboard.mappers.mapper_factory import MapperFactory, MapperFactoryError, mapper_factory
from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.mappers.wichita_mapper import WichitaMapper

__all__ = [
    'BaseMapper',
    'MappingError',
    'MedtronicsMapper',
    'WichitaMapper',
    'MapperFactory',
    'mapper_factory',
    'MapperFactoryError'
]