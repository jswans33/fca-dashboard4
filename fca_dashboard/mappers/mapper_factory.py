"""
Mapper factory module for the FCA Dashboard application.

This module provides a factory for creating mappers based on the source system.
"""

from typing import Dict, Optional, Type

from fca_dashboard.config.settings import settings
from fca_dashboard.mappers.base_mapper import BaseMapper, MappingError
from fca_dashboard.mappers.medtronics_mapper import MedtronicsMapper
from fca_dashboard.mappers.wichita_mapper import WichitaMapper
from fca_dashboard.utils.error_handler import DataTransformationError
from fca_dashboard.utils.logging_config import get_logger


class MapperFactoryError(DataTransformationError):
    """Exception raised for errors in the mapper factory."""
    pass


class MapperFactory:
    """
    Factory for creating mappers based on the source system.
    
    This class follows the Factory pattern to create the appropriate mapper
    for a given source system.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the mapper factory.
        
        Args:
            logger: Optional logger instance. If None, a default logger will be created.
        """
        self.logger = logger or get_logger("mapper_factory")
        self._mapper_registry = self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[str, Type[BaseMapper]]:
        """
        Initialize the mapper registry.
        
        Returns:
            A dictionary mapping source system names to mapper classes.
        """
        # Register known mappers
        registry = {
            "Medtronics": MedtronicsMapper,
            "Wichita": WichitaMapper
        }
        
        # Add any additional mappers from settings
        mapper_settings = settings.get("mappers.registry", {})
        if mapper_settings:
            for source_system, mapper_class_path in mapper_settings.items():
                try:
                    # Import the mapper class dynamically
                    module_path, class_name = mapper_class_path.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    mapper_class = getattr(module, class_name)
                    
                    # Register the mapper
                    registry[source_system] = mapper_class
                    self.logger.info(f"Registered mapper for {source_system}: {mapper_class_path}")
                except (ImportError, AttributeError) as e:
                    self.logger.error(f"Error registering mapper for {source_system}: {str(e)}")
        return registry
    
    def register_mapper(self, source_system: str, mapper_class: Type[BaseMapper]) -> None:
        """
        Register a mapper for a source system.
        
        Args:
            source_system: The source system identifier.
            mapper_class: The mapper class to register.
        """
        self.logger.info(f"Registering mapper for {source_system}: {mapper_class.__name__}")
        self._mapper_registry[source_system] = mapper_class
    
    def create_mapper(self, source_system: str, logger=None) -> BaseMapper:
        """
        Create a mapper for the specified source system.
        
        Args:
            source_system: The source system identifier.
            logger: Optional logger instance to pass to the mapper.
            
        Returns:
            A mapper instance for the specified source system.
            
        Raises:
            MapperFactoryError: If no mapper is registered for the source system.
        """
        if source_system not in self._mapper_registry:
            error_msg = f"No mapper registered for source system: {source_system}"
            self.logger.error(error_msg)
            raise MapperFactoryError(error_msg)
        
        mapper_class = self._mapper_registry[source_system]
        return mapper_class(logger)


# Create a singleton instance of the mapper factory
mapper_factory = MapperFactory()