"""
Reference Data Management Package

This package provides a modular approach to managing reference data from multiple sources:
- OmniClass taxonomy
- Uniformat classification
- MasterFormat classification
- MCAA abbreviations and glossary
- SMACNA manufacturer data
- ASHRAE service life data
- Energize Denver service life data
"""

from nexusml.core.reference.base import ReferenceDataSource
from nexusml.core.reference.classification import (
    ClassificationDataSource,
    MasterFormatDataSource,
    OmniClassDataSource,
    UniformatDataSource,
)
from nexusml.core.reference.glossary import (
    GlossaryDataSource,
    MCAAAbbrDataSource,
    MCAAGlossaryDataSource,
)
from nexusml.core.reference.manager import ReferenceManager
from nexusml.core.reference.manufacturer import (
    ManufacturerDataSource,
    SMACNADataSource,
)
from nexusml.core.reference.service_life import (
    ASHRAEDataSource,
    EnergizeDenverDataSource,
    ServiceLifeDataSource,
)

__all__ = [
    "ReferenceDataSource",
    "ClassificationDataSource",
    "OmniClassDataSource",
    "UniformatDataSource",
    "MasterFormatDataSource",
    "GlossaryDataSource",
    "MCAAGlossaryDataSource",
    "MCAAAbbrDataSource",
    "ManufacturerDataSource",
    "SMACNADataSource",
    "ServiceLifeDataSource",
    "ASHRAEDataSource",
    "EnergizeDenverDataSource",
    "ReferenceManager",
]
