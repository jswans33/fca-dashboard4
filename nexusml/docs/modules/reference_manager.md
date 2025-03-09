# Module: reference_manager

## Overview

The `reference_manager` module provides a unified interface for managing reference data from multiple sources in the NexusML system. It serves as a wrapper around the more modular implementation in the reference package, re-exporting the ReferenceManager class and providing a backward compatibility function.

This module simplifies access to various reference data sources, including:

1. Classification systems (OmniClass, Uniformat, MasterFormat)
2. Glossaries and abbreviations (MCAA)
3. Manufacturer information (SMACNA)
4. Service life data (ASHRAE, Energize Denver)
5. Equipment taxonomy

The ReferenceManager class follows the Facade pattern to provide a simple interface to the complex subsystem of reference data sources, making it easier to work with reference data throughout the NexusML system.

## Functions

### `get_reference_manager(config_path=None)`

Get an instance of the ReferenceManager.

**Parameters:**

- `config_path` (optional): Path to the configuration file. If None, uses the default path.

**Returns:**

- ReferenceManager instance

**Example:**
```python
from nexusml.core.reference_manager import get_reference_manager

# Get a reference manager with default configuration
ref_manager = get_reference_manager()

# Get a reference manager with custom configuration
ref_manager = get_reference_manager("path/to/custom_config.yml")
```

**Notes:**

- This function is provided for backward compatibility
- It simply creates and returns a new instance of the ReferenceManager class

## Imported Classes

### Class: ReferenceManager

Unified manager for all reference data sources.

#### Attributes

- `config` (Dict[str, Any]): Configuration dictionary
- `base_path` (Path): Base path for resolving relative paths
- `omniclass` (OmniClassDataSource): OmniClass data source
- `uniformat` (UniformatDataSource): Uniformat data source
- `masterformat` (MasterFormatDataSource): MasterFormat data source
- `mcaa_glossary` (MCAAGlossaryDataSource): MCAA glossary data source
- `mcaa_abbreviations` (MCAAAbbrDataSource): MCAA abbreviations data source
- `smacna` (SMACNADataSource): SMACNA data source
- `ashrae` (ASHRAEDataSource): ASHRAE data source
- `energize_denver` (EnergizeDenverDataSource): Energize Denver data source
- `equipment_taxonomy` (EquipmentTaxonomyDataSource): Equipment taxonomy data source
- `data_sources` (List[ReferenceDataSource]): List of all data sources for batch operations

#### Methods

##### `__init__(config_path: Optional[str] = None)`

Initialize the reference manager.

**Parameters:**

- `config_path` (Optional[str], optional): Path to the reference configuration file. If None, uses the default path.

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager with default configuration
ref_manager = ReferenceManager()

# Create a reference manager with custom configuration
ref_manager = ReferenceManager("path/to/custom_config.yml")
```

##### `load_all() -> None`

Load all reference data sources.

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager
ref_manager = ReferenceManager()

# Load all reference data sources
ref_manager.load_all()
```

##### `get_omniclass_description(code: str) -> Optional[str]`

Get the OmniClass description for a code.

**Parameters:**

- `code` (str): OmniClass code

**Returns:**

- Optional[str]: Description or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get OmniClass description
code = "23-13 11 11"
description = ref_manager.get_omniclass_description(code)
print(f"OmniClass {code}: {description}")
```

##### `get_uniformat_description(code: str) -> Optional[str]`

Get the Uniformat description for a code.

**Parameters:**

- `code` (str): Uniformat code

**Returns:**

- Optional[str]: Description or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get Uniformat description
code = "D3010"
description = ref_manager.get_uniformat_description(code)
print(f"Uniformat {code}: {description}")
```

##### `find_uniformat_codes_by_keyword(keyword: str, max_results: int = 10) -> List[Dict[str, str]]`

Find Uniformat codes by keyword.

**Parameters:**

- `keyword` (str): Keyword to search for
- `max_results` (int, optional): Maximum number of results to return. Default is 10.

**Returns:**

- List[Dict[str, str]]: List of dictionaries with Uniformat code, title, and MasterFormat number

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Find Uniformat codes by keyword
keyword = "chiller"
results = ref_manager.find_uniformat_codes_by_keyword(keyword, max_results=5)
for result in results:
    print(f"Uniformat {result['uniformat_code']}: {result['title']}")
```

##### `get_masterformat_description(code: str) -> Optional[str]`

Get the MasterFormat description for a code.

**Parameters:**

- `code` (str): MasterFormat code

**Returns:**

- Optional[str]: Description or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get MasterFormat description
code = "23 64 23"
description = ref_manager.get_masterformat_description(code)
print(f"MasterFormat {code}: {description}")
```

##### `get_term_definition(term: str) -> Optional[str]`

Get the definition for a term from the MCAA glossary.

**Parameters:**

- `term` (str): Term to look up

**Returns:**

- Optional[str]: Definition or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get term definition
term = "HVAC"
definition = ref_manager.get_term_definition(term)
print(f"{term}: {definition}")
```

##### `get_abbreviation_meaning(abbr: str) -> Optional[str]`

Get the meaning of an abbreviation from the MCAA abbreviations.

**Parameters:**

- `abbr` (str): Abbreviation to look up

**Returns:**

- Optional[str]: Meaning or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get abbreviation meaning
abbr = "AHU"
meaning = ref_manager.get_abbreviation_meaning(abbr)
print(f"{abbr}: {meaning}")
```

##### `find_manufacturers_by_product(product: str) -> List[Dict[str, Any]]`

Find manufacturers that produce a specific product.

**Parameters:**

- `product` (str): Product description or keyword

**Returns:**

- List[Dict[str, Any]]: List of manufacturer information dictionaries

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Find manufacturers by product
product = "chiller"
manufacturers = ref_manager.find_manufacturers_by_product(product)
for manufacturer in manufacturers:
    print(f"Manufacturer: {manufacturer['name']}")
```

##### `get_service_life(equipment_type: str) -> Dict[str, Any]`

Get service life information for an equipment type.

**Parameters:**

- `equipment_type` (str): Equipment type description

**Returns:**

- Dict[str, Any]: Dictionary with service life information

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get service life information
equipment_type = "Chiller"
service_life = ref_manager.get_service_life(equipment_type)
print(f"Service life for {equipment_type}:")
print(f"  Median: {service_life.get('median_years')} years")
print(f"  Range: {service_life.get('min_years')} - {service_life.get('max_years')} years")
print(f"  Source: {service_life.get('source')}")
```

**Notes:**

- This method tries to find service life information from multiple sources in the following order:
  1. ASHRAE
  2. Energize Denver
  3. Equipment Taxonomy
- It returns the first non-default result found

##### `get_equipment_info(equipment_type: str) -> Optional[Dict[str, Any]]`

Get detailed information about an equipment type.

**Parameters:**

- `equipment_type` (str): Equipment type description

**Returns:**

- Optional[Dict[str, Any]]: Dictionary with equipment information or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get equipment information
equipment_type = "Chiller"
info = ref_manager.get_equipment_info(equipment_type)
if info:
    print(f"Equipment information for {equipment_type}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
```

##### `get_equipment_maintenance_hours(equipment_type: str) -> Optional[float]`

Get maintenance hours for an equipment type.

**Parameters:**

- `equipment_type` (str): Equipment type description

**Returns:**

- Optional[float]: Maintenance hours or None if not found

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get maintenance hours
equipment_type = "Chiller"
hours = ref_manager.get_equipment_maintenance_hours(equipment_type)
print(f"Maintenance hours for {equipment_type}: {hours}")
```

##### `get_equipment_by_category(category: str) -> List[Dict[str, Any]]`

Get all equipment in a specific category.

**Parameters:**

- `category` (str): Asset category

**Returns:**

- List[Dict[str, Any]]: List of equipment dictionaries

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get equipment by category
category = "HVAC"
equipment_list = ref_manager.get_equipment_by_category(category)
print(f"Equipment in category {category}:")
for equipment in equipment_list:
    print(f"  {equipment.get('Equip Name ID')}")
```

##### `get_equipment_by_system(system_type: str) -> List[Dict[str, Any]]`

Get all equipment in a specific system type.

**Parameters:**

- `system_type` (str): System type

**Returns:**

- List[Dict[str, Any]]: List of equipment dictionaries

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Get equipment by system
system_type = "Cooling"
equipment_list = ref_manager.get_equipment_by_system(system_type)
print(f"Equipment in system {system_type}:")
for equipment in equipment_list:
    print(f"  {equipment.get('Equip Name ID')}")
```

##### `validate_data() -> Dict[str, Dict[str, Any]]`

Validate all reference data sources to ensure data quality.

**Returns:**

- Dict[str, Dict[str, Any]]: Dictionary with validation results for each data source

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager

# Create a reference manager
ref_manager = ReferenceManager()

# Validate all reference data
validation_results = ref_manager.validate_data()
for source, result in validation_results.items():
    print(f"Validation results for {source}:")
    print(f"  Valid: {result.get('valid', False)}")
    if 'errors' in result and result['errors']:
        print(f"  Errors: {len(result['errors'])}")
        for error in result['errors'][:3]:  # Show first 3 errors
            print(f"    - {error}")
```

**Notes:**

- This method checks:
  1. If data is loaded
  2. If required columns exist
  3. If data has the expected structure
  4. Basic data quality checks (nulls, duplicates, etc.)
- It loads data if not already loaded

##### `enrich_equipment_data(df: pd.DataFrame) -> pd.DataFrame`

Enrich equipment data with reference information.

**Parameters:**

- `df` (pd.DataFrame): DataFrame with equipment data

**Returns:**

- pd.DataFrame: Enriched DataFrame

**Example:**
```python
from nexusml.core.reference.manager import ReferenceManager
import pandas as pd

# Create a reference manager and load data
ref_manager = ReferenceManager()
ref_manager.load_all()

# Create a DataFrame with equipment data
data = pd.DataFrame({
    "equipment_type": ["Chiller", "Air Handler", "Pump"],
    "equipment_name": ["Centrifugal Chiller", "VAV AHU", "Circulation Pump"],
    "uniformat_code": ["D3010", None, None],
    "masterformat_code": ["23 64 23", None, None]
})

# Enrich the data
enriched_data = ref_manager.enrich_equipment_data(data)
print(enriched_data)
```

**Notes:**

- This method adds the following information if the corresponding columns exist:
  - OmniClass descriptions
  - Uniformat descriptions
  - MasterFormat descriptions
  - Service life information
  - Maintenance hours
  - Equipment taxonomy information
- It also tries to find missing classification codes based on equipment names

## Usage Examples

### Basic Usage

```python
from nexusml.core.reference_manager import get_reference_manager

# Get a reference manager
ref_manager = get_reference_manager()

# Load all reference data
ref_manager.load_all()

# Get OmniClass description
omniclass_code = "23-13 11 11"
omniclass_description = ref_manager.get_omniclass_description(omniclass_code)
print(f"OmniClass {omniclass_code}: {omniclass_description}")

# Get Uniformat description
uniformat_code = "D3010"
uniformat_description = ref_manager.get_uniformat_description(uniformat_code)
print(f"Uniformat {uniformat_code}: {uniformat_description}")

# Get MasterFormat description
masterformat_code = "23 64 23"
masterformat_description = ref_manager.get_masterformat_description(masterformat_code)
print(f"MasterFormat {masterformat_code}: {masterformat_description}")
```

### Finding Classification Codes by Keyword

```python
from nexusml.core.reference_manager import get_reference_manager

# Get a reference manager
ref_manager = get_reference_manager()
ref_manager.load_all()

# Find Uniformat codes by keyword
keyword = "chiller"
results = ref_manager.find_uniformat_codes_by_keyword(keyword, max_results=5)
print(f"Uniformat codes for '{keyword}':")
for result in results:
    print(f"  {result['uniformat_code']}: {result['title']}")
    if result.get('masterformat_code'):
        print(f"    MasterFormat: {result['masterformat_code']}")
```

### Getting Service Life Information

```python
from nexusml.core.reference_manager import get_reference_manager

# Get a reference manager
ref_manager = get_reference_manager()
ref_manager.load_all()

# Get service life information for different equipment types
equipment_types = ["Chiller", "Air Handler", "Pump", "Boiler"]
for equipment_type in equipment_types:
    service_life = ref_manager.get_service_life(equipment_type)
    print(f"Service life for {equipment_type}:")
    print(f"  Median: {service_life.get('median_years')} years")
    print(f"  Range: {service_life.get('min_years')} - {service_life.get('max_years')} years")
    print(f"  Source: {service_life.get('source')}")
```

### Enriching Equipment Data

```python
from nexusml.core.reference_manager import get_reference_manager
import pandas as pd

# Get a reference manager
ref_manager = get_reference_manager()
ref_manager.load_all()

# Create a DataFrame with equipment data
data = pd.DataFrame({
    "equipment_type": ["Chiller", "Air Handler", "Pump"],
    "equipment_name": ["Centrifugal Chiller", "VAV AHU", "Circulation Pump"],
    "uniformat_code": ["D3010", None, None],
    "masterformat_code": ["23 64 23", None, None]
})

# Enrich the data
enriched_data = ref_manager.enrich_equipment_data(data)

# Display the enriched data
print("Enriched Data:")
print(enriched_data)

# Show added columns
new_columns = [col for col in enriched_data.columns if col not in data.columns]
print("\nNew columns added:")
print(new_columns)

# Show service life information
print("\nService Life Information:")
for i, row in enriched_data.iterrows():
    print(f"{row['equipment_type']}:")
    print(f"  Median: {row['service_life_median']} years")
    print(f"  Range: {row['service_life_min']} - {row['service_life_max']} years")
    print(f"  Source: {row['service_life_source']}")
```

### Validating Reference Data

```python
from nexusml.core.reference_manager import get_reference_manager

# Get a reference manager
ref_manager = get_reference_manager()

# Validate all reference data
validation_results = ref_manager.validate_data()

# Print summary of validation results
print("Validation Results Summary:")
for source, result in validation_results.items():
    valid = result.get('valid', False)
    error_count = len(result.get('errors', []))
    warning_count = len(result.get('warnings', []))
    
    status = "✅ Valid" if valid else "❌ Invalid"
    print(f"{source}: {status}")
    if error_count > 0:
        print(f"  Errors: {error_count}")
    if warning_count > 0:
        print(f"  Warnings: {warning_count}")

# Print detailed results for invalid sources
print("\nDetailed Results for Invalid Sources:")
for source, result in validation_results.items():
    if not result.get('valid', False):
        print(f"\n{source}:")
        if 'errors' in result and result['errors']:
            print("  Errors:")
            for error in result['errors']:
                print(f"    - {error}")
        if 'warnings' in result and result['warnings']:
            print("  Warnings:")
            for warning in result['warnings']:
                print(f"    - {warning}")
```

## Dependencies

- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **pandas**: Used for DataFrame operations
- **yaml**: Used for loading YAML configuration
- **nexusml.core.reference.base**: Used for ReferenceDataSource base class
- **nexusml.core.reference.classification**: Used for classification data sources
- **nexusml.core.reference.equipment**: Used for equipment taxonomy data source
- **nexusml.core.reference.glossary**: Used for glossary data sources
- **nexusml.core.reference.manufacturer**: Used for manufacturer data sources
- **nexusml.core.reference.service_life**: Used for service life data sources
- **nexusml.core.reference.validation**: Used for data validation functions

## Notes and Warnings

- The reference_manager module is a wrapper around the more modular implementation in the reference package
- The get_reference_manager function is provided for backward compatibility
- The ReferenceManager class follows the Facade pattern to provide a simple interface to the complex subsystem of reference data sources
- The default configuration path is "config/reference_config.yml" relative to the project root
- If the configuration file cannot be loaded, an empty dictionary is used
- The base path for resolving relative paths is the parent directory of the project root
- The ReferenceManager loads data lazily - data is only loaded when needed
- The validate_data method loads all data sources if they are not already loaded
- The enrich_equipment_data method adds various reference information to equipment data, but only if the corresponding columns exist
