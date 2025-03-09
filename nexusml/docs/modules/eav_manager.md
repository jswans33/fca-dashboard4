# Module: eav_manager

## Overview

The `eav_manager` module manages the Entity-Attribute-Value (EAV) structure for equipment attributes in the NexusML system. This module provides functionality to:

1. Load attribute templates for different equipment types
2. Validate equipment attributes against templates
3. Generate attribute templates for equipment based on ML predictions
4. Fill in missing attributes using ML predictions and rules
5. Transform feature sets by adding EAV attributes

The EAV pattern is particularly useful for handling equipment with varying attributes, allowing for flexible data storage and retrieval while maintaining structure.

## Classes

### Class: EAVManager

Manages the Entity-Attribute-Value (EAV) structure for equipment attributes.

#### Attributes

- `templates_path` (str): Path to the JSON file containing attribute templates
- `templates` (Dict): Dictionary of loaded attribute templates

#### Methods

##### `__init__(templates_path: Optional[str] = None)`

Initialize the EAV Manager with templates.

**Parameters:**
- `templates_path` (Optional[str]): Path to the JSON file containing attribute templates. If None, uses the default path.

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

# Create EAV manager with default templates
eav_manager = EAVManager()

# Create EAV manager with custom templates
custom_eav_manager = EAVManager("path/to/custom_templates.json")
```

##### `load_templates() -> None`

Load attribute templates from the JSON file.

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()
# Reload templates if needed
eav_manager.load_templates()
```

**Notes:**
- This method is called automatically during initialization
- The default templates path is `nexusml/config/eav/equipment_attributes.json`
- If the templates file cannot be loaded, an empty dictionary is used

##### `get_equipment_template(equipment_type: str) -> Dict[str, Any]`

Get the attribute template for a specific equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment (e.g., "Chiller", "Air Handler")

**Returns:**
- Dict[str, Any]: Dictionary containing the attribute template, or an empty dict if not found

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get template for a specific equipment type
chiller_template = eav_manager.get_equipment_template("Chiller")
print(chiller_template)

# Template matching is flexible
centrifugal_chiller_template = eav_manager.get_equipment_template("Centrifugal Chiller")
print(centrifugal_chiller_template)  # Will match "Chiller" template
```

**Notes:**
- The method tries to find a match in the following order:
  1. Exact match
  2. Case-insensitive match
  3. Partial match (e.g., "Centrifugal Chiller" should match "Chiller")
- Returns an empty dictionary if no match is found

##### `get_required_attributes(equipment_type: str) -> List[str]`

Get required attributes for a given equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- List[str]: List of required attribute names

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get required attributes for a chiller
required_attrs = eav_manager.get_required_attributes("Chiller")
print(f"Required attributes for Chiller: {required_attrs}")
```

##### `get_optional_attributes(equipment_type: str) -> List[str]`

Get optional attributes for a given equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- List[str]: List of optional attribute names

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get optional attributes for a chiller
optional_attrs = eav_manager.get_optional_attributes("Chiller")
print(f"Optional attributes for Chiller: {optional_attrs}")
```

##### `get_all_attributes(equipment_type: str) -> List[str]`

Get all attributes (required and optional) for a given equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- List[str]: List of all attribute names

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get all attributes for a chiller
all_attrs = eav_manager.get_all_attributes("Chiller")
print(f"All attributes for Chiller: {all_attrs}")
```

##### `get_attribute_unit(equipment_type: str, attribute: str) -> str`

Get the unit for a specific attribute of an equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment
- `attribute` (str): The attribute name

**Returns:**
- str: Unit string, or empty string if not found

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get unit for a specific attribute
unit = eav_manager.get_attribute_unit("Chiller", "Capacity")
print(f"Unit for Chiller Capacity: {unit}")  # e.g., "tons"
```

##### `get_classification_ids(equipment_type: str) -> Dict[str, str]`

Get the classification IDs (OmniClass, MasterFormat, Uniformat) for an equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- Dict[str, str]: Dictionary with classification IDs

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get classification IDs for a chiller
classification_ids = eav_manager.get_classification_ids("Chiller")
print(f"Classification IDs for Chiller: {classification_ids}")
```

**Notes:**
- Returns a dictionary with the following keys:
  - `omniclass_id`: OmniClass classification ID
  - `masterformat_id`: MasterFormat classification ID
  - `uniformat_id`: Uniformat classification ID
- If a classification ID is not found, an empty string is returned for that key

##### `get_performance_fields(equipment_type: str) -> Dict[str, Dict[str, Any]]`

Get the performance fields for an equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- Dict[str, Dict[str, Any]]: Dictionary with performance fields

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Get performance fields for a chiller
performance_fields = eav_manager.get_performance_fields("Chiller")
print(f"Performance fields for Chiller: {performance_fields}")
```

**Notes:**
- Performance fields typically include information like service life, maintenance interval, etc.
- Each performance field is a dictionary with additional information like default values, units, etc.

##### `validate_attributes(equipment_type: str, attributes: Dict[str, Any]) -> Dict[str, List[str]]`

Validate attributes against the template for an equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment
- `attributes` (Dict[str, Any]): Dictionary of attribute name-value pairs

**Returns:**
- Dict[str, List[str]]: Dictionary with validation results:
  - `missing_required`: List of missing required attributes
  - `unknown`: List of attributes not in the template

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Define some attributes for a chiller
chiller_attrs = {
    "Capacity": 500,
    "EER": 10.5,
    "Refrigerant": "R-134a"
}

# Validate attributes against the template
validation_results = eav_manager.validate_attributes("Chiller", chiller_attrs)
print(f"Validation results: {validation_results}")

# Check if there are any missing required attributes
if validation_results["missing_required"]:
    print(f"Missing required attributes: {validation_results['missing_required']}")

# Check if there are any unknown attributes
if validation_results["unknown"]:
    print(f"Unknown attributes: {validation_results['unknown']}")
```

##### `generate_attribute_template(equipment_type: str) -> Dict[str, Any]`

Generate an attribute template for an equipment type.

**Parameters:**
- `equipment_type` (str): The type of equipment

**Returns:**
- Dict[str, Any]: Dictionary with attribute template

**Example:**
```python
from nexusml.core.eav_manager import EAVManager

eav_manager = EAVManager()

# Generate attribute template for a chiller
template = eav_manager.generate_attribute_template("Chiller")
print(f"Generated template for Chiller: {template}")
```

**Notes:**
- The generated template includes:
  - `equipment_type`: The type of equipment
  - `classification`: Classification IDs (OmniClass, MasterFormat, Uniformat)
  - `required_attributes`: Dictionary of required attributes with units
  - `optional_attributes`: Dictionary of optional attributes with units
  - `performance_fields`: Dictionary of performance fields
- If no template is found for the equipment type, returns a dictionary with an error message

##### `fill_missing_attributes(equipment_type: str, attributes: Dict[str, Any], description: str, model=None) -> Dict[str, Any]`

Fill in missing attributes using ML predictions and rules.

**Parameters:**
- `equipment_type` (str): The type of equipment
- `attributes` (Dict[str, Any]): Dictionary of existing attribute name-value pairs
- `description` (str): Text description of the equipment
- `model` (optional): Optional ML model to use for predictions

**Returns:**
- Dict[str, Any]: Dictionary with filled attributes

**Example:**
```python
from nexusml.core.eav_manager import EAVManager
from nexusml.core.model import EquipmentClassifier

eav_manager = EAVManager()

# Define some attributes for a chiller (with missing attributes)
chiller_attrs = {
    "Capacity": 500,
    "EER": None,  # Missing value
    "Refrigerant": None  # Missing value
}

# Load a model for predicting attributes
model = EquipmentClassifier()
model.load_model("path/to/model.pkl")

# Fill missing attributes
description = "500 ton centrifugal chiller with R-134a refrigerant"
filled_attrs = eav_manager.fill_missing_attributes(
    "Chiller", chiller_attrs, description, model
)
print(f"Filled attributes: {filled_attrs}")
```

**Notes:**
- The method first identifies missing attributes (those not in the attributes dictionary or with None values)
- It fills in performance fields from template defaults
- If a model is provided and it has a `predict_attributes` method, it uses the model to predict missing attributes
- Returns a copy of the original attributes dictionary with missing attributes filled in

### Class: EAVTransformer

Transformer that adds EAV attributes to the feature set. This class follows the scikit-learn transformer interface.

#### Attributes

- `eav_manager` (EAVManager): Instance of EAVManager used for attribute management

#### Methods

##### `__init__(eav_manager: Optional[EAVManager] = None)`

Initialize the EAV Transformer.

**Parameters:**
- `eav_manager` (Optional[EAVManager]): EAVManager instance. If None, creates a new one.

**Example:**
```python
from nexusml.core.eav_manager import EAVManager, EAVTransformer

# Create transformer with default EAV manager
transformer = EAVTransformer()

# Create transformer with custom EAV manager
eav_manager = EAVManager("path/to/custom_templates.json")
custom_transformer = EAVTransformer(eav_manager)
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**
- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**
- self: Returns the transformer instance

**Example:**
```python
from nexusml.core.eav_manager import EAVTransformer
import pandas as pd

transformer = EAVTransformer()

# Fit the transformer (no-op)
transformer.fit(pd.DataFrame())
```

##### `transform(X)`

Transform the input DataFrame by adding EAV attributes.

**Parameters:**
- `X`: Input DataFrame with at least 'Equipment_Category' column

**Returns:**
- Transformed DataFrame with EAV attributes

**Example:**
```python
from nexusml.core.eav_manager import EAVTransformer
import pandas as pd

# Create sample data
data = pd.DataFrame({
    "Equipment_Category": ["Chiller", "Air Handler", "Pump"],
    "Description": [
        "500 ton centrifugal chiller",
        "10,000 CFM air handler",
        "100 GPM circulation pump"
    ]
})

# Create transformer
transformer = EAVTransformer()

# Transform data
transformed_data = transformer.transform(data)
print(transformed_data.head())
```

**Notes:**
- The method adds the following columns to the input DataFrame:
  - `omniclass_id`: OmniClass classification ID
  - `masterformat_id`: MasterFormat classification ID
  - `uniformat_id`: Uniformat classification ID
  - `default_service_life`: Default service life from performance fields
  - `maintenance_interval`: Default maintenance interval from performance fields
  - `required_attribute_count`: Number of required attributes for the equipment type
- If the 'Equipment_Category' column is not found, adds empty columns with default values

## Functions

### `get_eav_manager() -> EAVManager`

Get an instance of the EAVManager.

**Returns:**
- EAVManager: Instance of EAVManager

**Example:**
```python
from nexusml.core.eav_manager import get_eav_manager

# Get an EAV manager instance
eav_manager = get_eav_manager()

# Use the EAV manager
template = eav_manager.get_equipment_template("Chiller")
```

## Usage Examples

### Basic Usage

```python
from nexusml.core.eav_manager import EAVManager

# Create EAV manager
eav_manager = EAVManager()

# Get template for a specific equipment type
chiller_template = eav_manager.get_equipment_template("Chiller")

# Get required and optional attributes
required_attrs = eav_manager.get_required_attributes("Chiller")
optional_attrs = eav_manager.get_optional_attributes("Chiller")

# Get classification IDs
classification_ids = eav_manager.get_classification_ids("Chiller")

# Generate attribute template
template = eav_manager.generate_attribute_template("Chiller")

# Print results
print(f"Required attributes: {required_attrs}")
print(f"Optional attributes: {optional_attrs}")
print(f"Classification IDs: {classification_ids}")
print(f"Generated template: {template}")
```

### Validating Equipment Attributes

```python
from nexusml.core.eav_manager import EAVManager

# Create EAV manager
eav_manager = EAVManager()

# Define attributes for different equipment types
chiller_attrs = {
    "Capacity": 500,
    "EER": 10.5,
    "Refrigerant": "R-134a",
    "Custom_Field": "Some value"  # Not in template
}

ahu_attrs = {
    "Airflow": 10000,
    "Static_Pressure": 2.5
    # Missing required attributes
}

# Validate attributes
chiller_validation = eav_manager.validate_attributes("Chiller", chiller_attrs)
ahu_validation = eav_manager.validate_attributes("Air Handler", ahu_attrs)

# Print validation results
print("Chiller validation:")
print(f"  Missing required: {chiller_validation['missing_required']}")
print(f"  Unknown attributes: {chiller_validation['unknown']}")

print("\nAHU validation:")
print(f"  Missing required: {ahu_validation['missing_required']}")
print(f"  Unknown attributes: {ahu_validation['unknown']}")
```

### Using EAVTransformer in a Pipeline

```python
from nexusml.core.eav_manager import EAVTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Create sample data
data = pd.DataFrame({
    "Equipment_Category": ["Chiller", "Air Handler", "Pump", "Chiller", "Air Handler"],
    "Description": [
        "500 ton centrifugal chiller",
        "10,000 CFM air handler",
        "100 GPM circulation pump",
        "300 ton screw chiller",
        "5,000 CFM air handler"
    ],
    "Service_Life": [20, 15, 10, 25, 18]
})

# Define target variable
y = pd.Series(["HVAC", "HVAC", "Plumbing", "HVAC", "HVAC"])

# Create pipeline with EAV transformer
pipeline = Pipeline([
    ('eav', EAVTransformer()),
    ('classifier', RandomForestClassifier())
])

# Fit the pipeline
pipeline.fit(data, y)

# Make predictions
new_data = pd.DataFrame({
    "Equipment_Category": ["Pump", "Chiller"],
    "Description": ["200 GPM circulation pump", "400 ton centrifugal chiller"],
    "Service_Life": [12, 22]
})

predictions = pipeline.predict(new_data)
print(f"Predictions: {predictions}")
```

### Complete Example with Attribute Filling

```python
from nexusml.core.eav_manager import EAVManager
import pandas as pd

# Create EAV manager
eav_manager = EAVManager()

# Define equipment data
equipment_data = [
    {
        "type": "Chiller",
        "description": "500 ton centrifugal chiller with R-134a refrigerant",
        "attributes": {
            "Capacity": 500,
            "EER": None,  # Missing value
            "Refrigerant": "R-134a"
        }
    },
    {
        "type": "Air Handler",
        "description": "10,000 CFM air handler with MERV 13 filters",
        "attributes": {
            "Airflow": 10000,
            "Static_Pressure": None,  # Missing value
            "Filter_Type": "MERV 13"
        }
    }
]

# Process each equipment
for equipment in equipment_data:
    # Get template
    template = eav_manager.get_equipment_template(equipment["type"])
    
    # Validate attributes
    validation = eav_manager.validate_attributes(
        equipment["type"], equipment["attributes"]
    )
    
    # Fill missing attributes
    filled_attrs = eav_manager.fill_missing_attributes(
        equipment["type"], 
        equipment["attributes"],
        equipment["description"]
    )
    
    # Print results
    print(f"\nEquipment Type: {equipment['type']}")
    print(f"Description: {equipment['description']}")
    print(f"Original Attributes: {equipment['attributes']}")
    print(f"Validation Results: {validation}")
    print(f"Filled Attributes: {filled_attrs}")
    
    # Get classification IDs
    classification = eav_manager.get_classification_ids(equipment["type"])
    print(f"Classification IDs: {classification}")
```

## Dependencies

- **json**: Used for loading JSON templates
- **os**: Used for file operations
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **numpy**: Used for numerical operations
- **pandas**: Used for DataFrame operations
- **sklearn.base**: Used for BaseEstimator and TransformerMixin classes
- **nexusml.config**: Used for getting project root

## Notes and Warnings

- The module relies on template files in JSON format. Make sure these files are properly set up before using the EAV manager.
- The default templates path is relative to the project root. If you're using the module from a different location, you may need to provide an absolute path.
- The EAVTransformer requires an 'Equipment_Category' column in the input DataFrame. If this column is not present, it will add empty columns with default values.
- The `fill_missing_attributes` method can use an ML model to predict missing attributes, but the model must have a `predict_attributes` method.
- Template matching is flexible, allowing for partial matches. This can be useful when dealing with variations in equipment type names, but it may also lead to unexpected matches if not used carefully.