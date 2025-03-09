# Module: feature_engineering

## Overview

The `feature_engineering` module handles feature transformations for the equipment classification model in the NexusML system. It follows the Single Responsibility Principle by focusing solely on feature engineering tasks. This module provides functionality to:

1. Combine multiple text columns into a single column
2. Clean and transform numeric columns
3. Create hierarchical category structures
4. Map columns from source to target
5. Map equipment descriptions to classification system IDs
6. Apply multiple transformations based on a configuration file
7. Enhance features with hierarchical structure and more granular categories

The module is designed to be flexible and configurable, with many components supporting dependency injection for better testability.

## Classes

### Class: TextCombiner

Combines multiple text columns into one column.

#### Attributes

- `columns` (List[str]): List of column names to combine
- `separator` (str): String used to join the column values
- `new_column` (str): Name of the new combined column

#### Methods

##### `__init__(columns: List[str], separator: str = " ", new_column: str = "combined_text")`

Initialize the transformer.

**Parameters:**

- `columns` (List[str]): List of column names to combine
- `separator` (str, optional): String used to join the column values. Default is a space.
- `new_column` (str, optional): Name of the new combined column. Default is "combined_text".

**Example:**
```python
from nexusml.core.feature_engineering import TextCombiner
import pandas as pd

# Create a DataFrame with multiple text columns
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"],
    "Description": ["500 ton", "10,000 CFM", "100 GPM"]
})

# Create a TextCombiner to combine "Asset Category" and "Equip Name ID"
combiner = TextCombiner(
    columns=["Asset Category", "Equip Name ID"],
    separator=" ",
    new_column="combined_text"
)

# Apply the transformation
transformed_df = combiner.transform(df)
print(transformed_df["combined_text"])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by combining specified text columns.

**Parameters:**

- `X`: Input DataFrame with columns to combine

**Returns:**

- Transformed DataFrame with the new combined column

**Notes:**

- If some of the specified columns are missing, the transformer will use only the available columns
- If none of the specified columns are available, it creates an empty column with the specified name
- All values are converted to strings before combining
- NaN values are filled with empty strings

### Class: NumericCleaner

Cleans and transforms numeric columns.

#### Attributes

- `column` (str): Name of the column to clean
- `new_name` (str): Name of the new cleaned column
- `fill_value` (Union[int, float]): Value to use for filling NaN values
- `dtype` (str): Data type to convert the column to ("float" or "int")

#### Methods

##### `__init__(column: str, new_name: Optional[str] = None, fill_value: Union[int, float] = 0, dtype: str = "float")`

Initialize the transformer.

**Parameters:**

- `column` (str): Name of the column to clean
- `new_name` (Optional[str], optional): Name of the new cleaned column. If None, uses the original column name.
- `fill_value` (Union[int, float], optional): Value to use for filling NaN values. Default is 0.
- `dtype` (str, optional): Data type to convert the column to. Default is "float".

**Example:**
```python
from nexusml.core.feature_engineering import NumericCleaner
import pandas as pd

# Create a DataFrame with a numeric column containing NaN values
df = pd.DataFrame({
    "Service Life": [20, None, 15, None, 10],
    "Capacity": ["500", "300", None, "200", "100"]
})

# Create a NumericCleaner for the "Service Life" column
cleaner = NumericCleaner(
    column="Service Life",
    new_name="service_life",
    fill_value=0,
    dtype="int"
)

# Apply the transformation
transformed_df = cleaner.transform(df)
print(transformed_df["service_life"])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by cleaning the specified numeric column.

**Parameters:**

- `X`: Input DataFrame with the column to clean

**Returns:**

- Transformed DataFrame with the cleaned numeric column

**Notes:**

- If the specified column doesn't exist, it creates a new column with the default fill value
- NaN values are filled with the specified fill value
- The column is converted to the specified data type (float or int)

### Class: HierarchyBuilder

Creates hierarchical category columns by combining parent columns.

#### Attributes

- `new_column` (str): Name of the new hierarchical column
- `parent_columns` (List[str]): List of parent column names to combine
- `separator` (str): String used to join the parent column values

#### Methods

##### `__init__(new_column: str, parent_columns: List[str], separator: str = "-")`

Initialize the transformer.

**Parameters:**

- `new_column` (str): Name of the new hierarchical column
- `parent_columns` (List[str]): List of parent column names to combine
- `separator` (str, optional): String used to join the parent column values. Default is "-".

**Example:**
```python
from nexusml.core.feature_engineering import HierarchyBuilder
import pandas as pd

# Create a DataFrame with parent columns
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"]
})

# Create a HierarchyBuilder to create a hierarchical "Equipment_Type" column
builder = HierarchyBuilder(
    new_column="Equipment_Type",
    parent_columns=["Asset Category", "Equip Name ID"],
    separator="-"
)

# Apply the transformation
transformed_df = builder.transform(df)
print(transformed_df["Equipment_Type"])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by creating a hierarchical column from parent columns.

**Parameters:**

- `X`: Input DataFrame with parent columns

**Returns:**

- Transformed DataFrame with the new hierarchical column

**Notes:**

- If some of the specified parent columns are missing, the transformer will use only the available columns
- If none of the specified parent columns are available, it creates an empty column with the specified name
- All values are converted to strings before combining

### Class: ColumnMapper

Maps source columns to target columns.

#### Attributes

- `mappings` (List[Dict[str, str]]): List of mappings from source to target columns

#### Methods

##### `__init__(mappings: List[Dict[str, str]])`

Initialize the transformer.

**Parameters:**

- `mappings` (List[Dict[str, str]]): List of mappings from source to target columns. Each mapping is a dictionary with "source" and "target" keys.

**Example:**
```python
from nexusml.core.feature_engineering import ColumnMapper
import pandas as pd

# Create a DataFrame with source columns
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"]
})

# Create a ColumnMapper to map source columns to target columns
mapper = ColumnMapper([
    {"source": "Asset Category", "target": "Equipment_Category"},
    {"source": "Equip Name ID", "target": "Equipment_Type"}
])

# Apply the transformation
transformed_df = mapper.transform(df)
print(transformed_df[["Equipment_Category", "Equipment_Type"]])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by mapping source columns to target columns.

**Parameters:**

- `X`: Input DataFrame with source columns

**Returns:**

- Transformed DataFrame with target columns

**Notes:**

- If a source column doesn't exist, a warning is printed and the mapping is skipped
- The target column will have the same values as the source column

### Class: KeywordClassificationMapper

Maps equipment descriptions to classification system IDs using keyword matching.

#### Attributes

- `name` (str): Name of the classification system
- `source_column` (str): Column containing text to search for keywords
- `target_column` (str): Column to store the matched classification code
- `reference_manager` (str): Reference manager to use for keyword matching
- `max_results` (int): Maximum number of results to consider
- `confidence_threshold` (float): Minimum confidence score to accept a match
- `ref_manager` (ReferenceManager): Instance of ReferenceManager

#### Methods

##### `__init__(name: str, source_column: str, target_column: str, reference_manager: str = "uniformat_keywords", max_results: int = 1, confidence_threshold: float = 0.0)`

Initialize the transformer.

**Parameters:**

- `name` (str): Name of the classification system
- `source_column` (str): Column containing text to search for keywords
- `target_column` (str): Column to store the matched classification code
- `reference_manager` (str, optional): Reference manager to use for keyword matching. Default is "uniformat_keywords".
- `max_results` (int, optional): Maximum number of results to consider. Default is 1.
- `confidence_threshold` (float, optional): Minimum confidence score to accept a match. Default is 0.0.

**Example:**
```python
from nexusml.core.feature_engineering import KeywordClassificationMapper
import pandas as pd

# Create a DataFrame with equipment descriptions
df = pd.DataFrame({
    "combined_text": [
        "500 ton centrifugal chiller",
        "10,000 CFM air handler with MERV 13 filters",
        "100 GPM circulation pump"
    ],
    "Uniformat_Class": ["", "", ""]  # Empty target column
})

# Create a KeywordClassificationMapper for Uniformat classification
mapper = KeywordClassificationMapper(
    name="Uniformat",
    source_column="combined_text",
    target_column="Uniformat_Class",
    reference_manager="uniformat_keywords",
    max_results=1
)

# Apply the transformation
transformed_df = mapper.transform(df)
print(transformed_df["Uniformat_Class"])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by adding classification codes based on keyword matching.

**Parameters:**

- `X`: Input DataFrame with source column

**Returns:**

- Transformed DataFrame with target column containing classification codes

**Notes:**

- Currently only supports Uniformat classification with the "uniformat_keywords" reference manager
- Only processes rows where the target column is empty or NaN
- If the source column doesn't exist, a warning is printed and the target column is set to empty strings

### Class: ClassificationSystemMapper

Maps equipment categories to classification system IDs (OmniClass, MasterFormat, Uniformat).

#### Attributes

- `name` (str): Name of the classification system
- `source_column` (Union[str, List[str]]): Column(s) containing equipment categories
- `target_column` (str): Column to store the classification ID
- `mapping_type` (str): Type of mapping to use ("eav" or custom)
- `mapping_function` (Optional[str]): Name of the mapping function to use
- `eav_manager` (EAVManager): Instance of EAVManager

#### Methods

##### `__init__(name: str, source_column: Union[str, List[str]], target_column: str, mapping_type: str = "eav", mapping_function: Optional[str] = None, eav_manager: Optional[EAVManager] = None)`

Initialize the transformer.

**Parameters:**

- `name` (str): Name of the classification system
- `source_column` (Union[str, List[str]]): Column(s) containing equipment categories
- `target_column` (str): Column to store the classification ID
- `mapping_type` (str, optional): Type of mapping to use. Default is "eav".
- `mapping_function` (Optional[str], optional): Name of the mapping function to use. Default is None.
- `eav_manager` (Optional[EAVManager], optional): Instance of EAVManager. If None, creates a new one.

**Example:**
```python
from nexusml.core.feature_engineering import ClassificationSystemMapper
from nexusml.core.eav_manager import EAVManager
import pandas as pd

# Create a DataFrame with equipment categories
df = pd.DataFrame({
    "Equipment_Category": ["Chiller", "Air Handler", "Pump"]
})

# Create a ClassificationSystemMapper for OmniClass classification
mapper = ClassificationSystemMapper(
    name="OmniClass",
    source_column="Equipment_Category",
    target_column="OmniClass_ID",
    mapping_type="eav",
    eav_manager=EAVManager()
)

# Apply the transformation
transformed_df = mapper.transform(df)
print(transformed_df["OmniClass_ID"])
```

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame by adding classification IDs.

**Parameters:**

- `X`: Input DataFrame with source column(s)

**Returns:**

- Transformed DataFrame with target column containing classification IDs

**Notes:**

- Supports two mapping types:
  - "eav": Uses the EAVManager to get classification IDs
  - Custom: Uses a specified mapping function (e.g., "enhanced_masterformat_mapping")
- For "eav" mapping, supports OmniClass, MasterFormat, and Uniformat classification systems
- For custom mapping, supports the "enhanced_masterformat_mapping" function
- If the source column doesn't exist, a warning is printed and the target column is set to empty strings

### Class: GenericFeatureEngineer

A generic feature engineering transformer that applies multiple transformations based on a configuration file.

#### Attributes

- `config_path` (Optional[str]): Path to the YAML configuration file
- `transformers` (List): List of transformer instances
- `config` (Dict): Configuration dictionary
- `eav_manager` (EAVManager): Instance of EAVManager

#### Methods

##### `__init__(config_path: Optional[str] = None, eav_manager: Optional[EAVManager] = None)`

Initialize the transformer with a configuration file path.

**Parameters:**

- `config_path` (Optional[str], optional): Path to the YAML configuration file. If None, uses the default path.
- `eav_manager` (Optional[EAVManager], optional): EAVManager instance. If None, uses the one from the DI container.

**Example:**
```python
from nexusml.core.feature_engineering import GenericFeatureEngineer
import pandas as pd

# Create a DataFrame with raw features
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"],
    "Service Life": [20, 15, 10]
})

# Create a GenericFeatureEngineer with default configuration
engineer = GenericFeatureEngineer()

# Apply all transformations
transformed_df = engineer.transform(df)
print(transformed_df.columns)
```

##### `_load_config()`

Load the configuration from the YAML file.

**Notes:**

- If no config_path is provided, uses the default path: "config/feature_config.yml" relative to the project root
- The configuration file should be in YAML format

##### `fit(X, y=None)`

Fit method (does nothing but is required for the transformer interface).

**Parameters:**

- `X`: Input data (not used)
- `y`: Target data (not used)

**Returns:**

- self: Returns the transformer instance

##### `transform(X)`

Transform the input DataFrame based on the configuration.

**Parameters:**

- `X`: Input DataFrame with raw features

**Returns:**

- Transformed DataFrame with enhanced features

**Notes:**

- Applies the following transformations in order:
  1. Column mappings
  2. Text combinations
  3. Numeric column cleaning
  4. Hierarchical categories
  5. Keyword classification mappings
  6. Classification system mappings
  7. EAV integration (if enabled)
- Each transformation is applied only if the corresponding section exists in the configuration

## Functions

### `enhance_features(df: pd.DataFrame, feature_engineer: Optional[GenericFeatureEngineer] = None) -> pd.DataFrame`

Enhanced feature engineering with hierarchical structure and more granular categories.

**Parameters:**

- `df` (pd.DataFrame): Input dataframe with raw features
- `feature_engineer` (Optional[GenericFeatureEngineer], optional): Feature engineer instance. If None, uses the one from the DI container.

**Returns:**

- pd.DataFrame: DataFrame with enhanced features

**Example:**
```python
from nexusml.core.feature_engineering import enhance_features
import pandas as pd

# Create a DataFrame with raw features
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"],
    "Service Life": [20, 15, 10]
})

# Apply feature engineering
enhanced_df = enhance_features(df)
print(enhanced_df.columns)
```

**Notes:**

- This function uses the GenericFeatureEngineer transformer to apply transformations based on the configuration file
- If no feature_engineer is provided, it gets one from the DI container

### `create_hierarchical_categories(df: pd.DataFrame) -> pd.DataFrame`

Create hierarchical category structure to better handle "Other" categories.

**Parameters:**

- `df` (pd.DataFrame): Input dataframe with basic features

**Returns:**

- pd.DataFrame: DataFrame with hierarchical category features

**Example:**
```python
from nexusml.core.feature_engineering import create_hierarchical_categories
import pandas as pd

# Create a DataFrame with basic features
df = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"],
    "Precon System": ["HVAC", "HVAC", "Plumbing"],
    "Operations System": ["Cooling", "Air Distribution", "Water Distribution"]
})

# Create hierarchical categories
hierarchical_df = create_hierarchical_categories(df)
print(hierarchical_df[["Equipment_Type", "System_Subtype"]])
```

**Notes:**

- This function is kept for backward compatibility
- It adds two hierarchical columns:
  - "Equipment_Type": Combines "Asset Category" and "Equip Name ID" with a hyphen
  - "System_Subtype": Combines "Precon System" and "Operations System" with a hyphen
- If the required columns don't exist, it adds default "Unknown" values

### `load_masterformat_mappings() -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]`

Load MasterFormat mappings from JSON files.

**Returns:**

- Tuple[Dict[str, Dict[str, str]], Dict[str, str]]: Primary and equipment-specific mappings

**Example:**
```python
from nexusml.core.feature_engineering import load_masterformat_mappings

# Load MasterFormat mappings
primary_mapping, equipment_specific_mapping = load_masterformat_mappings()

# Print the mappings
print("Primary mapping keys:", list(primary_mapping.keys()))
print("Equipment-specific mapping keys:", list(equipment_specific_mapping.keys()))
```

**Notes:**

- Loads mappings from two JSON files:
  - "config/mappings/masterformat_primary.json": Primary mappings by Uniformat class and system type
  - "config/mappings/masterformat_equipment.json": Equipment-specific mappings
- If the files cannot be loaded, returns empty dictionaries

### `enhanced_masterformat_mapping(uniformat_class: str, system_type: str, equipment_category: str, equipment_subcategory: Optional[str] = None, eav_manager: Optional[EAVManager] = None) -> str`

Enhanced mapping with better handling of specialty equipment types.

**Parameters:**

- `uniformat_class` (str): Uniformat classification
- `system_type` (str): System type
- `equipment_category` (str): Equipment category
- `equipment_subcategory` (Optional[str], optional): Equipment subcategory. Default is None.
- `eav_manager` (Optional[EAVManager], optional): EAV manager instance. If None, uses the one from the DI container.

**Returns:**

- str: MasterFormat classification code

**Example:**
```python
from nexusml.core.feature_engineering import enhanced_masterformat_mapping

# Get MasterFormat code for a chiller
masterformat_code = enhanced_masterformat_mapping(
    uniformat_class="H",
    system_type="HVAC",
    equipment_category="Chiller",
    equipment_subcategory="Centrifugal Chiller"
)
print(f"MasterFormat code: {masterformat_code}")
```

**Notes:**

- Tries to find a MasterFormat code in the following order:
  1. Equipment-specific mapping based on equipment_subcategory
  2. Primary mapping based on uniformat_class and system_type
  3. EAV-based mapping based on equipment_category
  4. Fallback mapping based on uniformat_class
- Fallback mappings:
  - "H": "23 00 00" (HVAC)
  - "P": "22 00 00" (Plumbing)
  - "SM": "23 00 00" (HVAC)
  - "R": "11 40 00" (Foodservice Equipment)
- Returns "00 00 00" if no match is found

## Configuration File Format

The GenericFeatureEngineer uses a YAML configuration file to define the transformations to apply. The default path is "config/feature_config.yml" relative to the project root.

### Example Configuration

```yaml
# Column mappings from source to target
column_mappings:
  - source: "Asset Category"
    target: "Equipment_Category"
  - source: "Equip Name ID"
    target: "Equipment_Type"

# Text combinations
text_combinations:
  - columns: ["Asset Category", "Equip Name ID", "Description"]
    separator: " "
    name: "combined_text"

# Numeric column cleaning
numeric_columns:
  - name: "Service Life"
    new_name: "service_life"
    fill_value: 0
    dtype: "float"

# Hierarchical categories
hierarchies:
  - new_col: "Equipment_Type"
    parents: ["Asset Category", "Equip Name ID"]
    separator: "-"
  - new_col: "System_Subtype"
    parents: ["Precon System", "Operations System"]
    separator: "-"

# Keyword classification mappings
keyword_classifications:
  - name: "Uniformat"
    source_column: "combined_text"
    target_column: "Uniformat_Class"
    reference_manager: "uniformat_keywords"
    max_results: 1

# Classification system mappings
classification_systems:
  - name: "OmniClass"
    source_column: "Equipment_Category"
    target_column: "OmniClass_ID"
    mapping_type: "eav"
  - name: "MasterFormat"
    source_columns: ["Uniformat_Class", "System_Type", "Equipment_Category", "Equipment_Type"]
    target_column: "MasterFormat_ID"
    mapping_function: "enhanced_masterformat_mapping"
  - name: "Uniformat"
    source_column: "Equipment_Category"
    target_column: "Uniformat_ID"
    mapping_type: "eav"

# EAV integration
eav_integration:
  enabled: true
```

## Dependencies

- **json**: Used for loading JSON mappings
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **numpy**: Used for numerical operations
- **pandas**: Used for DataFrame operations
- **yaml**: Used for loading YAML configuration
- **sklearn.base**: Used for BaseEstimator and TransformerMixin classes
- **nexusml.config**: Used for getting project root
- **nexusml.core.eav_manager**: Used for EAVManager and EAVTransformer
- **nexusml.core.di.decorators**: Used for dependency injection
- **nexusml.core.di.provider**: Used for DI container
- **nexusml.core.reference.manager**: Used for ReferenceManager

## Notes and Warnings

- The module uses dependency injection for better testability and configurability
- The GenericFeatureEngineer class is the main entry point for feature engineering
- The configuration file is the central place to define transformations
- Most transformers handle missing columns gracefully by providing warnings and default values
- The KeywordClassificationMapper currently only supports Uniformat classification
- The ClassificationSystemMapper supports both EAV-based and custom mapping functions
- The enhanced_masterformat_mapping function provides a sophisticated mapping strategy with multiple fallbacks
