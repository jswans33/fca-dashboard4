# Module: data_mapper

## Overview

The `data_mapper` module handles mapping between different data formats in the NexusML system. It primarily focuses on:

1. Converting staging data to the format expected by ML models
2. Transforming ML model predictions to the format required by the master database

This module is a critical component in the data pipeline, ensuring that data flows correctly between different parts of the system.

## Classes

### Class: DataMapper

A utility class that maps data between different formats, specifically from staging data to ML model input and from ML model output to master database fields.

#### Attributes

- `column_mapping` (Dict[str, str]): Dictionary mapping staging columns to model input columns
- `required_fields` (Dict[str, str]): Dictionary of required fields with default values
- `numeric_fields` (Dict[str, float]): Dictionary of numeric fields with default values

#### Methods

##### `__init__(column_mapping: Optional[Dict[str, str]] = None)`

Initialize the data mapper with an optional column mapping.

**Parameters:**
- `column_mapping` (Optional[Dict[str, str]]): Dictionary mapping staging columns to model input columns. If not provided, a default mapping is used.

**Example:**
```python
# Create a data mapper with default column mapping
mapper = DataMapper()

# Create a data mapper with custom column mapping
custom_mapping = {
    "asset_tag": "equipment_tag",
    "equipment_type": "category_name",
    "system_type": "mcaa_system_category"
}
custom_mapper = DataMapper(column_mapping=custom_mapping)
```

##### `map_staging_to_model_input(staging_df: pd.DataFrame) -> pd.DataFrame`

Maps staging data columns to the format expected by the ML model.

**Parameters:**
- `staging_df` (pd.DataFrame): DataFrame from staging table

**Returns:**
- pd.DataFrame: DataFrame with columns mapped to what the ML model expects

**Example:**
```python
import pandas as pd
from nexusml.core.data_mapper import DataMapper

# Create sample staging data
staging_data = pd.DataFrame({
    "asset_tag": ["AHU-01", "CHW-02"],
    "equipment_type": ["Air Handler", "Chiller"],
    "system_type": ["HVAC", "Cooling"],
    "condition_score": [4, 3]
})

# Create mapper with custom mapping
mapper = DataMapper({
    "equipment_tag": "asset_tag",
    "category_name": "equipment_type",
    "mcaa_system_category": "system_type"
})

# Map staging data to model input format
model_input = mapper.map_staging_to_model_input(staging_data)
print(model_input.head())
```

##### `map_predictions_to_master_db(predictions: Dict[str, Any]) -> Dict[str, Any]`

Maps model predictions to master database fields.

**Parameters:**
- `predictions` (Dict[str, Any]): Dictionary of predictions from the ML model

**Returns:**
- Dict[str, Any]: Dictionary with fields mapped to master DB structure

**Example:**
```python
from nexusml.core.data_mapper import DataMapper

# Sample model predictions
predictions = {
    "category_name": "Air Handler",
    "mcaa_system_category": "HVAC",
    "uniformat_code": "D3050",
    "masterformat_code": "23 74 13",
    "equipment_tag": "AHU-01"
}

# Map predictions to master database format
mapper = DataMapper()
db_fields = mapper.map_predictions_to_master_db(predictions)
print(db_fields)
```

##### `_map_to_category_id(equipment_category: str) -> int`

Maps an equipment category name to a CategoryID for the master database.

**Parameters:**
- `equipment_category` (str): The equipment category name

**Returns:**
- int: CategoryID as an integer

**Note:**
This is a private method used internally by the class. In a real implementation, this would query the Equipment_Categories table or use a mapping dictionary. Currently, it uses a simple hash function to generate a positive integer.

## Functions

### `map_staging_to_model_input(staging_df: pd.DataFrame) -> pd.DataFrame`

A convenience function that maps staging data columns to the format expected by the ML model.

**Parameters:**
- `staging_df` (pd.DataFrame): DataFrame from staging table

**Returns:**
- pd.DataFrame: DataFrame with columns mapped to what the ML model expects

**Example:**
```python
import pandas as pd
from nexusml.core.data_mapper import map_staging_to_model_input

# Create sample staging data
staging_data = pd.DataFrame({
    "equipment_tag": ["AHU-01", "CHW-02"],
    "category_name": ["Air Handler", "Chiller"],
    "mcaa_system_category": ["HVAC", "Cooling"],
    "condition_score": ["4", "3"]  # Note: strings that need conversion
})

# Map staging data to model input format
model_input = map_staging_to_model_input(staging_data)
print(model_input.head())
```

### `map_predictions_to_master_db(predictions: Dict[str, Any]) -> Dict[str, Any]`

A convenience function that maps model predictions to master database fields.

**Parameters:**
- `predictions` (Dict[str, Any]): Dictionary of predictions from the ML model

**Returns:**
- Dict[str, Any]: Dictionary with fields mapped to master DB structure

**Example:**
```python
from nexusml.core.data_mapper import map_predictions_to_master_db

# Sample model predictions
predictions = {
    "category_name": "Air Handler",
    "mcaa_system_category": "HVAC",
    "uniformat_code": "D3050",
    "masterformat_code": "23 74 13",
    "equipment_tag": "AHU-01"
}

# Map predictions to master database format
db_fields = map_predictions_to_master_db(predictions)
print(db_fields)
```

## Usage Examples

### Basic Usage with Default Mapping

```python
import pandas as pd
from nexusml.core.data_mapper import map_staging_to_model_input, map_predictions_to_master_db

# Load staging data
staging_data = pd.read_csv("staging_data.csv")

# Map to model input format
model_input = map_staging_to_model_input(staging_data)

# After model prediction, map results to database format
model_predictions = {
    "category_name": "Air Handler",
    "mcaa_system_category": "HVAC",
    "uniformat_code": "D3050",
    "equipment_tag": "AHU-01"
}

db_fields = map_predictions_to_master_db(model_predictions)
```

### Custom Mapping for Non-Standard Column Names

```python
import pandas as pd
from nexusml.core.data_mapper import DataMapper

# Define custom mapping for non-standard column names
custom_mapping = {
    "equipment_tag": "Asset_ID",
    "category_name": "Equipment_Type",
    "mcaa_system_category": "System_Category",
    "condition_score": "Condition_Rating"
}

# Create mapper with custom mapping
mapper = DataMapper(column_mapping=custom_mapping)

# Load staging data with non-standard column names
staging_data = pd.read_csv("custom_format_data.csv")

# Map to model input format
model_input = mapper.map_staging_to_model_input(staging_data)

# Continue with model training or prediction
```

### Complete Pipeline Example

```python
import pandas as pd
from nexusml.core.data_mapper import DataMapper
from nexusml.core.model import EquipmentClassifier

# 1. Load staging data
staging_data = pd.read_csv("new_equipment_data.csv")

# 2. Create mapper and map to model input format
mapper = DataMapper()
model_input = mapper.map_staging_to_model_input(staging_data)

# 3. Load trained model
classifier = EquipmentClassifier()
classifier.load_model("models/equipment_classifier.pkl")

# 4. Make predictions
predictions_list = []
for _, row in model_input.iterrows():
    # Get prediction for this row
    prediction = classifier.predict_from_row(row)
    
    # Map prediction to database format
    db_fields = mapper.map_predictions_to_master_db(prediction)
    predictions_list.append(db_fields)

# 5. Convert to DataFrame for bulk database operations
results_df = pd.DataFrame(predictions_list)

# 6. Save results
results_df.to_csv("mapped_predictions.csv", index=False)
```

## Dependencies

- **pandas**: Used for DataFrame operations and data manipulation
- **typing**: Used for type hints

## Notes and Warnings

- The `_map_to_category_id` method is a placeholder that uses a simple hash function. In a production environment, this should be replaced with a proper lookup to the Equipment_Categories table.
- The default column mapping assumes specific column names in the staging data. If your staging data uses different column names, you should provide a custom mapping.
- Numeric fields like `condition_score` and `initial_cost` are automatically converted to numeric values, with errors coerced to NaN and then filled with default values.
- Required fields are filled with default values if missing in the input data.