# Command-Line Tool: classify_equipment.py

## Overview

The `classify_equipment.py` script is a command-line tool for classifying equipment data with any column structure. It provides a flexible way to process equipment data from various sources, map it to the expected model format, classify it, and output the results in a format ready for database import.

Key features include:

1. **Flexible Input Handling**: Accepts CSV or Excel files with any column structure
2. **Dynamic Field Mapping**: Maps input fields to the expected model format using a configurable mapping
3. **Automatic Classification**: Uses the NexusML model to classify equipment based on descriptions
4. **EAV Integration**: Includes Entity-Attribute-Value templates for each equipment type
5. **Database-Ready Output**: Formats results for easy integration with databases
6. **Multiple Output Formats**: Supports JSON and CSV output formats

## Usage

```bash
python classify_equipment.py input_file [--output OUTPUT] [--config CONFIG]
```

### Arguments

- `input_file`: Path to input file (CSV or Excel)
- `--output`, `-o`: Path to output file (JSON or CSV). If not provided, defaults to input_file_classified.json
- `--config`, `-c`: Path to classification configuration file. If not provided, uses the default configuration.

### Examples

#### Basic Usage

```bash
python classify_equipment.py equipment_data.csv
```

This will process `equipment_data.csv` and save the results to `equipment_data_classified.json`.

#### Specifying Output File

```bash
python classify_equipment.py equipment_data.xlsx --output classified_results.csv
```

This will process `equipment_data.xlsx` and save the results to `classified_results.csv` in CSV format.

#### Using Custom Configuration

```bash
python classify_equipment.py equipment_data.csv --config custom_mapping.json --output results.json
```

This will process `equipment_data.csv` using the mapping configuration in `custom_mapping.json` and save the results to `results.json`.

## Functions

### `process_any_input_file(input_file, output_file=None, config_file=None)`

Process equipment data with any column structure.

**Parameters:**

- `input_file` (str): Path to input file (CSV, Excel)
- `output_file` (str, optional): Path to output CSV or JSON file. If None, defaults to input_file_classified.json.
- `config_file` (str, optional): Path to classification config file. If None, uses the default configuration.

**Returns:**

- list: List of classification results for each equipment item

**Example:**
```python
from nexusml.classify_equipment import process_any_input_file

# Process a CSV file
results = process_any_input_file("equipment_data.csv")

# Process an Excel file with custom output and configuration
results = process_any_input_file(
    "equipment_data.xlsx",
    output_file="classified_results.json",
    config_file="custom_mapping.json"
)
```

## Input Format

The script accepts CSV or Excel files with any column structure. The DynamicFieldMapper will attempt to map the input columns to the expected model format based on the configuration.

Example input CSV:
```csv
Asset ID,Equipment Name,System,Description,Expected Life
A001,Centrifugal Chiller,HVAC,500 ton water-cooled centrifugal chiller,20
A002,Air Handler,HVAC,10000 CFM air handler with MERV 13 filters,15
A003,Circulation Pump,Plumbing,100 GPM circulation pump,10
```

## Output Format

### JSON Output

The JSON output format includes detailed information for each equipment item:

```json
[
  {
    "original_data": {
      "Asset ID": "A001",
      "Equipment Name": "Centrifugal Chiller",
      "System": "HVAC",
      "Description": "500 ton water-cooled centrifugal chiller",
      "Expected Life": 20
    },
    "classification": {
      "category_name": "Chiller",
      "uniformat_code": "D3010",
      "mcaa_system_category": "HVAC",
      "Equipment_Type": "Chiller-Centrifugal",
      "System_Subtype": "Cooling",
      "Asset Tag": "A001",
      "MasterFormat_Class": "23 64 16",
      "Equipment_Category": "Chiller",
      "required_attributes": [
        "cooling_capacity_tons",
        "chiller_type",
        "refrigerant"
      ],
      "master_db_mapping": {
        "asset_category": "Chiller",
        "asset_type": "Centrifugal",
        "system_type": "HVAC"
      }
    },
    "db_fields": {
      "category_name": {
        "value": "Chiller",
        "table": "assets",
        "field": "asset_category",
        "id_field": "asset_id"
      },
      "Equipment_Type": {
        "value": "Chiller-Centrifugal",
        "table": "assets",
        "field": "asset_type",
        "id_field": "asset_id"
      }
    },
    "eav_template": {
      "equipment_type": "Chiller",
      "required_attributes": [
        "cooling_capacity_tons",
        "chiller_type",
        "refrigerant"
      ],
      "classification_ids": {
        "omniclass_id": "23-13 11 11",
        "masterformat_id": "23 64 16",
        "uniformat_id": "D3010"
      }
    }
  }
]
```

### CSV Output

The CSV output format flattens the results into a tabular structure:

```csv
original_Asset ID,original_Equipment Name,original_System,original_Description,original_Expected Life,class_category_name,class_uniformat_code,class_mcaa_system_category,class_Equipment_Type,class_System_Subtype,class_Asset Tag,class_MasterFormat_Class,class_Equipment_Category,db_category_name_value,db_category_name_table,db_Equipment_Type_value,db_Equipment_Type_table
A001,Centrifugal Chiller,HVAC,500 ton water-cooled centrifugal chiller,20,Chiller,D3010,HVAC,Chiller-Centrifugal,Cooling,A001,23 64 16,Chiller,Chiller,assets,Chiller-Centrifugal,assets
A002,Air Handler,HVAC,10000 CFM air handler with MERV 13 filters,15,Air Handler,D3040,HVAC,Air Handler-VAV,Air Distribution,A002,23 74 13,Air Handler,Air Handler,assets,Air Handler-VAV,assets
A003,Circulation Pump,Plumbing,100 GPM circulation pump,10,Pump,D2010,Plumbing,Pump-Circulation,Water Distribution,A003,22 11 23,Pump,Pump,assets,Pump-Circulation,assets
```

## Configuration

The script can use a custom configuration file to map input fields to the expected model format. The configuration file should be in JSON format and specify:

1. Field mappings from input columns to model fields
2. Classification targets
3. Database field mappings

Example configuration:
```json
{
  "field_mappings": [
    {
      "input_fields": ["Asset ID", "Equipment ID", "Tag Number"],
      "model_field": "Asset Tag",
      "required": false
    },
    {
      "input_fields": ["Equipment Name", "Asset Name", "Description"],
      "model_field": "Asset Category",
      "required": true
    },
    {
      "input_fields": ["System", "System Type"],
      "model_field": "System_Type",
      "required": false
    },
    {
      "input_fields": ["Expected Life", "Service Life", "Useful Life"],
      "model_field": "Service Life",
      "required": false
    }
  ],
  "classification_targets": [
    "Equipment_Category",
    "Uniformat_Class",
    "System_Type",
    "Equipment_Type",
    "System_Subtype"
  ],
  "db_field_mapping": {
    "Equipment_Category": {
      "table": "assets",
      "field": "asset_category",
      "id_field": "asset_id"
    },
    "Equipment_Type": {
      "table": "assets",
      "field": "asset_type",
      "id_field": "asset_id"
    },
    "System_Type": {
      "table": "systems",
      "field": "system_type",
      "id_field": "system_id"
    }
  }
}
```

## Process Flow

1. **Load Input Data**: The script loads the input file (CSV or Excel) into a pandas DataFrame.
2. **Map Fields**: Using the DynamicFieldMapper, it maps the input columns to the expected model format.
3. **Train Model**: It trains the classification model using the train_enhanced_model function.
4. **Process Each Row**: For each row in the mapped DataFrame:
   - It creates a description by combining relevant text fields.
   - It extracts the service life value if available.
   - It uses the model to predict the equipment classification.
   - It gets the EAV template for the predicted equipment type.
   - It maps the prediction to database fields based on the configuration.
5. **Save Results**: It saves the results to the specified output file in JSON or CSV format.

## Dependencies

- **argparse**: Used for command-line argument parsing
- **json**: Used for JSON serialization
- **os**: Used for file path operations
- **sys**: Used for modifying the Python path
- **pathlib**: Used for path manipulation
- **pandas**: Used for data manipulation
- **nexusml.core.dynamic_mapper**: Used for mapping input fields to model format
- **nexusml.core.eav_manager**: Used for EAV template management
- **nexusml.core.model**: Used for model training and prediction

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The input file must be in CSV or Excel format.
- The output file can be in JSON or CSV format, determined by the file extension.
- If no output file is specified, the script will create a JSON file with the same name as the input file but with "_classified.json" appended.
- The script will train a new model each time it is run, which can be time-consuming. For production use, consider modifying the script to load a pre-trained model.
- The DynamicFieldMapper attempts to match input columns to expected model fields based on the configuration, but it may not always find perfect matches. Review the results to ensure accurate mapping.
- The script processes each row independently, which may not be optimal for very large datasets. For large datasets, consider batching the processing.
