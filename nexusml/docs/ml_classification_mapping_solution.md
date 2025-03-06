# ML Classification Mapping Solution

## Problem Overview

Your ML classification model faced two major issues:

1. **Column Name Mismatch**: Your staging data (Excel/CSV) has columns like
   "Asset Name", "System Category", etc., but your ML model expects fields like
   "Asset Category", "Equip Name ID", etc.

2. **Missing Master Database Fields**: Your master database requires specific
   fields (CategoryID, LocationID, EquipmentTag) that must be derived from
   classifications.

## Solution Architecture

We've implemented a bidirectional mapping solution that:

1. Maps staging data columns → ML model input format
2. Maps ML model predictions → master database fields

### How EAV is Involved

Your system uses an **Entity-Attribute-Value (EAV)** model for flexible
equipment attributes. The EAV integration is preserved and enhanced in our
solution:

- **EAV Manager**: The existing `EAVManager` class handles attribute templates,
  classification IDs, and performance fields
- **EAV Integration**: The `predict_with_enhanced_model` function still includes
  EAV information in predictions
- **Classification Systems**: OmniClass, MasterFormat, and Uniformat IDs are
  still mapped through the EAV system

Our solution adds a layer before and after the EAV processing to handle the
column mapping and database field mapping.

## Implementation Details

### 1. Data Mapper Module

We created `data_mapper.py` to handle:

```python
# Mapping staging data to ML model input
def map_staging_to_model_input(staging_df):
    # Maps columns like "Asset Name" → "Asset Name" (preserved for ML model)
    # Handles numeric fields properly (Service Life, Motor HP)
    # Fills required fields with defaults

# Mapping ML predictions to master database fields
def map_predictions_to_master_db(predictions):
    # Maps Equipment_Category → CategoryID
    # Includes EquipmentTag, LocationID
    # Preserves classification IDs from EAV
```

### 2. Updated Feature Configuration

Modified `feature_config.yml` to work with staging data column names:

```yaml
text_combinations:
  - name: 'combined_text'
    columns:
      ['Asset Name', 'Manufacturer', 'Model Number', 'System Category', ...]

column_mappings:
  - source: 'Asset Name'
    target: 'Equipment_Category'

  - source: 'Trade'
    target: 'Uniformat_Class'
```

### 3. Integration with Existing Pipeline

Updated `model.py` to:

- Apply data mapping before feature engineering
- Include master database field mapping in predictions
- Preserve all EAV functionality

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Staging Data│     │  Data Mapper│     │ ML Model    │     │ EAV Manager │
│ (CSV/Excel) │────▶│  (Column    │────▶│ (Feature    │────▶│ (Templates, │
│             │     │   Mapping)  │     │  Engineering)│     │  Class IDs) │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│ Master DB   │     │  Data Mapper│     │ Prediction  │           │
│ Fields      │◀────│  (DB Field  │◀────│ Results     │◀──────────┘
│             │     │   Mapping)  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
```

## Example Usage

```python
from nexusml.core.model import EquipmentClassifier
from nexusml.core.data_mapper import map_staging_to_model_input

# Load staging data
staging_df = pd.read_csv("staging_data.csv")

# Initialize and train classifier
classifier = EquipmentClassifier()
classifier.train()

# Process equipment
for _, row in staging_df.iterrows():
    # Create description from relevant fields
    description = f"{row['Asset Name']} {row['System Category']} {row['Manufacturer']}"
    service_life = float(row.get('Service Life', 20.0))
    asset_tag = row.get('Asset Tag', '')

    # Get prediction with master DB mapping
    prediction = classifier.predict(description, service_life, asset_tag)

    # Access master database fields
    db_fields = prediction['master_db_mapping']
    print(f"CategoryID: {db_fields['CategoryID']}")
    print(f"EquipmentTag: {db_fields['EquipmentTag']}")
```

## Summary

This solution preserves all the existing EAV functionality while adding:

1. **Input Mapping**: Staging data columns → ML model format
2. **Output Mapping**: ML predictions → Master database fields

The EAV system still handles equipment attributes, classification IDs, and
performance fields, but now it works seamlessly with your staging data and
master database requirements.
