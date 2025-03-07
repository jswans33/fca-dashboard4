# SOP 008: Machine Learning Model Training Procedure

## 1. Purpose and Scope

This Standard Operating Procedure (SOP) outlines the process for training the
equipment classification machine learning model using the NexusML framework. It
covers data preparation, feature engineering, model training, evaluation, and
deployment.

## 2. Prerequisites

Before beginning the model training process, ensure the following prerequisites
are met:

- Python 3.8 or higher is installed
- NexusML package is installed and configured
- Access to reference data sources
- Sufficient training data available
- Required dependencies installed (see `requirements.txt`)

## 3. Environment Setup

### 3.1 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.2 Configure Reference Data

Ensure the reference data configuration file is properly set up at
`nexusml/config/reference_config.yml`. This file should specify paths to all
reference data sources:

```yaml
# Example reference_config.yml
paths:
  omniclass: 'files/omniclass_tables/omniclass_table23.csv'
  uniformat: 'files/uniformat/uniformat_classification.csv'
  masterformat: 'files/masterformat/omniclass_mf_div_22_23_tables.csv'
  mcaa_glossary: 'files/mcaa-glossary/Glossary.pdf.md'
  mcaa_abbreviations: 'files/mcaa-glossary/Abbreviations.pdf.md'
  smacna: 'files/smacna-manufacturers-list/smacna_manufacturers.csv'
  ashrae: 'files/service_life/ashrae_service_life.csv'
  energize_denver: 'files/energize-denver/ed-technical-guidance.csv'
  equipment_taxonomy: 'files/equipment_book/USEC_EquipmentBook_2024-07-15.csv'
```

### 3.3 Configure Feature Engineering

Review and update the feature configuration file at
`nexusml/config/feature_config.yml` to ensure it includes all necessary feature
transformations. The configuration differs based on the data format you're
using.

#### 3.3.1 Legacy Format Configuration

```yaml
# Legacy format feature_config.yml
text_combinations:
  - name: 'combined_text'
    columns:
      [
        'Asset Name',
        'Manufacturer',
        'Model Number',
        'System Category',
        'Sub System Type',
        'Sub System Classification',
      ]
    separator: ' '

numeric_columns:
  - name: 'Service Life'
    new_name: 'service_life'
    fill_value: 20
    dtype: 'float'

hierarchies:
  - new_col: 'Equipment_Type'
    parents: ['System Category', 'Asset Name']
    separator: '-'

keyword_classifications:
  - name: 'Uniformat'
    source_column: 'combined_text'
    target_column: 'Uniformat_Class'
    reference_manager: 'uniformat_keywords'
    max_results: 1

classification_systems:
  - name: 'OmniClass'
    source_column: 'Equipment_Category'
    target_column: 'OmniClass_ID'
    mapping_type: 'eav'

eav_integration:
  enabled: true
  equipment_type_column: 'Equipment_Category'
  description_column: 'combined_text'
  service_life_column: 'service_life'
  add_classification_ids: true
  add_performance_fields: true
```

#### 3.3.2 Production Format Configuration (Recommended)

```yaml
# Production format feature_config.yml
text_combinations:
  - name: 'combined_text'
    columns:
      [
        'equipment_tag',
        'manufacturer',
        'model',
        'category_name',
        'mcaa_system_category',
        'building_name',
      ]
    separator: ' '

numeric_columns:
  - name: 'initial_cost'
    new_name: 'initial_cost'
    fill_value: 0
    dtype: 'float'

  - name: 'condition_score'
    new_name: 'service_life' # Map condition_score to service_life
    fill_value: 3.0
    dtype: 'float'

hierarchies:
  - new_col: 'Equipment_Type'
    parents: ['mcaa_system_category', 'category_name']
    separator: '-'

  - new_col: 'System_Subtype'
    parents: ['mcaa_system_category', 'category_name']
    separator: '-'

column_mappings:
  - source: 'category_name'
    target: 'Equipment_Category'

  - source: 'uniformat_code'
    target: 'Uniformat_Class'

  - source: 'mcaa_system_category'
    target: 'System_Type'

classification_systems:
  - name: 'OmniClass'
    source_column: 'omniclass_code'
    target_column: 'OmniClass_ID'
    mapping_type: 'direct' # Use direct mapping instead of eav

  - name: 'MasterFormat'
    source_column: 'masterformat_code'
    target_column: 'MasterFormat_ID'
    mapping_type: 'direct' # Use direct mapping instead of function

  - name: 'Uniformat'
    source_column: 'uniformat_code'
    target_column: 'Uniformat_ID'
    mapping_type: 'direct' # Use direct mapping instead of eav

# Use the ID columns directly
direct_mappings:
  - source: 'CategoryID'
    target: 'Equipment_Subcategory'

  - source: 'OmniClassID'
    target: 'OmniClass_ID'

  - source: 'UniFormatID'
    target: 'Uniformat_ID'

  - source: 'MasterFormatID'
    target: 'MasterFormat_ID'

  - source: 'MCAAID'
    target: 'MCAA_ID'

  - source: 'LocationID'
    target: 'Location_ID'

eav_integration:
  enabled: false # Disable EAV integration since we're using direct mappings
```

## 4. Data Preparation

### 4.1 Prepare Training Data

The model supports two data formats: the legacy format and the new production
format.

#### 4.1.1 Legacy Format

The legacy format expects a CSV or Excel file with the following columns:

- Asset Name
- Asset Tag
- Trade
- System Category
- Sub System Type
- Sub System Classification
- Manufacturer
- Model Number
- Service Life

#### 4.1.2 Production Format (Recommended)

The production format uses the actual database column names and includes
classification IDs directly:

- equipment_tag
- manufacturer
- model
- category_name
- omniclass_code
- uniformat_code
- masterformat_code
- mcaa_system_category
- building_name
- initial_cost
- condition_score
- CategoryID
- OmniClassID
- UniFormatID
- MasterFormatID
- MCAAID
- LocationID

Place your training data in the appropriate location (default:
`files/training-data/` or `nexusml/data/training_data/`).

### 4.2 Data Validation

Validate your training data to ensure it meets quality standards. The validation
process checks for required columns and data quality issues.

#### 4.2.1 Using the Model Training Pipeline (Recommended)

The model training pipeline automatically validates the data before training:

```bash
# Validate and train
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv
```

#### 4.2.2 Using the Python API

```python
from nexusml.core.data_preprocessing import validate_training_data

# Validate training data
validation_results = validate_training_data("path/to/your/training_data.csv")
print(validation_results)
```

#### 4.2.3 Required Columns

The validation process checks for different required columns based on the data
format:

**Legacy Format:**

- Asset Name
- Asset Tag
- Trade
- System Category
- Sub System Type
- Sub System Classification
- Manufacturer
- Model Number
- Service Life

**Production Format:**

- equipment_tag
- manufacturer
- model
- category_name
- omniclass_code
- uniformat_code
- masterformat_code
- mcaa_system_category
- building_name
- initial_cost
- condition_score
- CategoryID
- OmniClassID
- UniFormatID
- MasterFormatID
- MCAAID
- LocationID

## 5. Model Training Process

### 5.1 Using the Model Training Pipeline (Recommended)

The recommended way to train the model is to use the production-ready model
training pipeline, which provides a comprehensive workflow with command-line
arguments, proper logging, evaluation, and model versioning.

#### 5.1.1 Using the Command-Line Interface

```bash
# Basic usage
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv

# With hyperparameter optimization and visualizations
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv --optimize --visualize

# With custom configuration files
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv \
  --feature-config path/to/custom/feature_config.yml \
  --reference-config path/to/custom/reference_config.yml
```

#### 5.1.2 Using the Shell Script

For convenience, a shell script is provided that wraps the pipeline with common
options:

```bash
# Make the script executable
chmod +x nexusml/scripts/train_model.sh

# Basic usage
./nexusml/scripts/train_model.sh -d files/training-data/your_data.csv

# With hyperparameter optimization and visualizations
./nexusml/scripts/train_model.sh -d files/training-data/your_data.csv -p -v

# Get help on all available options
./nexusml/scripts/train_model.sh -h
```

The pipeline performs the following steps:

1. Loads reference data
2. Validates training data
3. Trains the model (with optional hyperparameter optimization)
4. Evaluates the model
5. Saves the model with versioning
6. Generates visualizations (if requested)
7. Makes a sample prediction

The trained model and metadata are saved to the specified output directory
(default: `outputs/models/`).

### 5.2 Basic Training (Python API)

For programmatic use, you can train the model with default settings:

```python
from nexusml.core.model import EquipmentClassifier

# Create classifier instance
classifier = EquipmentClassifier()

# Train the model with default settings
classifier.train()

# Save the trained model
classifier.save_model("path/to/save/model.pkl")
```

### 5.3 Advanced Training with Custom Parameters (Python API)

For more control over the training process:

```python
from nexusml.core.model import EquipmentClassifier

# Create classifier instance
classifier = EquipmentClassifier(sampling_strategy="direct")

# Train with custom parameters
classifier.train(
    data_path="path/to/your/training_data.csv",
    feature_config_path="path/to/custom/feature_config.yml",
    test_size=0.3,
    random_state=42
)

# Save the trained model
classifier.save_model("path/to/save/model.pkl")
```

### 5.4 Training with Hyperparameter Optimization (Python API)

To optimize model hyperparameters:

```python
from nexusml.core.model import EquipmentClassifier
from nexusml.core.model_building import optimize_hyperparameters

# Create classifier instance
classifier = EquipmentClassifier()

# Load data and prepare for training
classifier.load_data("path/to/your/training_data.csv")
x_train, x_test, y_train, y_test = classifier.prepare_data()

# Optimize hyperparameters
optimized_model = optimize_hyperparameters(classifier.model, x_train, y_train)

# Update classifier with optimized model
classifier.model = optimized_model

# Evaluate optimized model
evaluation_results = classifier.evaluate(x_test, y_test)
print(evaluation_results)

# Save the optimized model
classifier.save_model("path/to/save/optimized_model.pkl")
```

## 6. Evaluation and Validation

### 6.1 Using the Model Training Pipeline (Recommended)

The model training pipeline automatically performs comprehensive evaluation and
generates metrics. When using the `--visualize` flag, it also creates
visualizations:

```bash
# Train and evaluate with visualizations
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv --visualize
```

This will generate:

- Evaluation metrics (accuracy, F1 score) for each classification target
- Analysis of "Other" category performance
- Analysis of misclassifications
- Category distribution visualizations
- Confusion matrices for each classification target

The visualizations are saved to the `[output_dir]/visualizations/` directory
(default: `outputs/models/visualizations/`).

### 6.2 Basic Evaluation (Python API)

Evaluate the trained model's performance:

```python
from nexusml.core.model import EquipmentClassifier
from nexusml.core.evaluation import enhanced_evaluation

# Load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/your/model.pkl")

# Evaluate on test data
evaluation_results = classifier.evaluate("path/to/test_data.csv")
print(evaluation_results)
```

### 6.3 Detailed Analysis (Python API)

Perform a more detailed analysis of model performance:

```python
from nexusml.core.model import EquipmentClassifier
from nexusml.core.evaluation import (
    analyze_other_category_features,
    analyze_other_misclassifications,
    visualize_confusion_matrix
)

# Load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/your/model.pkl")

# Load test data
classifier.load_data("path/to/test_data.csv", is_training=False)
x_test, y_test = classifier.get_test_data()

# Make predictions
y_pred = classifier.model.predict(x_test)

# Analyze "Other" category performance
other_analysis = analyze_other_category_features(classifier.model, x_test, y_test)
print(other_analysis)

# Analyze misclassifications
misclassification_analysis = analyze_other_misclassifications(x_test, y_test, y_pred)
print(misclassification_analysis)

# Visualize confusion matrix
visualize_confusion_matrix(y_test, y_pred, "Equipment_Category", "confusion_matrix.png")
```

### 6.4 Interpreting Evaluation Results

When evaluating the model, pay attention to:

1. **Overall Accuracy**: The percentage of correctly classified instances.
2. **F1 Score**: A balance of precision and recall, especially important for
   imbalanced classes.
3. **Confusion Matrix**: Shows where the model is making mistakes between
   classes.
4. **"Other" Category Performance**: How well the model handles items that don't
   fit into specific categories.
5. **Feature Importance**: Which features are most influential in making
   predictions.

The model training pipeline logs these metrics and saves them in the metadata
file alongside the model.

## 7. Using the Trained Model

### 7.1 Using the Model Training Pipeline (Recommended)

The model training pipeline automatically makes a sample prediction after
training:

```bash
# Train and make a sample prediction
python -m nexusml.train_model_pipeline --data-path files/training-data/your_data.csv
```

You can also use the trained model directly for predictions:

```python
from nexusml.core.model import EquipmentClassifier

# Load the latest model trained by the pipeline
classifier = EquipmentClassifier()
classifier.load_model("outputs/models/equipment_classifier_latest.pkl")

# Make a prediction
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0
prediction = classifier.predict(description, service_life)

# Print prediction results
for key, value in prediction.items():
    if key != "attribute_template" and key != "master_db_mapping":
        print(f"{key}: {value}")
```

### 7.2 Making Predictions (Python API)

Use the trained model to classify equipment:

```python
from nexusml.core.model import EquipmentClassifier

# Load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/your/model.pkl")

# Make a prediction
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0
prediction = classifier.predict(description, service_life)

# Print prediction results
for key, value in prediction.items():
    if key != "attribute_template":  # Skip printing the full template
        print(f"{key}: {value}")

# Print attribute template
template = prediction["attribute_template"]
print(f"Equipment Type: {template['equipment_type']}")
print(f"Classification: {template['classification']}")
print("Required Attributes:")
for attr, info in template["required_attributes"].items():
    print(f"  {attr}: {info}")
```

### 7.3 Batch Processing

Process multiple equipment items in batch. The prediction script supports both
legacy and production data formats:

```python
import pandas as pd
from nexusml.core.model import EquipmentClassifier

# Load a trained model (can use the latest model from the pipeline)
classifier = EquipmentClassifier()
classifier.load_model("outputs/models/equipment_classifier_latest.pkl")

# Load batch data
batch_data = pd.read_csv("path/to/batch_data.csv")

# Process each item
results = []
for _, row in batch_data.iterrows():
    # For production format data
    if "equipment_tag" in batch_data.columns and "manufacturer" in batch_data.columns:
        # Create a combined description from multiple columns
        description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"
        service_life = float(row.get("condition_score", 20.0))
        asset_tag = str(row.get("equipment_tag", ""))
    # For legacy format data
    else:
        description = row["Description"]
        service_life = row.get("Service Life", 0.0)
        asset_tag = row.get("Asset Tag", "")

    prediction = classifier.predict(description, service_life, asset_tag)
    results.append(prediction)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("prediction_results.csv", index=False)
```

### 7.4 Using the Prediction Script

The NexusML package includes a dedicated prediction script that supports both
legacy and production data formats:

```python
#!/usr/bin/env python
"""
Equipment Classification Prediction Script

This script loads a trained model and makes predictions on new equipment descriptions.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from nexusml.core.model import EquipmentClassifier

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Make equipment classification predictions")
    parser.add_argument("--model-path", type=str, default="outputs/models/equipment_classifier_latest.pkl",
                        help="Path to the trained model file")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to the input CSV file with equipment descriptions")
    parser.add_argument("--output-file", type=str, default="prediction_results.csv",
                        help="Path to save the prediction results")
    parser.add_argument("--description-column", type=str, default="Description",
                        help="Column name containing equipment descriptions (for legacy format)")
    parser.add_argument("--service-life-column", type=str, default="Service Life",
                        help="Column name containing service life values (for legacy format)")
    parser.add_argument("--asset-tag-column", type=str, default="Asset Tag",
                        help="Column name containing asset tags (for legacy format)")
    args = parser.parse_args()

    # Load the model
    classifier = EquipmentClassifier()
    classifier.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")

    # Load input data
    input_data = pd.read_csv(args.input_file)
    print(f"Loaded {len(input_data)} items from {args.input_file}")

    # Check if we have the production format columns
    has_production_format = all(col in input_data.columns for col in ["equipment_tag", "manufacturer", "model"])

    # Make predictions
    results = []
    for i, row in input_data.iterrows():
        # For production format data
        if has_production_format:
            # Create a combined description from multiple columns
            description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"
            service_life = float(row.get("condition_score", 20.0))
            asset_tag = str(row.get("equipment_tag", ""))
        # For legacy format data
        else:
            description = row[args.description_column]
            service_life = row.get(args.service_life_column, 20.0)
            asset_tag = row.get(args.asset_tag_column, "")

        prediction = classifier.predict(description, service_life, asset_tag)
        results.append(prediction)

        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(input_data)} items")

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    print(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
```

Use the prediction script as follows:

```bash
# For legacy format data
python -m nexusml.predict --input-file path/to/legacy_data.csv --output-file predictions.csv

# For production format data
python -m nexusml.predict --input-file path/to/production_data.csv --output-file predictions.csv
```

### 7.5 Understanding Prediction Warnings

When running the prediction script, you may see warnings like:

```
Warning: Source column 'category_name' not found in DataFrame. Skipping mapping to 'Equipment_Category'.
Warning: Columns ['equipment_tag', 'manufacturer', 'model', 'category_name', 'mcaa_system_category', 'building_name'] not found for TextCombiner. Using available columns only.
No columns available for TextCombiner. Creating empty column combined_text.
```

These warnings indicate that the input data doesn't match the expected format.
The model will still make predictions, but with reduced accuracy due to missing
features. To resolve these warnings:

1. **Check your input data format**: Ensure it matches either the legacy format
   (with 'Description', 'Service Life', etc.) or the production format (with
   'equipment_tag', 'manufacturer', etc.).

2. **For legacy format data**: Make sure your CSV has a 'Description' column
   containing the equipment description.

3. **For production format data**: Ensure all required columns are present
   ('equipment_tag', 'manufacturer', 'model', etc.).

4. **Use the appropriate feature configuration**: If using legacy format data,
   use the legacy feature configuration. If using production format data, use
   the production feature configuration.

5. **Note about column presence warnings**: As of the March 2025 update, if you
   see these warnings but your input data actually contains the columns
   mentioned (like 'category_name'), it may be due to an internal processing
   issue. The fix described in section 9.5 addresses this by using the actual
   values from your input data instead of relying on the model's internal
   feature engineering process.

Despite these warnings, the model will still produce predictions based on the
available information. The sample output shows successful predictions for
equipment categories, system types, and equipment types.

#### 7.5.1 Warnings vs. Actual Data Content

It's important to understand that sometimes warnings about missing columns may
appear even when those columns exist in your input data. This can happen due to
how the model processes data internally:

1. **Input data contains classification columns**: Your CSV has columns like
   'category_name', 'uniformat_code', etc.
2. **Feature engineering process**: The model's internal feature engineering may
   not correctly recognize these columns during certain processing steps.
3. **Warning messages appear**: You see warnings about missing columns despite
   them being present in your input data.
4. **Actual values are used**: With the March 2025 update, the model will still
   use the actual values from your input data for the final prediction, ignoring
   these warnings.

If you're using the latest version of the model (post-March 2025), you can
safely ignore these warnings as long as your input data contains the required
columns and the predictions in the output file match your expected equipment
types.

## 8. Troubleshooting

### 8.1 Pipeline Troubleshooting

When using the model training pipeline, check the log files for detailed
information:

```bash
# View the most recent log file
ls -lt logs/model_training_*.log | head -1 | xargs cat
```

Common pipeline issues:

| Issue                              | Possible Cause                  | Solution                                      |
| ---------------------------------- | ------------------------------- | --------------------------------------------- |
| Pipeline fails to start            | Missing dependencies            | Run `pip install -r requirements.txt`         |
| Data validation fails              | Incorrect data format           | Check data format against required columns    |
| Model training fails               | Memory issues or corrupted data | Try with a smaller dataset or clean data      |
| Visualization errors               | Missing matplotlib dependencies | Install matplotlib and seaborn                |
| Permission denied for shell script | Script not executable           | Run `chmod +x nexusml/scripts/train_model.sh` |

### 8.2 Common Issues and Solutions

| Issue                               | Possible Cause                          | Solution                                      |
| ----------------------------------- | --------------------------------------- | --------------------------------------------- |
| Missing reference data              | Incorrect paths in reference_config.yml | Update paths to point to correct files        |
| Low classification accuracy         | Insufficient training data              | Add more diverse training examples            |
| "Other" category overrepresentation | Imbalanced training data                | Use balanced sampling or adjust class weights |
| Missing features                    | Incorrect feature configuration         | Update feature_config.yml                     |
| Model fails to load                 | Incompatible pickle version             | Retrain model with current environment        |
| NaN values in predictions           | Missing required features               | Check feature engineering configuration       |
| Slow training performance           | Large dataset or complex model          | Use a smaller dataset or simplify the model   |
| Column not found warnings           | Input data format mismatch              | Ensure data format matches configuration      |

### 8.3 Validating Reference Data

If you suspect issues with reference data:

```python
from nexusml.core.reference.manager import ReferenceManager

# Initialize reference manager
ref_manager = ReferenceManager()

# Validate all reference data sources
validation_results = ref_manager.validate_data()

# Print validation results
for source, result in validation_results.items():
    print(f"Validation results for {source}:")
    print(f"  Status: {'Valid' if result['valid'] else 'Invalid'}")
    if not result['valid']:
        print(f"  Issues: {result['issues']}")
```

### 8.4 Debugging the Model

For more detailed debugging of the model:

```python
from nexusml.core.model import EquipmentClassifier
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
classifier = EquipmentClassifier()
classifier.load_model("outputs/models/equipment_classifier_latest.pkl")

# Inspect model internals
print(f"Model type: {type(classifier.model)}")
print(f"Feature names: {classifier.model.feature_names_in_}")
print(f"Target names: {classifier.model.classes_}")

# Make a prediction with verbose output
prediction = classifier.predict("Heat Exchanger", 20.0, verbose=True)
```

## 9. Recent Updates and Improvements

### 9.1 Production Data Format Support

The model training pipeline has been updated to support the production data
format, which uses actual database column names and includes classification IDs
directly. This format is recommended for all new training data.

Key improvements:

- Direct support for database column names (equipment_tag, manufacturer, model,
  etc.)
- Direct mapping of classification IDs (OmniClassID, UniFormatID, etc.)
- Simplified feature engineering with direct mappings instead of EAV integration
- Improved prediction accuracy with direct classification codes

### 9.2 Data Mapper Updates

The data mapper has been updated to support both legacy and production data
formats:

- Automatic detection of data format based on column names
- Mapping of production format columns to model input format
- Direct use of classification IDs when available
- Fallback to EAV integration for legacy data

### 9.3 Model Prediction Improvements

The prediction functionality has been enhanced to support both data formats:

- Updated `predict.py` script with support for both formats
- Automatic format detection based on column names
- Improved error handling and logging
- Backward compatibility with legacy format

### 9.4 Handling Warning Messages

The updated pipeline includes improved warning handling:

- Clear warning messages for missing columns
- Graceful fallback to default values when columns are missing
- Detailed logging of feature engineering steps
- Support for mixed format data with partial column sets

### 9.5 Input Data Alignment Fix (March 2025)

A significant improvement has been made to address the misalignment between
input data and model predictions. Previously, the model would ignore the actual
equipment types in the input data and make predictions based solely on text
descriptions, leading to misclassifications.

#### 9.5.1 Issue Description

When using the prediction script with input data that already contained
equipment classification information (such as `category_name`, `uniformat_code`,
etc.), the model would:

- Ignore these existing classifications
- Make new predictions based only on the text description
- Generate warnings about missing columns despite them being present in the
  input data

This resulted in misclassifications where, for example, a "Rooftop Unit" might
be classified as an "Exhaust Fan" despite the correct classification being
present in the input data.

#### 9.5.2 Solution Implemented

The `predict_from_row` method in the `EquipmentClassifier` class has been
modified to use the actual values from the input data instead of making
predictions when the data already contains classification information:

```python
# Instead of making predictions, use the actual values from the input data
result = {
    "category_name": row.get("category_name", "Unknown"),
    "uniformat_code": row.get("uniformat_code", ""),
    "mcaa_system_category": row.get("mcaa_system_category", ""),
    "Equipment_Type": row.get("Equipment_Type", ""),
    "System_Subtype": row.get("System_Subtype", ""),
    "Asset Tag": asset_tag,
}
```

This change ensures that the model uses the actual equipment types, codes, and
categories from the input data, rather than trying to predict them.

#### 9.5.3 Fields Being Predicted

The model now handles the following fields differently:

1. **Fields Used Directly from Input Data (when available):**

   - `category_name` - The equipment category (e.g., "Rooftop Unit", "Pump",
     "Chiller")
   - `uniformat_code` - The Uniformat classification code (e.g., "D3050",
     "D2020")
   - `mcaa_system_category` - The MCAA system category (e.g., "HVAC Equipment",
     "Plumbing Equipment")
   - `Equipment_Type` - The hierarchical equipment type (e.g., "HVAC
     Equipment-Rooftop Unit")
   - `System_Subtype` - The system subtype (e.g., "HVAC Equipment-Rooftop Unit")

2. **Fields Still Predicted by the Model:**
   - `MasterFormat_Class` - Derived from other classifications using the
     `enhanced_masterformat_mapping` function
   - `attribute_template` - Generated based on the equipment type
   - `master_db_mapping` - Mapping to master database fields

This approach ensures that existing classification data is preserved while still
providing the benefits of the model's attribute templates and database mappings.

## 10. References

- NexusML Documentation
- Model Training Pipeline (`nexusml/train_model_pipeline.py`)
- Model Training Shell Script (`nexusml/scripts/train_model.sh`)
- Feature Engineering Documentation (`nexusml/core/feature_engineering.py`)
- Model Building Documentation (`nexusml/core/model_building.py`)
- Reference Manager Documentation (`nexusml/core/reference/manager.py`)
- EAV Manager Documentation (`nexusml/core/eav_manager.py`)
- Evaluation Documentation (`nexusml/core/evaluation.py`)
- Data Mapper Documentation (`nexusml/core/data_mapper.py`)
