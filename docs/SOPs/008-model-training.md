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

Validate your training data to ensure it meets quality standards:

```python
from nexusml.core.data_preprocessing import validate_training_data

# Validate training data
validation_results = validate_training_data("path/to/your/training_data.csv")
print(validation_results)
```

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

Process multiple equipment items in batch:

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
    description = row["Description"]
    service_life = row.get("Service Life", 0.0)
    asset_tag = row.get("Asset Tag", "")

    prediction = classifier.predict(description, service_life, asset_tag)
    results.append(prediction)

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv("prediction_results.csv", index=False)
```

### 7.4 Creating a Prediction Script

For regular use, you can create a dedicated prediction script:

```python
#!/usr/bin/env python
"""
Equipment Classification Prediction Script

This script loads a trained model and makes predictions on new equipment descriptions.
"""

import argparse
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
    args = parser.parse_args()

    # Load the model
    classifier = EquipmentClassifier()
    classifier.load_model(args.model_path)
    print(f"Model loaded from: {args.model_path}")

    # Load input data
    input_data = pd.read_csv(args.input_file)
    print(f"Loaded {len(input_data)} items from {args.input_file}")

    # Make predictions
    results = []
    for i, row in input_data.iterrows():
        description = row["Description"]
        service_life = row.get("Service Life", 20.0)
        asset_tag = row.get("Asset Tag", "")

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

Save this script as `nexusml/predict.py` and use it as follows:

```bash
# Make predictions on a batch of equipment descriptions
python -m nexusml.predict --input-file path/to/equipment_data.csv --output-file predictions.csv
```

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

## 9. References

- NexusML Documentation
- Model Training Pipeline (`nexusml/train_model_pipeline.py`)
- Model Training Shell Script (`nexusml/scripts/train_model.sh`)
- Feature Engineering Documentation (`nexusml/core/feature_engineering.py`)
- Model Building Documentation (`nexusml/core/model_building.py`)
- Reference Manager Documentation (`nexusml/core/reference/manager.py`)
- EAV Manager Documentation (`nexusml/core/eav_manager.py`)
- Evaluation Documentation (`nexusml/core/evaluation.py`)
