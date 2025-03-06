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
transformations:

```yaml
# Example feature_config.yml
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

## 4. Data Preparation

### 4.1 Prepare Training Data

Ensure your training data is in the correct format. The model expects a CSV or
Excel file with the following columns:

- Asset Name
- Asset Tag
- Trade
- System Category
- Sub System Type
- Sub System Classification
- Manufacturer
- Model Number
- Service Life

Place your training data in the appropriate location (default:
`files/training-data/`).

### 4.2 Data Validation

Validate your training data to ensure it meets quality standards:

```python
from nexusml.core.data_preprocessing import validate_training_data

# Validate training data
validation_results = validate_training_data("path/to/your/training_data.csv")
print(validation_results)
```

## 5. Model Training Process

### 5.1 Basic Training

To train the model with default settings:

```python
from nexusml.core.model import EquipmentClassifier

# Create classifier instance
classifier = EquipmentClassifier()

# Train the model with default settings
classifier.train()

# Save the trained model
classifier.save_model("path/to/save/model.pkl")
```

### 5.2 Advanced Training with Custom Parameters

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

### 5.3 Training with Hyperparameter Optimization

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

### 6.1 Basic Evaluation

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

### 6.2 Detailed Analysis

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

## 7. Using the Trained Model

### 7.1 Making Predictions

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

### 7.2 Batch Processing

Process multiple equipment items in batch:

```python
import pandas as pd
from nexusml.core.model import EquipmentClassifier

# Load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/your/model.pkl")

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

## 8. Troubleshooting

### 8.1 Common Issues and Solutions

| Issue                               | Possible Cause                          | Solution                                      |
| ----------------------------------- | --------------------------------------- | --------------------------------------------- |
| Missing reference data              | Incorrect paths in reference_config.yml | Update paths to point to correct files        |
| Low classification accuracy         | Insufficient training data              | Add more diverse training examples            |
| "Other" category overrepresentation | Imbalanced training data                | Use balanced sampling or adjust class weights |
| Missing features                    | Incorrect feature configuration         | Update feature_config.yml                     |
| Model fails to load                 | Incompatible pickle version             | Retrain model with current environment        |

### 8.2 Validating Reference Data

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

## 9. References

- NexusML Documentation
- Feature Engineering Documentation (`nexusml/core/feature_engineering.py`)
- Model Building Documentation (`nexusml/core/model_building.py`)
- Reference Manager Documentation (`nexusml/core/reference/manager.py`)
- EAV Manager Documentation (`nexusml/core/eav_manager.py`)
