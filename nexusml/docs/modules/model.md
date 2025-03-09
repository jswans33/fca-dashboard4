# Module: model

## Overview

The `model` module implements a comprehensive machine learning pipeline for classifying equipment based on text descriptions and numeric features, with integrated EAV (Entity-Attribute-Value) structure. This module serves as the central component of the NexusML system, bringing together various other modules to create a complete equipment classification solution.

Key features include:

1. **Combined Text and Numeric Features**:
   - Uses a ColumnTransformer to incorporate both text features (via TF-IDF) and numeric features (like service_life) into a single model.

2. **Improved Handling of Imbalanced Classes**:
   - Uses class_weight='balanced_subsample' in the RandomForestClassifier for handling imbalanced classes.

3. **Better Evaluation Metrics**:
   - Uses f1_macro scoring for hyperparameter optimization, which is more appropriate for imbalanced classes than accuracy.
   - Provides detailed analysis of "Other" category performance.

4. **Feature Importance Analysis**:
   - Analyzes the importance of both text and numeric features in classifying equipment.

5. **EAV Integration**:
   - Incorporates EAV structure for flexible equipment attributes
   - Uses classification systems (OmniClass, MasterFormat, Uniformat) for comprehensive taxonomy
   - Includes performance fields (service life, maintenance intervals) in feature engineering
   - Can predict missing attribute values based on equipment descriptions

## Classes

### Class: EquipmentClassifier

Comprehensive equipment classifier with EAV integration.

#### Attributes

- `model`: Trained ML model
- `feature_engineer` (GenericFeatureEngineer): Feature engineering transformer
- `eav_manager` (EAVManager): EAV manager for attribute templates
- `sampling_strategy` (str): Strategy for handling class imbalance

#### Methods

##### `__init__(model=None, feature_engineer: Optional[GenericFeatureEngineer] = None, eav_manager: Optional[EAVManager] = None, sampling_strategy: str = "direct")`

Initialize the equipment classifier.

**Parameters:**

- `model` (optional): Trained ML model. If None, needs to be trained.
- `feature_engineer` (Optional[GenericFeatureEngineer], optional): Feature engineering transformer. If None, creates a new one.
- `eav_manager` (Optional[EAVManager], optional): EAV manager for attribute templates. If None, creates a new one.
- `sampling_strategy` (str, optional): Strategy for handling class imbalance. Default is "direct".

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier with default settings
classifier = EquipmentClassifier()

# Create a classifier with a pre-trained model
import pickle
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)
classifier = EquipmentClassifier(model=model)
```

##### `train(data_path: Optional[str] = None, feature_config_path: Optional[str] = None, sampling_strategy: str = "direct", **kwargs) -> None`

Train the equipment classifier.

**Parameters:**

- `data_path` (Optional[str], optional): Path to the training data. If None, uses the default path.
- `feature_config_path` (Optional[str], optional): Path to the feature configuration. If None, uses the default path.
- `sampling_strategy` (str, optional): Strategy for handling class imbalance. Default is "direct".
- `**kwargs`: Additional parameters for training.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Train with default settings
classifier.train()

# Train with custom data and feature configuration
classifier.train(
    data_path="path/to/custom_data.csv",
    feature_config_path="path/to/custom_feature_config.yml",
    sampling_strategy="direct"
)
```

##### `load_model(model_path: str) -> None`

Load a trained model from a file.

**Parameters:**

- `model_path` (str): Path to the saved model file.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Load a trained model
classifier.load_model("path/to/trained_model.pkl")
```

##### `predict(description_or_data: Union[str, pd.DataFrame], service_life: float = 0.0, asset_tag: str = "") -> Union[Dict[str, Any], pd.DataFrame]`

Predict equipment classifications from a description or DataFrame.

**Parameters:**

- `description_or_data` (Union[str, pd.DataFrame]): Text description of the equipment or DataFrame with features.
- `service_life` (float, optional): Service life value. Default is 0.0. Only used with string description.
- `asset_tag` (str, optional): Asset tag for equipment. Default is "". Only used with string description.

**Returns:**

- Union[Dict[str, Any], pd.DataFrame]: Dictionary with classification results or DataFrame with predictions for each row.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier and load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/trained_model.pkl")

# Predict from a text description
description = "500 ton centrifugal chiller with R-134a refrigerant"
prediction = classifier.predict(description, service_life=20.0, asset_tag="CH-01")
print(prediction)

# Predict from a DataFrame
import pandas as pd
data = pd.DataFrame({
    "combined_text": [
        "500 ton centrifugal chiller with R-134a refrigerant",
        "10,000 CFM air handler with MERV 13 filters"
    ],
    "service_life": [20.0, 15.0]
})
predictions = classifier.predict(data)
print(predictions)
```

**Notes:**

- If a string description is provided, returns a dictionary with classification results and attribute template.
- If a DataFrame is provided, returns a DataFrame with predictions for each row.
- The method can handle both raw text descriptions and pre-processed features.
- For string input, the method adds an EAV attribute template for the predicted equipment type.

##### `predict_from_row(row: pd.Series) -> Dict[str, Any]`

Predict equipment classifications from a DataFrame row.

**Parameters:**

- `row` (pd.Series): A pandas Series representing a row from a DataFrame.

**Returns:**

- Dict[str, Any]: Dictionary with classification results and master DB mappings.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier
import pandas as pd

# Create a classifier and load a trained model
classifier = EquipmentClassifier()
classifier.load_model("path/to/trained_model.pkl")

# Create a DataFrame with equipment data
data = pd.DataFrame({
    "equipment_tag": ["CH-01"],
    "manufacturer": ["Carrier"],
    "model": ["30XA"],
    "category_name": ["Chiller"],
    "mcaa_system_category": ["HVAC"],
    "service_life": [20.0]
})

# Predict from a row
prediction = classifier.predict_from_row(data.iloc[0])
print(prediction)
```

**Notes:**

- This method is designed to work with rows that have already been processed by the feature engineering pipeline.
- It extracts the description, service life, and asset tag from the row.
- Instead of making predictions, it uses the actual values from the input data.
- It adds an EAV attribute template for the equipment type.
- It maps the predictions to master database fields.

##### `predict_attributes(equipment_type: str, description: str) -> Dict[str, Any]`

Predict attribute values for a given equipment type and description.

**Parameters:**

- `equipment_type` (str): Type of equipment.
- `description` (str): Text description of the equipment.

**Returns:**

- Dict[str, Any]: Dictionary with predicted attribute values.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Predict attribute values
equipment_type = "Chiller"
description = "500 ton centrifugal chiller with R-134a refrigerant"
attributes = classifier.predict_attributes(equipment_type, description)
print(attributes)
```

**Notes:**

- This is a rule-based attribute prediction based on keywords in the description.
- It extracts numeric values with units from the description using regular expressions.
- It looks for patterns like "100 tons", "5 HP", etc.
- It checks for equipment types like "centrifugal", "absorption", etc.
- In a real implementation, this would use ML to predict attribute values.

##### `fill_missing_attributes(equipment_type: str, attributes: Dict[str, Any], description: str) -> Dict[str, Any]`

Fill in missing attributes using ML predictions and rules.

**Parameters:**

- `equipment_type` (str): Type of equipment.
- `attributes` (Dict[str, Any]): Dictionary of existing attribute name-value pairs.
- `description` (str): Text description of the equipment.

**Returns:**

- Dict[str, Any]: Dictionary with filled attributes.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Define some attributes for a chiller (with missing attributes)
chiller_attrs = {
    "cooling_capacity_tons": 500,
    "chiller_type": None,  # Missing value
    "refrigerant": None  # Missing value
}

# Fill missing attributes
equipment_type = "Chiller"
description = "500 ton centrifugal chiller with R-134a refrigerant"
filled_attrs = classifier.fill_missing_attributes(
    equipment_type, chiller_attrs, description
)
print(filled_attrs)
```

**Notes:**

- This method uses the EAVManager to fill missing attributes.
- It delegates to the `fill_missing_attributes` method of the EAVManager.
- The EAVManager can use the classifier's `predict_attributes` method to predict missing values.

##### `validate_attributes(equipment_type: str, attributes: Dict[str, Any]) -> Dict[str, List[str]]`

Validate attributes against the template for an equipment type.

**Parameters:**

- `equipment_type` (str): Type of equipment.
- `attributes` (Dict[str, Any]): Dictionary of attribute name-value pairs.

**Returns:**

- Dict[str, List[str]]: Dictionary with validation results.

**Example:**
```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Define some attributes for a chiller
chiller_attrs = {
    "cooling_capacity_tons": 500,
    "chiller_type": "Centrifugal",
    "refrigerant": "R-134a",
    "custom_field": "Some value"  # Not in template
}

# Validate attributes
equipment_type = "Chiller"
validation_results = classifier.validate_attributes(equipment_type, chiller_attrs)
print(validation_results)
```

**Notes:**

- This method uses the EAVManager to validate attributes.
- It delegates to the `validate_attributes` method of the EAVManager.
- The validation results include missing required attributes and unknown attributes.

## Functions

### `train_enhanced_model(data_path: Optional[str] = None, sampling_strategy: str = "direct", feature_config_path: Optional[str] = None, eav_manager: Optional[EAVManager] = None, feature_engineer: Optional[GenericFeatureEngineer] = None, **kwargs) -> Tuple[Any, pd.DataFrame]`

Train and evaluate the enhanced model with EAV integration.

**Parameters:**

- `data_path` (Optional[str], optional): Path to the CSV file. If None, uses the standard location.
- `sampling_strategy` (str, optional): Strategy for handling class imbalance. Default is "direct".
- `feature_config_path` (Optional[str], optional): Path to the feature configuration file. If None, uses the standard location.
- `eav_manager` (Optional[EAVManager], optional): EAVManager instance. If None, uses the one from the DI container.
- `feature_engineer` (Optional[GenericFeatureEngineer], optional): GenericFeatureEngineer instance. If None, uses the one from the DI container.
- `**kwargs`: Additional parameters for the model.

**Returns:**

- Tuple[Any, pd.DataFrame]: Tuple containing the trained model and preprocessed dataframe.

**Example:**
```python
from nexusml.core.model import train_enhanced_model

# Train a model with default settings
model, df = train_enhanced_model()

# Train a model with custom settings
model, df = train_enhanced_model(
    data_path="path/to/custom_data.csv",
    sampling_strategy="direct",
    feature_config_path="path/to/custom_feature_config.yml"
)

# Save the trained model
import pickle
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

**Notes:**

- This function performs the following steps:
  1. Load and preprocess data
  2. Map staging data columns to model input format
  3. Apply Generic Feature Engineering with EAV integration
  4. Prepare training data with both text and numeric features
  5. Split the data into training and testing sets
  6. Print class distribution information
  7. Build enhanced model with balanced class weights
  8. Train the model
  9. Evaluate with focus on "Other" categories
  10. Analyze "Other" category features
  11. Analyze misclassifications for "Other" categories
- The function returns the trained model and the preprocessed dataframe.

### `predict_with_enhanced_model(model: Any, description: str, service_life: float = 0.0, asset_tag: str = "", eav_manager: Optional[EAVManager] = None) -> dict`

Make predictions with enhanced detail for "Other" categories.

**Parameters:**

- `model` (Any): Trained model pipeline.
- `description` (str): Text description to classify.
- `service_life` (float, optional): Service life value. Default is 0.0.
- `asset_tag` (str, optional): Asset tag for equipment. Default is "".
- `eav_manager` (Optional[EAVManager], optional): EAV manager instance. If None, uses the one from the DI container.

**Returns:**

- dict: Prediction results with classifications and master DB mappings.

**Example:**
```python
from nexusml.core.model import predict_with_enhanced_model
import pickle

# Load a trained model
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make a prediction
description = "500 ton centrifugal chiller with R-134a refrigerant"
prediction = predict_with_enhanced_model(
    model, description, service_life=20.0, asset_tag="CH-01"
)
print(prediction)
```

**Notes:**

- This function creates a DataFrame with the required structure for the pipeline.
- It predicts using the trained pipeline and extracts the predictions.
- It adds MasterFormat prediction with enhanced mapping.
- It adds EAV template information, including classification IDs and performance fields.
- It maps predictions to master database fields.

### `visualize_category_distribution(df: pd.DataFrame, output_dir: str = "outputs") -> Tuple[str, str]`

Visualize the distribution of categories in the dataset.

**Parameters:**

- `df` (pd.DataFrame): DataFrame with category columns.
- `output_dir` (str, optional): Directory to save visualizations. Default is "outputs".

**Returns:**

- Tuple[str, str]: Paths to the saved visualization files.

**Example:**
```python
from nexusml.core.model import visualize_category_distribution
import pandas as pd

# Load data
df = pd.read_csv("path/to/data.csv")

# Visualize category distribution
equipment_category_file, system_type_file = visualize_category_distribution(df)
print(f"Visualizations saved to: {equipment_category_file}, {system_type_file}")
```

**Notes:**

- This function creates two visualizations:
  1. Equipment Category Distribution
  2. System Type Distribution
- It saves the visualizations as PNG files in the specified output directory.
- It returns the paths to the saved visualization files.

### `train_model_from_any_data(df: pd.DataFrame, mapper: Optional[DynamicFieldMapper] = None)`

Train a model using data with any column structure.

**Parameters:**

- `df` (pd.DataFrame): Input DataFrame with arbitrary column names.
- `mapper` (Optional[DynamicFieldMapper], optional): Optional DynamicFieldMapper instance. If None, creates a new one.

**Returns:**

- Tuple: (trained model, transformed DataFrame)

**Example:**
```python
from nexusml.core.model import train_model_from_any_data
import pandas as pd

# Load data with arbitrary column names
df = pd.read_csv("path/to/data_with_arbitrary_columns.csv")

# Train a model
model, transformed_df = train_model_from_any_data(df)

# Save the trained model
import pickle
with open("trained_model.pkl", "wb") as f:
    pickle.dump(model, f)
```

**Notes:**

- This function is designed to work with data that has arbitrary column names.
- It uses a DynamicFieldMapper to map the input fields to the expected model fields.
- It applies Generic Feature Engineering with EAV integration.
- It prepares training data with both text and numeric features.
- It splits the data into training and testing sets.
- It builds and trains an enhanced model with balanced class weights.
- It evaluates the model with focus on "Other" categories.
- It returns the trained model and the transformed DataFrame.

## Usage Examples

### Basic Usage

```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Train the classifier
classifier.train()

# Make a prediction
description = "500 ton centrifugal chiller with R-134a refrigerant"
prediction = classifier.predict(description, service_life=20.0, asset_tag="CH-01")

# Print the prediction
print("Prediction:")
for key, value in prediction.items():
    if key != "attribute_template":  # Skip printing the full template
        print(f"{key}: {value}")

# Print the attribute template
print("\nAttribute Template:")
template = prediction["attribute_template"]
print(f"Equipment Type: {template['equipment_type']}")
print(f"Classification: {template['classification']}")
print("Required Attributes:")
for attr, info in template["required_attributes"].items():
    print(f"  {attr}: {info}")
```

### Loading a Pre-trained Model

```python
from nexusml.core.model import EquipmentClassifier
import pandas as pd

# Create a classifier
classifier = EquipmentClassifier()

# Load a pre-trained model
classifier.load_model("path/to/trained_model.pkl")

# Make predictions for multiple equipment descriptions
data = pd.DataFrame({
    "combined_text": [
        "500 ton centrifugal chiller with R-134a refrigerant",
        "10,000 CFM air handler with MERV 13 filters",
        "100 GPM circulation pump"
    ],
    "service_life": [20.0, 15.0, 10.0]
})

# Make predictions
predictions = classifier.predict(data)
print(predictions)
```

### Predicting and Filling Attributes

```python
from nexusml.core.model import EquipmentClassifier

# Create a classifier
classifier = EquipmentClassifier()

# Load a pre-trained model
classifier.load_model("path/to/trained_model.pkl")

# Define equipment type and description
equipment_type = "Chiller"
description = "500 ton centrifugal chiller with R-134a refrigerant"

# Define some attributes with missing values
attributes = {
    "cooling_capacity_tons": 500,
    "chiller_type": None,  # Missing value
    "refrigerant": None,  # Missing value
    "eer": None  # Missing value
}

# Predict attribute values
predicted_attrs = classifier.predict_attributes(equipment_type, description)
print("Predicted Attributes:")
print(predicted_attrs)

# Fill missing attributes
filled_attrs = classifier.fill_missing_attributes(equipment_type, attributes, description)
print("\nFilled Attributes:")
print(filled_attrs)

# Validate attributes
validation_results = classifier.validate_attributes(equipment_type, filled_attrs)
print("\nValidation Results:")
print(validation_results)
```

### Training from Custom Data

```python
from nexusml.core.model import train_model_from_any_data
import pandas as pd

# Load custom data
df = pd.read_csv("path/to/custom_data.csv")

# Print column names
print("Original Columns:")
print(df.columns)

# Train a model from custom data
model, transformed_df = train_model_from_any_data(df)

# Print transformed columns
print("\nTransformed Columns:")
print(transformed_df.columns)

# Save the trained model
import pickle
with open("custom_trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Create a classifier with the trained model
from nexusml.core.model import EquipmentClassifier
classifier = EquipmentClassifier(model=model)

# Make a prediction
description = "500 ton centrifugal chiller with R-134a refrigerant"
prediction = classifier.predict(description, service_life=20.0)
print("\nPrediction:")
for key, value in prediction.items():
    if key != "attribute_template":  # Skip printing the full template
        print(f"{key}: {value}")
```

### Visualizing Category Distribution

```python
from nexusml.core.model import EquipmentClassifier, visualize_category_distribution

# Create a classifier
classifier = EquipmentClassifier()

# Train the classifier
classifier.train()

# Visualize category distribution
equipment_category_file, system_type_file = visualize_category_distribution(
    classifier.df, output_dir="visualizations"
)

print("Visualizations saved to:")
print(f"  - {equipment_category_file}")
print(f"  - {system_type_file}")

# Display the visualizations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(12, 10))
plt.subplot(1, 2, 1)
img1 = mpimg.imread(equipment_category_file)
plt.imshow(img1)
plt.axis('off')
plt.title('Equipment Category Distribution')

plt.subplot(1, 2, 2)
img2 = mpimg.imread(system_type_file)
plt.imshow(img2)
plt.axis('off')
plt.title('System Type Distribution')

plt.tight_layout()
plt.show()
```

## Dependencies

- **os**: Used for file operations
- **typing**: Used for type hints
- **matplotlib.pyplot**: Used for visualizations
- **numpy**: Used for numerical operations
- **pandas**: Used for DataFrame operations
- **seaborn**: Used for visualizations
- **sklearn.model_selection**: Used for train_test_split
- **pickle**: Used for model serialization
- **re**: Used for regular expressions in attribute prediction
- **nexusml.core.data_mapper**: Used for mapping predictions to master DB
- **nexusml.core.data_preprocessing**: Used for loading and preprocessing data
- **nexusml.core.di.decorators**: Used for dependency injection
- **nexusml.core.dynamic_mapper**: Used for dynamic field mapping
- **nexusml.core.eav_manager**: Used for EAV structure
- **nexusml.core.evaluation**: Used for model evaluation
- **nexusml.core.feature_engineering**: Used for feature engineering
- **nexusml.core.model_building**: Used for building the model

## Notes and Warnings

- The module uses dependency injection for better testability and configurability
- The EquipmentClassifier class is the main entry point for equipment classification
- The model expects specific columns after feature engineering:
  - "combined_text" or "combined_features": Text features combined from multiple columns
  - "service_life": Numeric feature representing service life
- The predict_attributes method uses a rule-based approach for attribute prediction
  - In a real implementation, this would use ML to predict attribute values
- The module integrates with the EAV structure for flexible equipment attributes
- The module uses classification systems (OmniClass, MasterFormat, Uniformat) for comprehensive taxonomy
- The module includes performance fields (service life, maintenance intervals) in feature engineering
- The module can predict missing attribute values based on equipment descriptions
