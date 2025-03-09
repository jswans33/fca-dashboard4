# Module: model_building

## Overview

The `model_building` module defines the core model architecture for the equipment classification model in the NexusML system. It follows the Single Responsibility Principle by focusing solely on model definition and hyperparameter optimization. This module provides functionality to:

1. Build a comprehensive machine learning pipeline with feature engineering, text processing, and classification components
2. Optimize hyperparameters for better handling of all classes, including the "Other" category
3. Configure model parameters through YAML configuration files

The module creates a sophisticated model that incorporates both text features (via TF-IDF) and numeric features (like service life) using a ColumnTransformer to create a comprehensive feature representation.

## Functions

### `build_enhanced_model(sampling_strategy: str = "direct", feature_config_path: Optional[str] = None, **kwargs) -> Pipeline`

Build an enhanced model with configurable sampling strategy.

**Parameters:**

- `sampling_strategy` (str, optional): Sampling strategy to use. Currently, only "direct" is supported. Default is "direct".
- `feature_config_path` (Optional[str], optional): Path to the feature configuration file. If None, uses the default path.
- `**kwargs`: Additional parameters for the model.

**Returns:**

- Pipeline: Scikit-learn pipeline with feature engineering, preprocessor, and classifier.

**Example:**
```python
from nexusml.core.model_building import build_enhanced_model

# Build a model with default settings
model = build_enhanced_model()

# Build a model with custom feature configuration
model = build_enhanced_model(feature_config_path="path/to/custom_feature_config.yml")
```

**Notes:**

- The function tries to load settings from a configuration file (settings.yml) in the following order:
  1. From fca_dashboard.utils.path_util.get_config_path
  2. From nexusml/config/settings.yml
  3. From the path specified in the NEXUSML_CONFIG environment variable
  4. If no settings file is found, default values are used
- The pipeline consists of three main components:
  1. Feature engineering: GenericFeatureEngineer for transforming raw features
  2. Preprocessor: ColumnTransformer combining text and numeric features
     - Text features: TF-IDF vectorization of the "combined_features" column
     - Numeric features: StandardScaler for the "service_life" column
  3. Classifier: MultiOutputClassifier with RandomForestClassifier
- The TF-IDF vectorizer settings include:
  - max_features: Maximum number of features (default: 5000)
  - ngram_range: Range of n-grams to consider (default: (1, 3))
  - min_df: Minimum document frequency (default: 2)
  - max_df: Maximum document frequency (default: 0.9)
  - use_idf: Whether to use inverse document frequency (default: True)
  - sublinear_tf: Whether to apply sublinear scaling to term frequencies (default: True)
- The RandomForestClassifier settings include:
  - n_estimators: Number of trees (default: 200)
  - max_depth: Maximum depth of trees (default: None, unlimited)
  - min_samples_split: Minimum samples required to split a node (default: 2)
  - min_samples_leaf: Minimum samples required at a leaf node (default: 1)
  - class_weight: Weight for handling imbalanced classes (default: "balanced_subsample")
  - random_state: Random seed for reproducibility (default: 42)

### `optimize_hyperparameters(pipeline: Pipeline, x_train, y_train) -> Pipeline`

Optimize hyperparameters for better handling of all classes including "Other".

**Parameters:**

- `pipeline` (Pipeline): Model pipeline to optimize
- `x_train`: Training features
- `y_train`: Training targets

**Returns:**

- Pipeline: Optimized pipeline

**Example:**
```python
from nexusml.core.model_building import build_enhanced_model, optimize_hyperparameters
import pandas as pd

# Prepare training data
x_train = pd.DataFrame({
    "combined_features": ["Description of equipment 1", "Description of equipment 2"],
    "service_life": [15, 20]
})
y_train = pd.DataFrame({
    "Equipment_Category": ["Chiller", "Air Handler"],
    "Equipment_Type": ["Centrifugal", "VAV"]
})

# Build initial model
model = build_enhanced_model()

# Optimize hyperparameters
optimized_model = optimize_hyperparameters(model, x_train, y_train)
```

**Notes:**

- This function uses GridSearchCV to find the best hyperparameters for the model
- It optimizes both the text processing parameters and the classifier parameters
- The scoring metric is f1_macro, which is better for handling imbalanced classes than accuracy
- The parameter grid includes:
  - preprocessor__text__tfidf__max_features: [3000, 5000, 7000]
  - preprocessor__text__tfidf__ngram_range: [(1, 2), (1, 3)]
  - clf__estimator__n_estimators: [100, 200, 300]
  - clf__estimator__min_samples_leaf: [1, 2, 4]
- The input x_train must be a DataFrame with both 'combined_features' and 'service_life' columns
- The function prints the best parameters and cross-validation score

## Usage Examples

### Basic Model Building

```python
from nexusml.core.model_building import build_enhanced_model
import pandas as pd

# Create a simple dataset
data = pd.DataFrame({
    "Asset Category": ["Chiller", "Air Handler", "Pump"],
    "Equip Name ID": ["Centrifugal", "VAV", "Circulation"],
    "Description": ["500 ton", "10,000 CFM", "100 GPM"],
    "Service Life": [20, 15, 10]
})

# Build the model
model = build_enhanced_model()

# The model expects specific columns after feature engineering
# In a real scenario, you would apply feature engineering first
# For this example, we'll manually create the expected columns
processed_data = pd.DataFrame({
    "combined_features": [
        "Chiller Centrifugal 500 ton",
        "Air Handler VAV 10,000 CFM",
        "Pump Circulation 100 GPM"
    ],
    "service_life": [20, 15, 10]
})

# Make predictions
predictions = model.predict(processed_data)
print(predictions)
```

### Complete Model Training and Optimization

```python
from nexusml.core.model_building import build_enhanced_model, optimize_hyperparameters
from nexusml.core.feature_engineering import enhance_features
import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw data
data = pd.read_csv("equipment_data.csv")

# Apply feature engineering
processed_data = enhance_features(data)

# Prepare features and targets
X = processed_data[["combined_features", "service_life"]]
y = processed_data[["Equipment_Category", "Equipment_Type"]]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build initial model
model = build_enhanced_model()

# Optimize hyperparameters
print("Optimizing hyperparameters...")
optimized_model = optimize_hyperparameters(model, X_train, y_train)

# Evaluate the model
from sklearn.metrics import classification_report
y_pred = optimized_model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

for col in y_test.columns:
    print(f"\n{col} Classification Report:")
    print(classification_report(y_test[col], y_pred_df[col]))

# Save the optimized model
import pickle
with open("optimized_model.pkl", "wb") as f:
    pickle.dump(optimized_model, f)
```

### Custom Configuration

```python
from nexusml.core.model_building import build_enhanced_model
import pandas as pd
import os

# Set custom configuration path
os.environ["NEXUSML_CONFIG"] = "path/to/custom_settings.yml"

# Build model with custom feature configuration
model = build_enhanced_model(feature_config_path="path/to/custom_feature_config.yml")

# Process data and make predictions
# ...
```

## Configuration File Format

The model_building module uses a YAML configuration file (settings.yml) to define model parameters. The default path is "config/settings.yml" relative to the project root.

### Example Configuration

```yaml
classifier:
  tfidf:
    max_features: 5000
    ngram_range: [1, 3]
    min_df: 2
    max_df: 0.9
    use_idf: true
    sublinear_tf: true
  
  model:
    random_forest:
      n_estimators: 200
      max_depth: null  # null means unlimited depth
      min_samples_split: 2
      min_samples_leaf: 1
      class_weight: "balanced_subsample"
      random_state: 42
```

## Dependencies

- **os**: Used for environment variables
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **yaml**: Used for loading YAML configuration
- **sklearn.compose**: Used for ColumnTransformer
- **sklearn.ensemble**: Used for RandomForestClassifier
- **sklearn.feature_extraction.text**: Used for TfidfVectorizer
- **sklearn.multioutput**: Used for MultiOutputClassifier
- **sklearn.pipeline**: Used for Pipeline
- **sklearn.preprocessing**: Used for StandardScaler
- **sklearn.model_selection**: Used for GridSearchCV (in optimize_hyperparameters)
- **nexusml.core.feature_engineering**: Used for GenericFeatureEngineer

## Notes and Warnings

- The model expects specific columns after feature engineering:
  - "combined_features": Text features combined from multiple columns
  - "service_life": Numeric feature representing service life
- The feature engineering step in the pipeline is optional and is mainly used when the model is called directly
- In most cases, feature engineering should be applied separately before using the model
- The optimize_hyperparameters function can be computationally expensive, especially with large datasets
- The model is designed to handle imbalanced classes through the "balanced_subsample" class weight
- The scoring metric for hyperparameter optimization is f1_macro, which is better for imbalanced classes than accuracy
- The model uses a MultiOutputClassifier to handle multiple target columns simultaneously
