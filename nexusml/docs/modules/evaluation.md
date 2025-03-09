# Module: evaluation

## Overview

The `evaluation` module handles model evaluation and analysis of "Other" categories in classification tasks. This module follows the Single Responsibility Principle by focusing solely on model evaluation functionality. It provides tools for:

1. Evaluating model performance with detailed metrics
2. Analyzing the "Other" category performance specifically
3. Identifying important features for "Other" classification
4. Analyzing misclassifications related to the "Other" category

The module is particularly useful for understanding how well a model handles the "Other" category, which is often challenging in classification tasks.

## Functions

### `enhanced_evaluation(model: Pipeline, X_test: Union[pd.Series, pd.DataFrame], y_test: pd.DataFrame) -> pd.DataFrame`

Evaluates the model with a focus on "Other" categories performance.

**Parameters:**

- `model` (Pipeline): Trained model pipeline
- `X_test` (Union[pd.Series, pd.DataFrame]): Test features
- `y_test` (pd.DataFrame): Test targets

**Returns:**

- pd.DataFrame: Predictions dataframe

**Example:**
```python
from nexusml.core.evaluation import enhanced_evaluation
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming you have a trained model and test data
model = Pipeline([...])  # Your trained model pipeline
X_test = pd.DataFrame(...)  # Your test features
y_test = pd.DataFrame(...)  # Your test targets

# Evaluate the model
predictions = enhanced_evaluation(model, X_test, y_test)
```

**Notes:**

- This function has been updated to handle both Series and DataFrame inputs for X_test
- It supports the pipeline structure that uses both text and numeric features
- For each target column, it prints:
  - Classification report
  - Overall accuracy
  - "Other" category specific metrics (if present):
    - Accuracy
    - Precision
    - Recall
    - F1 Score

### `analyze_other_category_features(model: Pipeline, X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None`

Analyzes what features are most important for classifying items as "Other".

**Parameters:**

- `model` (Pipeline): Trained model pipeline
- `X_test` (pd.Series): Test features
- `y_test` (pd.DataFrame): Test targets
- `y_pred_df` (pd.DataFrame): Predictions dataframe

**Example:**
```python
from nexusml.core.evaluation import enhanced_evaluation, analyze_other_category_features
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming you have a trained model and test data
model = Pipeline([...])  # Your trained model pipeline
X_test = pd.Series(...)  # Your test features as a Series
y_test = pd.DataFrame(...)  # Your test targets

# Get predictions
predictions = enhanced_evaluation(model, X_test, y_test)

# Analyze features important for "Other" classification
analyze_other_category_features(model, X_test, y_test, predictions)
```

**Notes:**

- This function has been updated to work with the pipeline structure that uses a ColumnTransformer
- It extracts the Random Forest model from the pipeline
- It identifies and displays:
  - Top text features for "Other" classification
  - Feature importance from the Random Forest model
  - Importance and rank of the service_life feature (if present)
- The function expects X_test to be a pandas Series containing text data

### `analyze_other_misclassifications(X_test: pd.Series, y_test: pd.DataFrame, y_pred_df: pd.DataFrame) -> None`

Analyzes cases where "Other" was incorrectly predicted or missed.

**Parameters:**

- `X_test` (pd.Series): Test features
- `y_test` (pd.DataFrame): Test targets
- `y_pred_df` (pd.DataFrame): Predictions dataframe

**Example:**
```python
from nexusml.core.evaluation import enhanced_evaluation, analyze_other_misclassifications
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming you have a trained model and test data
model = Pipeline([...])  # Your trained model pipeline
X_test = pd.Series(...)  # Your test features
y_test = pd.DataFrame(...)  # Your test targets

# Get predictions
predictions = enhanced_evaluation(model, X_test, y_test)

# Analyze misclassifications related to "Other" category
analyze_other_misclassifications(X_test, y_test, predictions)
```

**Notes:**

- For each target column, it analyzes:
  - False Positives: Cases predicted as "Other" but actually something else
  - False Negatives: Cases that were "Other" but predicted as something else
- For each type of misclassification, it shows:
  - The total number of cases
  - Examples of the misclassified items (up to 5)
  - The text content (first 100 characters)
  - The actual or predicted class

## Usage Examples

### Basic Model Evaluation

```python
from nexusml.core.evaluation import enhanced_evaluation
from sklearn.pipeline import Pipeline
import pandas as pd

# Assuming you have a trained model and test data
model = Pipeline([...])  # Your trained model pipeline
X_test = pd.DataFrame({
    'combined_features': ["Description of equipment 1", "Description of equipment 2"],
    'service_life': [15, 20]
})
y_test = pd.DataFrame({
    'Equipment_Category': ["Chiller", "Other"],
    'Equipment_Type': ["Centrifugal", "Other"]
})

# Evaluate the model
predictions = enhanced_evaluation(model, X_test, y_test)
print(predictions)
```

### Complete Evaluation with "Other" Category Analysis

```python
from nexusml.core.evaluation import enhanced_evaluation, analyze_other_category_features, analyze_other_misclassifications
from sklearn.pipeline import Pipeline
import pandas as pd

# Load your trained model and test data
model = Pipeline([...])  # Your trained model pipeline
X_test = pd.Series([
    "Description of equipment 1",
    "Description of equipment 2",
    "Description of equipment 3",
    "Description of equipment 4",
    "Description of equipment 5"
])
y_test = pd.DataFrame({
    'Equipment_Category': ["Chiller", "Other", "Pump", "Other", "Air Handler"],
    'Equipment_Type': ["Centrifugal", "Other", "Circulation", "Other", "VAV"]
})

# Step 1: Get predictions and basic evaluation
print("Step 1: Basic Evaluation")
predictions = enhanced_evaluation(model, X_test, y_test)

# Step 2: Analyze features important for "Other" classification
print("\nStep 2: Feature Analysis for 'Other' Category")
analyze_other_category_features(model, X_test, y_test, predictions)

# Step 3: Analyze misclassifications
print("\nStep 3: Misclassification Analysis for 'Other' Category")
analyze_other_misclassifications(X_test, y_test, predictions)

# Step 4: Summarize results
print("\nSummary:")
for col in y_test.columns:
    other_count = (y_test[col] == "Other").sum()
    other_pred_count = (predictions[col] == "Other").sum()
    print(f"{col}:")
    print(f"  Actual 'Other' count: {other_count}")
    print(f"  Predicted 'Other' count: {other_pred_count}")
```

## Dependencies

- **typing**: Used for type hints
- **numpy**: Used for numerical operations
- **pandas**: Used for DataFrame operations
- **sklearn.metrics**: Used for evaluation metrics (accuracy_score, classification_report, f1_score)
- **sklearn.pipeline**: Used for Pipeline class

## Notes and Warnings

- The `analyze_other_category_features` function expects a specific pipeline structure with:
  - A 'preprocessor' step containing a ColumnTransformer
  - A 'clf' step containing a multi-output classifier with Random Forest estimators
- The function assumes the first transformer in the ColumnTransformer is for text features and includes a 'tfidf' step
- The function expects 'service_life' as a numeric feature
- If the pipeline structure changes, the function may need to be updated
- The `analyze_other_category_features` function requires X_test to be a pandas Series, not a DataFrame
