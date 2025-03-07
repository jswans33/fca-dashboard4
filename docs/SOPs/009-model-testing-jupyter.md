# SOP 009: Test-Driving the Equipment Classification Model in Jupyter Notebook

## 1. Purpose

This SOP provides a concise guide for testing the equipment classification model
using a Jupyter notebook.

## 2. Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab installed
- NexusML package installed
- Access to a trained model file (located in `outputs/models/` directory)

## 3. Setup Jupyter Environment

```bash
# Install Jupyter if needed
pip install jupyter

# Start Jupyter Notebook
jupyter notebook
```

## 4. Basic Model Testing Notebook

Create a new notebook and add the following cells:

### 4.1 Import Dependencies

```python
# Import required libraries
import pandas as pd
import numpy as np
from nexusml.core.model import EquipmentClassifier
import matplotlib.pyplot as plt
```

### 4.2 Load the Model

```python
# Initialize the classifier
classifier = EquipmentClassifier()

# Load a trained model
# The model files are stored in the outputs/models directory with timestamps in their names
# For example: equipment_classifier_20250306_161707.pkl
model_path = "outputs/models/equipment_classifier_20250306_161707.pkl"  # Use the actual model file name
classifier.load_model(model_path)
print(f"Model loaded from {model_path}")

# To find available model files, you can run:
# import os
# [f for f in os.listdir("outputs/models") if f.endswith(".pkl")]
```

### 4.3 Test with Sample Descriptions

```python
# Define test descriptions
test_descriptions = [
    "Trane XR-14 Rooftop Unit for HVAC system",
    "Grundfos CRE5-10 Pump for water circulation",
    "Caterpillar C32 Generator for emergency power",
    "York YK8000 Chiller for cooling system",
    "Daikin Vision Air Handler Unit for ventilation",
    "Tyco TY325 Fire Sprinkler for fire protection",
    "Greenheck GB-420 Exhaust Fan for ventilation",
    "Siemens 3AX78 Transformer for electrical distribution"
]

# Make predictions
results = []
for description in test_descriptions:
    prediction = classifier.predict(description)

    # Extract key information
    result = {
        "Description": description,
        "Equipment Category": prediction.get("category_name", "Unknown"),
        "System Type": prediction.get("mcaa_system_category", "Unknown"),
        "Equipment Type": prediction.get("Equipment_Type", "Unknown"),
        "MasterFormat": prediction.get("MasterFormat_Class", "Unknown")
    }
    results.append(result)

# Display results as a DataFrame
results_df = pd.DataFrame(results)
results_df
```

### 4.4 Visualize Results

```python
# Create a bar chart of equipment categories
plt.figure(figsize=(10, 6))
results_df["Equipment Category"].value_counts().plot(kind="barh")
plt.title("Predicted Equipment Categories")
plt.xlabel("Count")
plt.ylabel("Equipment Category")
plt.tight_layout()
plt.show()
```

## 5. Interactive Testing

Add an interactive cell for testing custom descriptions:

```python
# Function for interactive testing
def test_equipment_description(description, service_life=20.0):
    prediction = classifier.predict(description, service_life)

    print(f"Description: {description}")
    print(f"Service Life: {service_life} years\n")

    print(f"Equipment Category: {prediction.get('category_name', 'Unknown')}")
    print(f"System Type: {prediction.get('mcaa_system_category', 'Unknown')}")
    print(f"Equipment Type: {prediction.get('Equipment_Type', 'Unknown')}")
    print(f"MasterFormat Class: {prediction.get('MasterFormat_Class', 'Unknown')}")

    # Print attribute template if available
    if "attribute_template" in prediction and prediction["attribute_template"] != "Unknown":
        if isinstance(prediction["attribute_template"], dict) and "error" not in prediction["attribute_template"]:
            print("\nRequired Attributes:")
            for attr, info in prediction["attribute_template"].get("required_attributes", {}).items():
                print(f"  - {attr}: {info}")

    return prediction

# Test with your own description
test_equipment_description("Carrier 30XA air-cooled chiller with 500 ton capacity")
```

## 6. Batch Testing with CSV Data

Add a cell for testing with a CSV file:

```python
# Load test data from CSV
test_data_path = "path/to/your/test_data.csv"  # Adjust path as needed
test_data = pd.read_csv(test_data_path)

# Check if we have production format columns
has_production_format = all(col in test_data.columns for col in ["equipment_tag", "manufacturer", "model"])

# Process each item
batch_results = []
for _, row in test_data.iterrows():
    # For production format data
    if has_production_format:
        description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"
        service_life = float(row.get("condition_score", 20.0))
    # For legacy format data
    else:
        description = row["Description"]
        service_life = row.get("Service Life", 20.0)

    prediction = classifier.predict(description, service_life)

    # Extract key information
    result = {
        "Description": description,
        "Equipment Category": prediction.get("category_name", "Unknown"),
        "System Type": prediction.get("mcaa_system_category", "Unknown"),
        "Equipment Type": prediction.get("Equipment_Type", "Unknown"),
        "MasterFormat": prediction.get("MasterFormat_Class", "Unknown")
    }
    batch_results.append(result)

# Display results as a DataFrame
batch_results_df = pd.DataFrame(batch_results)
batch_results_df
```

## 7. Comparing Model Predictions with Actual Values

For testing the model's accuracy against known values:

```python
# Assuming test_data has actual category values
if "actual_category" in test_data.columns:
    comparison_df = pd.DataFrame({
        "Description": batch_results_df["Description"],
        "Actual Category": test_data["actual_category"],
        "Predicted Category": batch_results_df["Equipment Category"],
        "Match": test_data["actual_category"] == batch_results_df["Equipment Category"]
    })

    # Calculate accuracy
    accuracy = comparison_df["Match"].mean() * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")

    # Display comparison
    comparison_df
```

## 8. Saving Results

```python
# Save results to CSV
batch_results_df.to_csv("model_test_results.csv", index=False)
print("Results saved to model_test_results.csv")
```

## 9. Using the Random Guessing Script

For quick testing without setting up a Jupyter notebook, you can use the
provided random guessing script:

```bash
# Basic usage - generates 5 random descriptions
python -m nexusml.examples.random_guessing

# Generate 10 random descriptions
python -m nexusml.examples.random_guessing --num-samples 10

# Test with your own custom description
python -m nexusml.examples.random_guessing --custom "Carrier 30XA air-cooled chiller with 500 ton capacity for data center cooling"

# Use a specific model file
python -m nexusml.examples.random_guessing --model-path outputs/models/equipment_classifier_20250306_161707.pkl
```

The script will:

1. Load the specified model
2. Generate random equipment descriptions
3. Run each description through the model
4. Display the prediction results

This is useful for quick testing or when you want to see how the model performs
on a variety of equipment types without setting up a full Jupyter environment.

## 10. Troubleshooting

If you encounter issues:

- Verify the model path is correct
- Check that input data format matches what the model expects
- For memory issues, process data in smaller batches
- If getting warnings about missing columns, refer to SOP 008, section 7.5

## 11. Extracting Specific Classification Fields

If you only need to extract specific fields like equipment category or omniclass
codes from the model predictions, you can modify your code to focus on just
these fields:

### 11.1 Extracting Equipment Category Only

```python
# Function to extract just the equipment category
def get_equipment_category(description, service_life=20.0):
    classifier = EquipmentClassifier()
    classifier.load_model("outputs/models/equipment_classifier_20250306_161707.pkl")

    prediction = classifier.predict(description, service_life)

    # Extract just the equipment category
    category = prediction.get("category_name", "Unknown")

    return category

# Test with a description
description = "Trane XR-14 Rooftop Unit for HVAC system"
category = get_equipment_category(description)
print(f"Description: {description}")
print(f"Equipment Category: {category}")
```

### 11.2 Extracting OmniClass Information

```python
# Function to extract just the OmniClass information
def get_omniclass_info(description, service_life=20.0):
    classifier = EquipmentClassifier()
    classifier.load_model("outputs/models/equipment_classifier_20250306_161707.pkl")

    prediction = classifier.predict(description, service_life)

    # Extract OmniClass information
    omniclass_id = prediction.get("OmniClass_ID", "Unknown")

    # If you need more detailed OmniClass information from the attribute template
    attribute_template = prediction.get("attribute_template", {})
    if isinstance(attribute_template, dict) and "classification" in attribute_template:
        omniclass_details = attribute_template["classification"].get("OmniClass", {})
        return {
            "OmniClass_ID": omniclass_id,
            "OmniClass_Details": omniclass_details
        }

    return {"OmniClass_ID": omniclass_id}

# Test with a description
description = "Trane XR-14 Rooftop Unit for HVAC system"
omniclass_info = get_omniclass_info(description)
print(f"Description: {description}")
print(f"OmniClass Information: {omniclass_info}")
```

### 11.3 Batch Processing for Specific Fields

For processing multiple items and extracting only specific fields:

```python
import pandas as pd

# Load data
data = pd.read_csv("path/to/your/data.csv")

# Initialize classifier once
classifier = EquipmentClassifier()
classifier.load_model("outputs/models/equipment_classifier_20250306_161707.pkl")

# Extract specific fields for each item
results = []
for _, row in data.iterrows():
    description = row["Description"]  # Adjust column name as needed

    prediction = classifier.predict(description)

    # Extract only the fields you need
    result = {
        "Description": description,
        "Equipment_Category": prediction.get("category_name", "Unknown"),
        "OmniClass_ID": prediction.get("OmniClass_ID", "Unknown"),
        "MasterFormat_Class": prediction.get("MasterFormat_Class", "Unknown")
    }

    results.append(result)

# Convert to DataFrame and save
results_df = pd.DataFrame(results)
results_df.to_csv("extracted_classifications.csv", index=False)
```
