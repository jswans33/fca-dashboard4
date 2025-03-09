# Example: simple_example.py

## Overview

The `simple_example.py` script demonstrates the core functionality of the NexusML package in a simplified manner, focusing on the essential workflow without visualization components. It provides a straightforward example of how to use NexusML for equipment classification, from loading settings to training a model and making predictions.

Key features demonstrated:

1. **Settings Management**: Loading configuration settings from YAML files
2. **Model Training**: Training an enhanced equipment classification model
3. **Prediction**: Making predictions with the trained model
4. **Output Handling**: Saving prediction results to a file
5. **Integration Options**: Compatibility with both standalone and fca_dashboard contexts

## Usage

```bash
python -m nexusml.examples.simple_example
```

This will:
1. Load settings from a configuration file (or use defaults)
2. Train an enhanced model using the specified training data
3. Make a prediction for a sample equipment description
4. Print the prediction results to the console
5. Save the prediction results to a file

## Code Walkthrough

### Settings Management

The example demonstrates how to load settings from a configuration file, with fallbacks for different execution contexts:

```python
def load_settings():
    """
    Load settings from the configuration file
    
    Returns:
        dict: Configuration settings
    """
    # Try to find a settings file
    settings_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yml"
    
    if not settings_path.exists():
        # Check if we're running in the context of fca_dashboard
        try:
            from fca_dashboard.utils.path_util import get_config_path
            settings_path = get_config_path("settings.yml")
        except ImportError:
            # Not running in fca_dashboard context, use default settings
            return {
                'nexusml': {
                    'data_paths': {
                        'training_data': str(Path(__file__).resolve().parent.parent / "ingest" / "data" / "eq_ids.csv")
                    },
                    'examples': {
                        'output_dir': str(Path(__file__).resolve().parent / "outputs")
                    }
                }
            }
```

This function:
1. Tries to find a settings file in the project's config directory
2. If not found, checks if it's running in the fca_dashboard context
3. If neither is available, uses default settings

### Model Training

The example shows how to train an enhanced model using the NexusML core functionality:

```python
# Train enhanced model using the CSV file
print(f"Training the model using data from: {data_path}")
model, df = train_enhanced_model(data_path)
```

This single line of code handles:
1. Loading the training data from the specified path
2. Preprocessing the data
3. Training the model
4. Returning both the trained model and the processed DataFrame

### Making Predictions

The example demonstrates how to make predictions with the trained model:

```python
# Example prediction with service life
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years

print("\nMaking a prediction for:")
print(f"Description: {description}")
print(f"Service Life: {service_life} years")

prediction = predict_with_enhanced_model(model, description, service_life)

print("\nEnhanced Prediction:")
for key, value in prediction.items():
    print(f"{key}: {value}")
```

This code:
1. Defines a sample equipment description and service life
2. Makes a prediction using the trained model
3. Prints the prediction results to the console

### Saving Results

The example shows how to save prediction results to a file:

```python
# Save prediction results to file
print(f"\nSaving prediction results to {prediction_file}")
with open(prediction_file, 'w') as f:
    f.write("Enhanced Prediction Results\n")
    f.write("==========================\n\n")
    f.write("Input:\n")
    f.write(f"  Description: {description}\n")
    f.write(f"  Service Life: {service_life} years\n\n")
    f.write("Prediction:\n")
    for key, value in prediction.items():
        f.write(f"  {key}: {value}\n")
    
    # Add placeholder for model performance metrics
    f.write("\nModel Performance Metrics\n")
    f.write("========================\n")
    for target in ['Equipment_Category', 'Uniformat_Class', 'System_Type', 'Equipment_Type', 'System_Subtype']:
        f.write(f"{target} Classification:\n")
        f.write(f"  Precision: {0.80 + 0.03 * (5 - list(prediction.keys()).index(target)):.2f}\n")
        f.write(f"  Recall: {0.78 + 0.03 * (5 - list(prediction.keys()).index(target)):.2f}\n")
        f.write(f"  F1 Score: {0.79 + 0.03 * (5 - list(prediction.keys()).index(target)):.2f}\n")
        f.write(f"  Accuracy: {0.82 + 0.03 * (5 - list(prediction.keys()).index(target)):.2f}\n\n")
```

This code:
1. Opens a file for writing
2. Writes the prediction results in a formatted manner
3. Adds placeholder model performance metrics for demonstration purposes

## Expected Output

### Console Output

```
Training the model using data from: /path/to/nexusml/ingest/data/eq_ids.csv
Loading data from /path/to/nexusml/ingest/data/eq_ids.csv
Data loaded successfully: 1000 rows, 15 columns
Training enhanced model...
Model trained successfully

Making a prediction for:
Description: Heat Exchanger for Chilled Water system with Plate and Frame design
Service Life: 20.0 years

Enhanced Prediction:
Equipment_Category: Heat Exchanger
Uniformat_Class: D3010
System_Type: HVAC
Equipment_Type: Heat Exchanger-Plate and Frame
System_Subtype: Cooling
OmniClass_ID: 23-33 11 11
Uniformat_ID: D3010
MasterFormat_Class: 23 57 00
attribute_template: {'required_attributes': {'heat_exchanger_type': 'Plate and Frame', 'capacity_btu': 'Unknown', 'fluid_type': 'Water'}}
master_db_mapping: {'asset_category': 'Heat Exchanger', 'asset_type': 'Plate and Frame', 'system_type': 'HVAC'}

Saving prediction results to /path/to/nexusml/examples/outputs/example_prediction.txt
```

### File Output (example_prediction.txt)

```
Enhanced Prediction Results
==========================

Input:
  Description: Heat Exchanger for Chilled Water system with Plate and Frame design
  Service Life: 20.0 years

Prediction:
  Equipment_Category: Heat Exchanger
  Uniformat_Class: D3010
  System_Type: HVAC
  Equipment_Type: Heat Exchanger-Plate and Frame
  System_Subtype: Cooling
  OmniClass_ID: 23-33 11 11
  Uniformat_ID: D3010
  MasterFormat_Class: 23 57 00
  attribute_template: {'required_attributes': {'heat_exchanger_type': 'Plate and Frame', 'capacity_btu': 'Unknown', 'fluid_type': 'Water'}}
  master_db_mapping: {'asset_category': 'Heat Exchanger', 'asset_type': 'Plate and Frame', 'system_type': 'HVAC'}

Model Performance Metrics
========================
Equipment_Category Classification:
  Precision: 0.95
  Recall: 0.93
  F1 Score: 0.94
  Accuracy: 0.97

Uniformat_Class Classification:
  Precision: 0.92
  Recall: 0.90
  F1 Score: 0.91
  Accuracy: 0.94

System_Type Classification:
  Precision: 0.89
  Recall: 0.87
  F1 Score: 0.88
  Accuracy: 0.91

Equipment_Type Classification:
  Precision: 0.86
  Recall: 0.84
  F1 Score: 0.85
  Accuracy: 0.88

System_Subtype Classification:
  Precision: 0.83
  Recall: 0.81
  F1 Score: 0.82
  Accuracy: 0.85
```

## Key Concepts Demonstrated

### 1. Simplified API

The example demonstrates the simplified API of NexusML, which allows for training a model and making predictions with just a few lines of code:

```python
model, df = train_enhanced_model(data_path)
prediction = predict_with_enhanced_model(model, description, service_life)
```

### 2. Configuration Management

The example shows how to handle configuration settings in different execution contexts:
- Loading from a YAML file if available
- Using fca_dashboard utilities if running in that context
- Falling back to default settings if neither is available

### 3. Path Resolution

The example demonstrates robust path resolution techniques:
- Using `Path(__file__).resolve().parent` to find relative paths
- Creating output directories if they don't exist
- Handling both absolute and relative paths

### 4. Error Handling

The example includes basic error handling:
- Catching FileNotFoundError when loading settings
- Providing default values when settings are not found
- Printing warning messages for potential issues

### 5. Output Formatting

The example shows how to format and save output:
- Printing structured information to the console
- Writing formatted results to a file
- Including both input parameters and prediction results

## Dependencies

- **os**: Standard library module for file operations
- **pathlib**: Standard library module for path manipulation
- **yaml**: Used for loading YAML configuration files
- **nexusml.core.model**: Core module containing the model training and prediction functions

## Notes and Warnings

- The example assumes that the training data file exists at the specified path
- The model performance metrics in the output file are placeholders and not actual metrics
- The example does not include visualization components, which are available in other examples
- The example does not handle all possible error cases, such as invalid data formats
- The example is designed to work both standalone and in the context of fca_dashboard
- The default paths assume a specific project structure, which may need to be adjusted for your environment

## Extensions and Variations

### Adding Visualization

To extend this example with visualization components:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# After making the prediction
plt.figure(figsize=(10, 6))
sns.barplot(x=list(prediction.keys())[:5], y=list(prediction.values())[:5])
plt.title("Prediction Results")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "prediction_visualization.png"))
plt.show()
```

### Using Custom Feature Configuration

To use a custom feature configuration:

```python
# Train enhanced model with custom feature configuration
feature_config_path = "path/to/feature_config.yml"
model, df = train_enhanced_model(data_path, feature_config_path=feature_config_path)
```

### Batch Predictions

To make predictions for multiple descriptions:

```python
descriptions = [
    "Heat Exchanger for Chilled Water system with Plate and Frame design",
    "500 ton water-cooled centrifugal chiller",
    "10000 CFM air handler with MERV 13 filters"
]
service_lives = [20.0, 25.0, 15.0]

for desc, life in zip(descriptions, service_lives):
    prediction = predict_with_enhanced_model(model, desc, life)
    print(f"\nPrediction for: {desc} (Service Life: {life} years)")
    for key, value in prediction.items():
        print(f"{key}: {value}")
```

### Saving the Model

To save the trained model for later use:

```python
import pickle

# After training the model
model_path = os.path.join(output_dir, "trained_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# Later, to load the model
with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)
prediction = predict_with_enhanced_model(loaded_model, description, service_life)