# Example: advanced_example.py

## Overview

The `advanced_example.py` script demonstrates a more sophisticated usage of the NexusML package, including visualization components and proper type annotations. It extends the simple example with better code organization, error handling, and visualization capabilities. This example is ideal for users who want to understand how to integrate NexusML into larger, production-quality applications.

Key features demonstrated:

1. **Type Annotations**: Proper Python type hints for better code quality and IDE support
2. **Modular Design**: Well-structured code with separate functions for different tasks
3. **Visualization**: Generation of data distribution visualizations
4. **Error Handling**: Robust error handling for file operations and settings loading
5. **Constants**: Use of constants for default values
6. **Settings Management**: Advanced settings management with fallbacks and merging

## Usage

```bash
python -m nexusml.examples.advanced_example
```

This will:
1. Load settings from a configuration file (or use defaults)
2. Train an enhanced model using the specified training data
3. Make a prediction for a sample equipment description
4. Save the prediction results to a file
5. Generate and save visualizations of the data distribution

## Code Walkthrough

### Type Annotations

The example demonstrates proper type annotations for better code quality and IDE support:

```python
# Type aliases for better readability
ModelType = Any  # Replace with actual model type when known
PredictionDict = Dict[str, str]  # Dictionary with string keys and values
DataFrameType = Any  # Replace with actual DataFrame type when known

# Add type annotation for the imported function
def predict_with_enhanced_model(model: ModelType, description: str, service_life: float = 0) -> PredictionDict:
    """
    Wrapper with type annotation for the imported predict_with_enhanced_model function

    This wrapper ensures proper type annotations for the function.

    Args:
        model: The trained model
        description: Equipment description
        service_life: Service life in years

    Returns:
        PredictionDict: Dictionary with prediction results
    """
    # Call the original function and convert the result to the expected type
    result = _predict_with_enhanced_model(model, description, service_life)  # type: ignore
    # We know the result is a dictionary with string keys and values
    return {str(k): str(v) for k, v in result.items()}  # type: ignore
```

This approach:
1. Defines type aliases for better readability
2. Creates a wrapper function with proper type annotations
3. Ensures type safety by converting the result to the expected type
4. Uses type comments to suppress type checker warnings when necessary

### Modular Design

The example demonstrates a modular design with separate functions for different tasks:

```python
def get_default_settings() -> Dict[str, Any]:
    """Return default settings when configuration file is not found"""
    # ...

def load_settings() -> Dict[str, Any]:
    """Load settings from the configuration file"""
    # ...

def get_merged_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Merge settings from different sections for compatibility"""
    # ...

def get_paths_from_settings(merged_settings: Dict[str, Any]) -> Tuple[str, str, str, str, str]:
    """Extract paths from settings"""
    # ...

def make_prediction(model: ModelType, description: str, service_life: float) -> PredictionDict:
    """Make a prediction using the trained model"""
    # ...

def save_prediction_results(...) -> None:
    """Save prediction results to a file"""
    # ...

def generate_visualizations(df: DataFrameType, output_dir: str) -> Tuple[str, str]:
    """Generate visualizations for the data"""
    # ...
```

This modular approach:
1. Makes the code more maintainable and testable
2. Separates concerns into distinct functions
3. Provides clear documentation for each function
4. Makes the main function simpler and more readable

### Constants

The example uses constants for default values:

```python
# Constants
DEFAULT_TRAINING_DATA_PATH = "ingest/data/eq_ids.csv"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_PREDICTION_FILENAME = "example_prediction.txt"
TARGET_CLASSES = ["Equipment_Category", "Uniformat_Class", "System_Type", "Equipment_Type", "System_Subtype"]
```

This approach:
1. Makes the code more maintainable by centralizing default values
2. Makes it easier to change default values in the future
3. Improves code readability by using descriptive names

### Settings Management

The example demonstrates advanced settings management:

```python
def load_settings() -> Dict[str, Any]:
    """
    Load settings from the configuration file

    Returns:
        Dict[str, Any]: Configuration settings
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
            return get_default_settings()

    try:
        with open(settings_path, "r") as file:
            return yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError) as e:
        print(f"Error loading settings file at {settings_path}: {e}")
        # Return default settings
        return get_default_settings()
```

This function:
1. Tries to find a settings file in the project's config directory
2. If not found, checks if it's running in the fca_dashboard context
3. If neither is available, uses default settings
4. Handles potential errors when loading the settings file

### Visualization Generation

The example demonstrates how to generate visualizations:

```python
def generate_visualizations(df: DataFrameType, output_dir: str) -> Tuple[str, str]:
    """
    Generate visualizations for the data

    Args:
        df: DataFrame with the data
        output_dir: Directory to save visualizations

    Returns:
        Tuple[str, str]: Paths to the saved visualization files
    """
    print("\nGenerating visualizations...")

    # Use the visualize_category_distribution function from the model module
    equipment_category_file, system_type_file = visualize_category_distribution(df, output_dir)

    print(f"Visualizations saved to:")
    print(f"  - {equipment_category_file}")
    print(f"  - {system_type_file}")

    return equipment_category_file, system_type_file
```

This function:
1. Uses the visualize_category_distribution function from the model module
2. Saves the visualizations to the specified output directory
3. Returns the paths to the saved visualization files
4. Prints information about the saved visualizations

### Main Function

The main function ties everything together:

```python
def main() -> None:
    """
    Main function demonstrating the usage of the NexusML package
    """
    # Load and process settings
    settings = load_settings()
    merged_settings = get_merged_settings(settings)
    data_path, output_dir, equipment_category_file, system_type_file, prediction_file = get_paths_from_settings(
        merged_settings
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Train enhanced model using the CSV file
    print(f"Training the model using data from: {data_path}")
    model, df = train_enhanced_model(data_path)

    # Example prediction with service life
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = 20.0  # Example service life in years

    # Make prediction
    prediction = make_prediction(model, description, service_life)

    # Save prediction results
    save_prediction_results(
        prediction_file, prediction, description, service_life, equipment_category_file, system_type_file
    )

    # Generate visualizations
    equipment_category_file, system_type_file = generate_visualizations(df, output_dir)
```

This function:
1. Loads and processes settings
2. Creates the output directory if it doesn't exist
3. Trains the model
4. Makes a prediction
5. Saves the prediction results
6. Generates visualizations

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

Generating visualizations...
Visualizations saved to:
  - /path/to/nexusml/examples/outputs/equipment_category_distribution.png
  - /path/to/nexusml/examples/outputs/system_type_distribution.png
```

### File Output (example_prediction.txt)

Similar to the simple example, but with added visualization paths:

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

Visualizations saved to:
  - /path/to/nexusml/examples/outputs/equipment_category_distribution.png
  - /path/to/nexusml/examples/outputs/system_type_distribution.png
```

### Visualization Output

The example generates two visualization files:

1. **equipment_category_distribution.png**: A bar chart showing the distribution of equipment categories in the training data
2. **system_type_distribution.png**: A bar chart showing the distribution of system types in the training data

These visualizations help users understand the distribution of the training data and identify potential biases or imbalances.

## Key Concepts Demonstrated

### 1. Type Safety

The example demonstrates how to ensure type safety in Python code:

```python
# Type aliases for better readability
ModelType = Any  # Replace with actual model type when known
PredictionDict = Dict[str, str]  # Dictionary with string keys and values
DataFrameType = Any  # Replace with actual DataFrame type when known

# Add type annotation for the imported function
def predict_with_enhanced_model(model: ModelType, description: str, service_life: float = 0) -> PredictionDict:
    # ...
```

This approach:
1. Makes the code more maintainable and less error-prone
2. Provides better IDE support for code completion and error checking
3. Documents the expected types for function parameters and return values

### 2. Function Composition

The example demonstrates how to compose functions to create a clean and maintainable codebase:

```python
def main() -> None:
    # Load and process settings
    settings = load_settings()
    merged_settings = get_merged_settings(settings)
    data_path, output_dir, equipment_category_file, system_type_file, prediction_file = get_paths_from_settings(
        merged_settings
    )
    
    # ...
    
    # Make prediction
    prediction = make_prediction(model, description, service_life)
    
    # Save prediction results
    save_prediction_results(
        prediction_file, prediction, description, service_life, equipment_category_file, system_type_file
    )
    
    # Generate visualizations
    equipment_category_file, system_type_file = generate_visualizations(df, output_dir)
```

This approach:
1. Makes the code more readable and maintainable
2. Separates concerns into distinct functions
3. Makes it easier to test individual components
4. Allows for better error handling at each step

### 3. Error Handling

The example demonstrates robust error handling:

```python
try:
    with open(settings_path, "r") as file:
        return yaml.safe_load(file)
except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error loading settings file at {settings_path}: {e}")
    # Return default settings
    return get_default_settings()
```

This approach:
1. Catches specific exceptions rather than generic ones
2. Provides informative error messages
3. Gracefully falls back to default values when errors occur
4. Ensures the application continues to run even when errors occur

### 4. Visualization Integration

The example demonstrates how to integrate visualizations into the workflow:

```python
def generate_visualizations(df: DataFrameType, output_dir: str) -> Tuple[str, str]:
    # ...
    equipment_category_file, system_type_file = visualize_category_distribution(df, output_dir)
    # ...
    return equipment_category_file, system_type_file
```

This approach:
1. Separates visualization generation into its own function
2. Returns the paths to the generated visualization files
3. Integrates with the rest of the workflow
4. Provides useful visual information about the data

## Dependencies

- **os**: Standard library module for file operations
- **pathlib**: Standard library module for path manipulation
- **typing**: Standard library module for type hints
- **yaml**: Used for loading YAML configuration files
- **nexusml.core.model**: Core module containing the model training, prediction, and visualization functions

## Notes and Warnings

- The example assumes that the training data file exists at the specified path
- The model performance metrics in the output file are placeholders and not actual metrics
- The example uses type aliases (ModelType, PredictionDict, DataFrameType) that should be replaced with actual types in a real application
- The example is designed to work both standalone and in the context of fca_dashboard
- The default paths assume a specific project structure, which may need to be adjusted for your environment
- The example includes type ignore comments to suppress type checker warnings in certain cases

## Comparison with simple_example.py

The advanced example builds upon the simple example with several improvements:

1. **Type Annotations**: Adds proper type hints for better code quality and IDE support
2. **Modular Design**: Refactors the code into separate functions for different tasks
3. **Error Handling**: Adds more robust error handling for file operations and settings loading
4. **Constants**: Uses constants for default values
5. **Visualization**: Adds visualization generation
6. **Function Composition**: Uses function composition for a cleaner main function

## Extensions and Variations

### Adding Custom Visualizations

To add custom visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

def generate_custom_visualizations(df: DataFrameType, output_dir: str) -> str:
    """Generate custom visualizations for the data"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a custom visualization
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="service_life", bins=10)
    plt.title("Service Life Distribution")
    plt.xlabel("Service Life (years)")
    plt.ylabel("Count")
    
    # Save the visualization
    output_file = os.path.join(output_dir, "service_life_distribution.png")
    plt.savefig(output_file)
    plt.close()
    
    return output_file
```

### Adding Command-Line Arguments

To add command-line arguments:

```python
import argparse

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Advanced example of NexusML usage")
    parser.add_argument("--data-path", help="Path to the training data CSV file")
    parser.add_argument("--output-dir", help="Directory to save outputs")
    parser.add_argument("--description", help="Equipment description for prediction")
    parser.add_argument("--service-life", type=float, help="Service life in years")
    return parser.parse_args()

def main() -> None:
    """Main function demonstrating the usage of the NexusML package"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load and process settings
    settings = load_settings()
    merged_settings = get_merged_settings(settings)
    
    # Override settings with command-line arguments if provided
    data_path = args.data_path or merged_settings.get("data_paths", {}).get("training_data")
    output_dir = args.output_dir or merged_settings.get("examples", {}).get("output_dir")
    description = args.description or "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = args.service_life or 20.0
    
    # ... rest of the function ...
```

### Adding Logging

To add proper logging:

```python
import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/advanced_example.log"),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("advanced_example")

def main() -> None:
    """Main function demonstrating the usage of the NexusML package"""
    # Set up logging
    logger = setup_logging()
    
    # Load and process settings
    logger.info("Loading settings")
    settings = load_settings()
    merged_settings = get_merged_settings(settings)
    
    # ... rest of the function, using logger instead of print ...