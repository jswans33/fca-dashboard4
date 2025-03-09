# Command-Line Tool: train_model_pipeline.py

## Overview

The `train_model_pipeline.py` script implements a production-ready pipeline for training the equipment classification model following SOP 008. It provides a structured workflow with command-line arguments for flexibility, proper logging, comprehensive evaluation, and model versioning.

Key features include:

1. **Data Validation**: Validates training data to ensure it meets quality standards
2. **Feature Engineering**: Applies feature engineering to prepare data for model training
3. **Model Training**: Trains the equipment classification model with configurable parameters
4. **Hyperparameter Optimization**: Optionally performs hyperparameter optimization
5. **Model Evaluation**: Provides comprehensive evaluation metrics and analysis
6. **Model Versioning**: Saves models with timestamps and metadata for versioning
7. **Visualization Generation**: Creates visualizations of model performance and data distribution
8. **Sample Prediction**: Makes a sample prediction to verify model functionality

## Usage

```bash
python train_model_pipeline.py --data-path DATA_PATH [options]
```

### Arguments

#### Data Arguments

- `--data-path`: Path to the training data CSV file
- `--feature-config`: Path to the feature configuration YAML file
- `--reference-config`: Path to the reference configuration YAML file

#### Training Arguments

- `--test-size`: Proportion of data to use for testing (default: 0.3)
- `--random-state`: Random state for reproducibility (default: 42)
- `--sampling-strategy`: Sampling strategy for handling class imbalance (default: "direct", choices: ["direct"])

#### Optimization Arguments

- `--optimize`: Perform hyperparameter optimization (flag)

#### Output Arguments

- `--output-dir`: Directory to save the trained model and results (default: "outputs/models")
- `--model-name`: Base name for the saved model (default: "equipment_classifier")

#### Logging Arguments

- `--log-level`: Logging level (default: "INFO", choices: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

#### Visualization Arguments

- `--visualize`: Generate visualizations of model performance (flag)

### Examples

#### Basic Usage

```bash
python train_model_pipeline.py --data-path files/training-data/equipment_data.csv
```

This will train a model using the specified data file and save it to the default output directory.

#### With Hyperparameter Optimization and Visualizations

```bash
python train_model_pipeline.py --data-path files/training-data/equipment_data.csv --optimize --visualize
```

This will train a model with hyperparameter optimization and generate visualizations of model performance.

#### Custom Configuration and Output

```bash
python train_model_pipeline.py --data-path files/training-data/equipment_data.csv --feature-config config/custom_features.yml --reference-config config/custom_reference.yml --output-dir custom_models --model-name custom_classifier
```

This will train a model using custom feature and reference configurations and save it to a custom output directory with a custom name.

#### Detailed Logging

```bash
python train_model_pipeline.py --data-path files/training-data/equipment_data.csv --log-level DEBUG
```

This will train a model with detailed logging information.

## Functions

### `validate_training_data(data_path: str) -> Dict`

Validate the training data to ensure it meets quality standards.

**Parameters:**

- `data_path` (str): Path to the training data file

**Returns:**

- Dict: Dictionary with validation results

**Example:**
```python
from nexusml.train_model_pipeline import validate_training_data

# Validate training data
validation_results = validate_training_data("files/training-data/equipment_data.csv")
print(f"Valid: {validation_results.get('valid', False)}")
if not validation_results.get('valid', False):
    print("Issues:")
    for issue in validation_results.get('issues', []):
        print(f"  - {issue}")
```

**Notes:**

- This function checks:
  1. If the file exists and can be read
  2. If required columns are present
  3. If data types are correct
  4. If there are any missing values in critical columns

### `visualize_category_distribution(df: pd.DataFrame, output_dir: str = "outputs") -> Tuple[str, str]`

Visualize the distribution of categories in the dataset.

**Parameters:**

- `df` (pd.DataFrame): DataFrame with category columns
- `output_dir` (str, optional): Directory to save visualizations. Default is "outputs".

**Returns:**

- Tuple[str, str]: Tuple of paths to the saved visualization files

**Example:**
```python
from nexusml.train_model_pipeline import visualize_category_distribution
import pandas as pd

# Load data
df = pd.read_csv("files/training-data/equipment_data.csv")

# Visualize category distribution
equipment_category_file, system_type_file = visualize_category_distribution(df, "visualizations")
print(f"Equipment category distribution saved to: {equipment_category_file}")
print(f"System type distribution saved to: {system_type_file}")
```

**Notes:**

- This function creates two visualizations:
  1. Equipment Category Distribution
  2. System Type Distribution
- It saves the visualizations as PNG files in the specified output directory

### `visualize_confusion_matrix(y_true, y_pred, class_name: str, output_file: str) -> None`

Create and save a confusion matrix visualization.

**Parameters:**

- `y_true`: True labels
- `y_pred`: Predicted labels
- `class_name` (str): Name of the classification column
- `output_file` (str): Path to save the visualization

**Example:**
```python
from nexusml.train_model_pipeline import visualize_confusion_matrix
import pandas as pd
from sklearn.metrics import confusion_matrix

# Load true and predicted labels
y_true = pd.Series(["Chiller", "Air Handler", "Pump", "Chiller", "Air Handler"])
y_pred = pd.Series(["Chiller", "Air Handler", "Chiller", "Chiller", "Pump"])

# Create confusion matrix visualization
visualize_confusion_matrix(y_true, y_pred, "Equipment Category", "confusion_matrix.png")
```

**Notes:**

- This function creates a heatmap visualization of the confusion matrix
- It uses seaborn's heatmap function with annotations
- It saves the visualization to the specified output file

### `setup_logging(log_level: str = "INFO") -> logging.Logger`

Set up logging configuration.

**Parameters:**

- `log_level` (str, optional): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".

**Returns:**

- logging.Logger: Logger instance

**Example:**
```python
from nexusml.train_model_pipeline import setup_logging

# Set up logging with INFO level
logger = setup_logging("INFO")
logger.info("This is an info message")

# Set up logging with DEBUG level
logger = setup_logging("DEBUG")
logger.debug("This is a debug message")
```

**Notes:**

- This function creates a "logs" directory if it doesn't exist
- It creates a log file with a timestamp in the filename
- It configures logging to output to both a file and the console
- It returns a logger named "model_training"

### `parse_arguments() -> argparse.Namespace`

Parse command-line arguments.

**Returns:**

- argparse.Namespace: Parsed arguments

**Example:**
```python
from nexusml.train_model_pipeline import parse_arguments

# Parse command-line arguments
args = parse_arguments()
print(f"Data path: {args.data_path}")
print(f"Output directory: {args.output_dir}")
```

**Notes:**

- This function defines and parses command-line arguments for the script
- It provides default values for most arguments
- It returns a namespace object with the parsed arguments

### `load_reference_data(config_path: Optional[str] = None, logger: Optional[logging.Logger] = None) -> ReferenceManager`

Load reference data using the ReferenceManager.

**Parameters:**

- `config_path` (Optional[str], optional): Path to the reference configuration file. Default is None.
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- ReferenceManager: Initialized ReferenceManager with loaded data

**Example:**
```python
from nexusml.train_model_pipeline import load_reference_data, setup_logging

# Set up logging
logger = setup_logging("INFO")

# Load reference data
ref_manager = load_reference_data("config/reference_config.yml", logger)
```

**Notes:**

- This function creates a ReferenceManager instance with the specified configuration
- It loads all reference data sources
- It logs the process if a logger is provided

### `validate_data(data_path: str, logger: Optional[logging.Logger] = None) -> Dict`

Validate the training data to ensure it meets quality standards.

**Parameters:**

- `data_path` (str): Path to the training data
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- Dict: Validation results dictionary

**Example:**
```python
from nexusml.train_model_pipeline import validate_data, setup_logging

# Set up logging
logger = setup_logging("INFO")

# Validate data
validation_results = validate_data("files/training-data/equipment_data.csv", logger)
```

**Notes:**

- This function is a wrapper around validate_training_data that adds logging
- It logs the validation process and results if a logger is provided

### `train_model(data_path: Optional[str] = None, feature_config_path: Optional[str] = None, sampling_strategy: str = "direct", test_size: float = 0.3, random_state: int = 42, optimize_params: bool = False, logger: Optional[logging.Logger] = None) -> Tuple[EquipmentClassifier, pd.DataFrame, Dict]`

Train the equipment classification model.

**Parameters:**

- `data_path` (Optional[str], optional): Path to the training data. Default is None.
- `feature_config_path` (Optional[str], optional): Path to the feature configuration. Default is None.
- `sampling_strategy` (str, optional): Strategy for handling class imbalance. Default is "direct".
- `test_size` (float, optional): Proportion of data to use for testing. Default is 0.3.
- `random_state` (int, optional): Random state for reproducibility. Default is 42.
- `optimize_params` (bool, optional): Whether to perform hyperparameter optimization. Default is False.
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- Tuple[EquipmentClassifier, pd.DataFrame, Dict]: Tuple containing:
  - Trained EquipmentClassifier
  - Processed DataFrame
  - Dictionary with evaluation metrics

**Example:**
```python
from nexusml.train_model_pipeline import train_model, setup_logging

# Set up logging
logger = setup_logging("INFO")

# Train model
classifier, df, metrics = train_model(
    data_path="files/training-data/equipment_data.csv",
    feature_config_path="config/feature_config.yml",
    sampling_strategy="direct",
    test_size=0.3,
    random_state=42,
    optimize_params=True,
    logger=logger
)

# Print metrics
for col, col_metrics in metrics.items():
    print(f"{col}:")
    for metric_name, metric_value in col_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
```

**Notes:**

- This function creates an EquipmentClassifier instance and trains it with the specified parameters
- It optionally performs hyperparameter optimization
- It evaluates the model and calculates metrics
- It logs the process if a logger is provided

### `save_model(classifier: EquipmentClassifier, output_dir: str, model_name: str, metrics: Dict, logger: Optional[logging.Logger] = None) -> Dict`

Save the trained model and metadata.

**Parameters:**

- `classifier` (EquipmentClassifier): Trained EquipmentClassifier
- `output_dir` (str): Directory to save the model
- `model_name` (str): Base name for the model file
- `metrics` (Dict): Evaluation metrics
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- Dict: Dictionary with paths to saved files

**Example:**
```python
from nexusml.train_model_pipeline import save_model, setup_logging
from nexusml.core.model import EquipmentClassifier

# Set up logging
logger = setup_logging("INFO")

# Create classifier
classifier = EquipmentClassifier()
# ... train the classifier ...

# Save model
metrics = {"category_name": {"accuracy": 0.95, "f1_macro": 0.92}}
save_paths = save_model(
    classifier,
    "outputs/models",
    "equipment_classifier",
    metrics,
    logger
)

# Print paths
for key, path in save_paths.items():
    print(f"{key}: {path}")
```

**Notes:**

- This function saves the trained model and metadata to the specified directory
- It creates versioned files with timestamps
- It also creates copies of the latest model and metadata
- It logs the process if a logger is provided

### `generate_visualizations(classifier: EquipmentClassifier, df: pd.DataFrame, output_dir: str, logger: Optional[logging.Logger] = None) -> Dict`

Generate visualizations of model performance and data distribution.

**Parameters:**

- `classifier` (EquipmentClassifier): Trained EquipmentClassifier
- `df` (pd.DataFrame): Processed DataFrame
- `output_dir` (str): Directory to save visualizations
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- Dict: Dictionary with paths to visualization files

**Example:**
```python
from nexusml.train_model_pipeline import generate_visualizations, setup_logging
from nexusml.core.model import EquipmentClassifier

# Set up logging
logger = setup_logging("INFO")

# Create classifier
classifier = EquipmentClassifier()
# ... train the classifier ...

# Generate visualizations
viz_paths = generate_visualizations(
    classifier,
    df,
    "outputs/models",
    logger
)

# Print paths
for key, path in viz_paths.items():
    if isinstance(path, dict):
        print(f"{key}:")
        for subkey, subpath in path.items():
            print(f"  {subkey}: {subpath}")
    else:
        print(f"{key}: {path}")
```

**Notes:**

- This function generates visualizations of model performance and data distribution
- It creates a "visualizations" subdirectory in the specified output directory
- It generates category distribution visualizations and confusion matrices
- It logs the process if a logger is provided

### `make_sample_prediction(classifier: EquipmentClassifier, description: str = "Heat Exchanger for Chilled Water system with Plate and Frame design", service_life: float = 20.0, logger: Optional[logging.Logger] = None) -> Dict`

Make a sample prediction using the trained model.

**Parameters:**

- `classifier` (EquipmentClassifier): Trained EquipmentClassifier
- `description` (str, optional): Equipment description. Default is "Heat Exchanger for Chilled Water system with Plate and Frame design".
- `service_life` (float, optional): Service life value. Default is 20.0.
- `logger` (Optional[logging.Logger], optional): Logger instance. Default is None.

**Returns:**

- Dict: Prediction results

**Example:**
```python
from nexusml.train_model_pipeline import make_sample_prediction, setup_logging
from nexusml.core.model import EquipmentClassifier

# Set up logging
logger = setup_logging("INFO")

# Create classifier
classifier = EquipmentClassifier()
# ... train the classifier ...

# Make sample prediction
prediction = make_sample_prediction(
    classifier,
    "500 ton centrifugal chiller with R-134a refrigerant",
    20.0,
    logger
)

# Print prediction
for key, value in prediction.items():
    if key != "attribute_template" and key != "master_db_mapping":
        print(f"{key}: {value}")
```

**Notes:**

- This function makes a sample prediction using the trained model
- It logs the prediction process and results if a logger is provided
- It returns the prediction results as a dictionary

### `main()`

Main function to run the model training pipeline.

**Example:**
```python
from nexusml.train_model_pipeline import main

# Run the model training pipeline
main()
```

**Notes:**

- This function is called when the script is run directly
- It parses command-line arguments, sets up logging, and runs the model training pipeline
- It handles exceptions and exits with a non-zero status code if an error occurs

## Process Flow

The script follows a structured workflow for training the equipment classification model:

1. **Parse Arguments**: Parse command-line arguments to determine data path, feature configuration, output directory, and other options.
2. **Set Up Logging**: Configure logging to output to both a file and the console.
3. **Load Reference Data**: Load reference data using the ReferenceManager.
4. **Validate Training Data**: Validate the training data to ensure it meets quality standards.
5. **Train Model**:
   - Create an EquipmentClassifier instance
   - Train the model with the specified parameters
   - Optionally perform hyperparameter optimization
   - Evaluate the model and calculate metrics
6. **Save Model**:
   - Save the trained model to the specified directory
   - Create versioned files with timestamps
   - Create copies of the latest model and metadata
7. **Generate Visualizations** (if requested):
   - Create category distribution visualizations
   - Create confusion matrices
8. **Make Sample Prediction**:
   - Make a sample prediction using the trained model
   - Log the prediction results

## Output Files

The script generates the following output files:

1. **Model Files**:
   - `{model_name}_{timestamp}.pkl`: Versioned model file
   - `{model_name}_latest.pkl`: Copy of the latest model file
2. **Metadata Files**:
   - `{model_name}_{timestamp}_metadata.json`: Versioned metadata file
   - `{model_name}_latest_metadata.json`: Copy of the latest metadata file
3. **Visualization Files** (if requested):
   - `visualizations/equipment_category_distribution.png`: Equipment category distribution
   - `visualizations/system_type_distribution.png`: System type distribution
   - `visualizations/confusion_matrix_{column}.png`: Confusion matrices for each classification column
4. **Log Files**:
   - `logs/model_training_{timestamp}.log`: Log file with timestamp

## Dependencies

- **argparse**: Used for command-line argument parsing
- **datetime**: Used for timestamps
- **json**: Used for JSON serialization
- **logging**: Used for logging
- **os**: Used for file operations
- **pickle**: Used for model serialization
- **sys**: Used for system operations
- **time**: Used for timing
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **matplotlib.pyplot**: Used for visualizations
- **numpy**: Used for numerical operations
- **pandas**: Used for data manipulation
- **seaborn**: Used for visualizations
- **sklearn.metrics**: Used for evaluation metrics
- **sklearn.model_selection**: Used for train_test_split
- **nexusml.core.data_mapper**: Used for data mapping
- **nexusml.core.data_preprocessing**: Used for data preprocessing
- **nexusml.core.evaluation**: Used for model evaluation
- **nexusml.core.feature_engineering**: Used for feature engineering
- **nexusml.core.model**: Used for the EquipmentClassifier
- **nexusml.core.model_building**: Used for model building and optimization
- **nexusml.core.reference.manager**: Used for reference data management

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The script expects a training data file in CSV format.
- The script creates several directories if they don't exist:
  - "logs" directory for log files
  - Output directory for model files
  - "visualizations" subdirectory for visualization files
- The script validates the training data but continues with training even if validation fails (with a warning).
- The script uses a timestamp for versioning model files and log files.
- The script creates copies of the latest model and metadata files instead of symlinks (which require admin privileges on Windows).
- The hyperparameter optimization process can be time-consuming, especially with large datasets.
- The script logs detailed information about the training process, evaluation metrics, and sample prediction.
- If an error occurs during the training process, the script logs the error and exits with a non-zero status code.
