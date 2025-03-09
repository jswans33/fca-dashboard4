# Command-Line Tool: train_model_pipeline_v2.py

## Overview

The `train_model_pipeline_v2.py` script is an enhanced version of the equipment classification model training pipeline that introduces a pipeline orchestrator approach while maintaining backward compatibility with the original implementation. It provides a more modular and extensible way to train the equipment classification model.

Key features include:

1. **Dual Implementation**: Supports both the legacy training approach and the new pipeline orchestrator approach
2. **Pipeline Orchestration**: Uses a pipeline orchestrator to manage the training workflow
3. **Component Registry**: Uses a component registry to manage pipeline components
4. **Dependency Injection**: Leverages a dependency injection container for better testability and modularity
5. **Configuration-Based Validation**: Validates training data using required columns from a configuration file
6. **Performance Metrics**: Provides detailed execution time metrics for each component in the pipeline
7. **Backward Compatibility**: Maintains full compatibility with the original train_model_pipeline.py script

## Usage

```bash
python train_model_pipeline_v2.py --data-path DATA_PATH [options]
```

### Arguments

#### Data Arguments

- `--data-path`: Path to the training data CSV file
- `--feature-config-path`: Path to the feature configuration YAML file
- `--reference-config-path`: Path to the reference configuration YAML file

#### Training Arguments

- `--test-size`: Proportion of data to use for testing (default: 0.3)
- `--random-state`: Random state for reproducibility (default: 42)
- `--sampling-strategy`: Sampling strategy for handling class imbalance (default: "direct", choices: ["direct"])

#### Optimization Arguments

- `--optimize-hyperparameters`: Perform hyperparameter optimization (flag)

#### Output Arguments

- `--output-dir`: Directory to save the trained model and results (default: "outputs/models")
- `--model-name`: Base name for the saved model (default: "equipment_classifier")

#### Logging Arguments

- `--log-level`: Logging level (default: "INFO", choices: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

#### Visualization Arguments

- `--visualize`: Generate visualizations of model performance (flag)

#### Implementation Arguments

- `--use-orchestrator`: Use the pipeline orchestrator for training (flag, default: False)

### Examples

#### Basic Usage (Legacy Mode)

```bash
python train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv
```

This will train a model using the legacy implementation and save it to the default output directory.

#### Using Pipeline Orchestrator

```bash
python train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv --use-orchestrator
```

This will train a model using the pipeline orchestrator and save it to the default output directory.

#### With Hyperparameter Optimization and Visualizations

```bash
python train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv --optimize-hyperparameters --visualize --use-orchestrator
```

This will train a model using the pipeline orchestrator with hyperparameter optimization and generate visualizations of model performance.

#### Custom Configuration and Output

```bash
python train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv --feature-config-path config/custom_features.yml --reference-config-path config/custom_reference.yml --output-dir custom_models --model-name custom_classifier --use-orchestrator
```

This will train a model using the pipeline orchestrator with custom feature and reference configurations and save it to a custom output directory with a custom name.

## Functions

### `validate_data_from_config(data_path: str, logger=None) -> Dict`

Validate the training data using required columns from data_config.yml.

**Parameters:**

- `data_path` (str): Path to the training data
- `logger` (optional): Logger instance

**Returns:**

- Dict: Validation results dictionary

**Example:**
```python
from nexusml.train_model_pipeline_v2 import validate_data_from_config
import logging

# Set up logging
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Validate data
validation_results = validate_data_from_config("files/training-data/equipment_data.csv", logger)
print(f"Valid: {validation_results.get('valid', False)}")
if not validation_results.get('valid', False):
    print("Issues:")
    for issue in validation_results.get('issues', []):
        print(f"  - {issue}")
```

**Notes:**

- This function checks:
  1. If the file exists and can be read
  2. If required columns from production_data_config.yml are present
  3. If data types are correct
  4. If there are any missing values in critical columns
- If the configuration file doesn't exist or can't be parsed, it falls back to a hardcoded list of required columns

### `create_orchestrator(logger) -> PipelineOrchestrator`

Create a PipelineOrchestrator instance with registered components.

**Parameters:**

- `logger`: Logger instance

**Returns:**

- PipelineOrchestrator: Configured PipelineOrchestrator

**Example:**
```python
from nexusml.train_model_pipeline_v2 import create_orchestrator
import logging

# Set up logging
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Create orchestrator
orchestrator = create_orchestrator(logger)
```

**Notes:**

- This function creates a component registry and registers default implementations for:
  - DataLoader: Loads data from CSV or Excel files
  - DataPreprocessor: Preprocesses the input data
  - FeatureEngineer: Engineers features from the input data
  - ModelBuilder: Builds a machine learning model
  - ModelTrainer: Trains a model on the provided data
  - ModelEvaluator: Evaluates a trained model on test data
  - ModelSerializer: Saves and loads trained models
  - Predictor: Makes predictions using a trained model
- It creates a dependency injection container, pipeline factory, pipeline context, and pipeline orchestrator
- The implementations provided are simplified for demonstration purposes

### `train_with_orchestrator(args: TrainingArguments, logger) -> Tuple[Pipeline, Dict, Optional[Dict]]`

Train a model using the pipeline orchestrator.

**Parameters:**

- `args` (TrainingArguments): Training arguments
- `logger`: Logger instance

**Returns:**

- Tuple[Pipeline, Dict, Optional[Dict]]: Tuple containing:
  - Trained model
  - Metrics dictionary
  - Visualization paths dictionary (if visualize=True)

**Example:**
```python
from nexusml.train_model_pipeline_v2 import train_with_orchestrator
from nexusml.core.cli.training_args import TrainingArguments
import logging

# Set up logging
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Create arguments
args = TrainingArguments()
args.data_path = "files/training-data/equipment_data.csv"
args.feature_config_path = "config/feature_config.yml"
args.test_size = 0.3
args.random_state = 42
args.optimize_hyperparameters = True
args.output_dir = "outputs/models"
args.model_name = "equipment_classifier"
args.visualize = True

# Train model
model, metrics, viz_paths = train_with_orchestrator(args, logger)
```

**Notes:**

- This function creates a pipeline orchestrator and uses it to train a model
- It logs the execution summary, including component execution times
- It makes a sample prediction using the trained model
- It generates visualizations if requested
- It handles exceptions and logs detailed error information

### `make_sample_prediction_with_orchestrator(orchestrator: PipelineOrchestrator, model: Pipeline, logger, description: str = "Heat Exchanger for Chilled Water system with Plate and Frame design", service_life: float = 20.0) -> Dict`

Make a sample prediction using the trained model and orchestrator.

**Parameters:**

- `orchestrator` (PipelineOrchestrator): Pipeline orchestrator
- `model` (Pipeline): Trained model
- `logger`: Logger instance
- `description` (str, optional): Equipment description. Default is "Heat Exchanger for Chilled Water system with Plate and Frame design".
- `service_life` (float, optional): Service life value. Default is 20.0.

**Returns:**

- Dict: Prediction results

**Example:**
```python
from nexusml.train_model_pipeline_v2 import create_orchestrator, make_sample_prediction_with_orchestrator
import logging
from sklearn.pipeline import Pipeline

# Set up logging
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# Create orchestrator
orchestrator = create_orchestrator(logger)

# Create a dummy model
model = Pipeline([])

# Make sample prediction
prediction = make_sample_prediction_with_orchestrator(
    orchestrator,
    model,
    logger,
    "500 ton centrifugal chiller with R-134a refrigerant",
    20.0
)

# Print prediction
for key, value in prediction.items():
    print(f"{key}: {value}")
```

**Notes:**

- This function creates sample data for prediction
- It uses the orchestrator to make predictions
- It logs the prediction results
- It handles exceptions and logs detailed error information

### `main()`

Main function to run the model training pipeline.

**Example:**
```python
from nexusml.train_model_pipeline_v2 import main

# Run the model training pipeline
main()
```

**Notes:**

- This function is called when the script is run directly
- It parses command-line arguments, sets up logging, and runs the model training pipeline
- It handles exceptions and exits with a non-zero status code if an error occurs
- It supports both the legacy implementation and the new orchestrator-based implementation based on the `--use-orchestrator` flag

## Pipeline Components

The script defines several pipeline components that are registered with the component registry:

### DataLoader

Loads data from CSV or Excel files.

**Methods:**

- `load_data(data_path=None, **kwargs)`: Load data from a file (CSV or Excel)
- `get_config()`: Get the configuration for the data loader

### DataPreprocessor

Preprocesses the input data.

**Methods:**

- `preprocess(data, **kwargs)`: Preprocess the input data
- `verify_required_columns(data)`: Verify that all required columns exist in the DataFrame

### FeatureEngineer

Engineers features from the input data.

**Methods:**

- `engineer_features(data, **kwargs)`: Engineer features from the input data
- `fit(data, **kwargs)`: Fit the feature engineer to the input data
- `transform(data, **kwargs)`: Transform the input data using the fitted feature engineer

### ModelBuilder

Builds a machine learning model.

**Methods:**

- `build_model(**kwargs)`: Build a machine learning model
- `optimize_hyperparameters(model, x_train, y_train, **kwargs)`: Optimize hyperparameters for the model

### ModelTrainer

Trains a model on the provided data.

**Methods:**

- `train(model, x_train, y_train, **kwargs)`: Train a model on the provided data
- `cross_validate(model, x, y, **kwargs)`: Perform cross-validation on the model

### ModelEvaluator

Evaluates a trained model on test data.

**Methods:**

- `evaluate(model, x_test, y_test, **kwargs)`: Evaluate a trained model on test data
- `analyze_predictions(model, x_test, y_test, y_pred, **kwargs)`: Analyze model predictions in detail

### ModelSerializer

Saves and loads trained models.

**Methods:**

- `save_model(model, path, **kwargs)`: Save a trained model to disk
- `load_model(path, **kwargs)`: Load a trained model from disk

### Predictor

Makes predictions using a trained model.

**Methods:**

- `predict(model, data, **kwargs)`: Make predictions using a trained model
- `predict_proba(model, data, **kwargs)`: Make probability predictions using a trained model

## Process Flow

### Orchestrator Implementation

1. **Parse Arguments**: Parse command-line arguments to determine data path, feature configuration, output directory, and other options.
2. **Set Up Logging**: Configure logging to output to both a file and the console.
3. **Load Reference Data**: Load reference data using the ReferenceManager.
4. **Validate Training Data**: Validate the training data using required columns from a configuration file.
5. **Create Orchestrator**: Create a pipeline orchestrator with registered components.
6. **Train Model**: Use the orchestrator to train a model with the specified parameters.
7. **Log Execution Summary**: Log the execution summary, including component execution times.
8. **Make Sample Prediction**: Make a sample prediction using the trained model and orchestrator.
9. **Generate Visualizations**: Generate visualizations of model performance if requested.

### Legacy Implementation

1. **Parse Arguments**: Parse command-line arguments to determine data path, feature configuration, output directory, and other options.
2. **Set Up Logging**: Configure logging to output to both a file and the console.
3. **Load Reference Data**: Load reference data using the ReferenceManager.
4. **Validate Training Data**: Validate the training data using required columns from a configuration file.
5. **Train Model**: Use the legacy implementation to train a model with the specified parameters.
6. **Save Model**: Save the trained model to the specified directory.
7. **Generate Visualizations**: Generate visualizations of model performance if requested.
8. **Make Sample Prediction**: Make a sample prediction using the trained model.

## Dependencies

- **datetime**: Used for timestamps
- **json**: Used for JSON serialization
- **os**: Used for file operations
- **sys**: Used for system operations
- **time**: Used for timing
- **traceback**: Used for detailed error information
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **pandas**: Used for data manipulation
- **sklearn.pipeline**: Used for Pipeline class
- **nexusml.core.cli.training_args**: Used for training arguments
- **nexusml.core.di.container**: Used for dependency injection
- **nexusml.core.pipeline.context**: Used for pipeline context
- **nexusml.core.pipeline.factory**: Used for pipeline factory
- **nexusml.core.pipeline.orchestrator**: Used for pipeline orchestration
- **nexusml.core.pipeline.registry**: Used for component registry
- **nexusml.core.reference.manager**: Used for reference data management
- **nexusml.train_model_pipeline**: Used for legacy implementation
- **nexusml.config**: Used for configuration file paths

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The script expects a training data file in CSV format.
- The script creates several directories if they don't exist:
  - "logs" directory for log files
  - Output directory for model files
  - "visualizations" subdirectory for visualization files
- The script validates the training data but continues with training even if validation fails (with a warning).
- The script uses a timestamp for versioning model files and log files.
- The script supports both the legacy implementation and the new orchestrator-based implementation based on the `--use-orchestrator` flag.
- The implementations provided for the pipeline components are simplified for demonstration purposes.
- If the feature_config_path is not specified when using the orchestrator, it defaults to production_data_config.yml.
- The script handles exceptions and logs detailed error information.
- If an error occurs during the training process, the script logs the error and exits with a non-zero status code.

## Comparison with train_model_pipeline.py

While both `train_model_pipeline_v2.py` and `train_model_pipeline.py` are used for training the equipment classification model, `train_model_pipeline_v2.py` offers several advantages:

1. **Dual Implementation**: Supports both the legacy training approach and the new pipeline orchestrator approach
2. **Pipeline Orchestration**: Uses a pipeline orchestrator to manage the training workflow
3. **Component Registry**: Uses a component registry to manage pipeline components
4. **Dependency Injection**: Leverages a dependency injection container for better testability and modularity
5. **Configuration-Based Validation**: Validates training data using required columns from a configuration file
6. **Performance Metrics**: Provides detailed execution time metrics for each component in the pipeline

Choose `train_model_pipeline_v2.py` when you want to use the new pipeline orchestrator approach or when you need detailed execution time metrics. Choose `train_model_pipeline.py` when you want a simpler implementation without the additional dependencies.
