# Command-Line Tool: predict_v2.py

## Overview

The `predict_v2.py` script is an enhanced version of the equipment classification prediction tool that introduces a pipeline orchestrator approach while maintaining backward compatibility with the original implementation. It provides a more modular and extensible way to make predictions on equipment data.

Key features include:

1. **Dual Implementation**: Supports both the legacy prediction approach and the new pipeline orchestrator approach
2. **Pipeline Orchestration**: Uses a pipeline orchestrator to manage the prediction workflow
3. **Dependency Injection**: Leverages a dependency injection container for better testability and modularity
4. **Component Registry**: Uses a component registry to manage pipeline components
5. **Performance Metrics**: Provides detailed execution time metrics for each component in the pipeline
6. **Backward Compatibility**: Maintains full compatibility with the original predict.py script

## Usage

```bash
python predict_v2.py --input-file INPUT_FILE [--model-path MODEL_PATH] [--output-file OUTPUT_FILE] 
                     [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] 
                     [--description-column DESCRIPTION_COLUMN] 
                     [--service-life-column SERVICE_LIFE_COLUMN] 
                     [--asset-tag-column ASSET_TAG_COLUMN]
                     [--feature-config-path FEATURE_CONFIG_PATH]
                     [--use-orchestrator]
```

### Arguments

- `--input-file`: Path to the input CSV file with equipment descriptions (required)
- `--model-path`: Path to the trained model file (default: "outputs/models/equipment_classifier_latest.pkl")
- `--output-file`: Path to save the prediction results (default: "prediction_results.csv")
- `--log-level`: Logging level (default: "INFO", choices: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
- `--description-column`: Column name containing equipment descriptions (default: "Description")
- `--service-life-column`: Column name containing service life values (default: "Service Life")
- `--asset-tag-column`: Column name containing asset tags (default: "Asset Tag")
- `--feature-config-path`: Path to the feature configuration file (optional)
- `--use-orchestrator`: Use the pipeline orchestrator for prediction (flag, default: False)

### Examples

#### Basic Usage (Legacy Mode)

```bash
python predict_v2.py --input-file equipment_data.csv
```

This will use the legacy prediction implementation to process `equipment_data.csv` and save the results to `prediction_results.csv`.

#### Using Pipeline Orchestrator

```bash
python predict_v2.py --input-file equipment_data.csv --use-orchestrator
```

This will use the pipeline orchestrator to process `equipment_data.csv` and save the results to `prediction_results.csv`.

#### Custom Configuration with Orchestrator

```bash
python predict_v2.py --input-file equipment_data.csv --model-path custom_model.pkl --output-file custom_results.csv --feature-config-path custom_features.yml --use-orchestrator
```

This will use the pipeline orchestrator with custom model, output, and feature configuration.

#### Detailed Logging with Legacy Mode

```bash
python predict_v2.py --input-file equipment_data.csv --log-level DEBUG
```

This will use the legacy prediction implementation with detailed logging.

## Functions

### `create_orchestrator(logger: logging.Logger) -> PipelineOrchestrator`

Create a PipelineOrchestrator instance with all required components.

**Parameters:**

- `logger` (logging.Logger): Logger instance for logging messages.

**Returns:**

- PipelineOrchestrator: Configured PipelineOrchestrator instance.

**Example:**
```python
from nexusml.predict_v2 import create_orchestrator
import logging

# Set up logging
logger = logging.getLogger("equipment_prediction")
logger.setLevel(logging.INFO)

# Create orchestrator
orchestrator = create_orchestrator(logger)
```

**Notes:**

- Creates a component registry for pipeline components
- Creates a dependency injection container
- Creates a pipeline factory using the registry and container
- Creates a pipeline context for sharing data between components
- Creates and returns a pipeline orchestrator

### `run_legacy_prediction(args, logger: logging.Logger) -> None`

Run the prediction using the legacy implementation.

**Parameters:**

- `args`: Command-line arguments.
- `logger` (logging.Logger): Logger instance for logging messages.

**Raises:**

- SystemExit: If an error occurs during prediction.

**Example:**
```python
from nexusml.predict_v2 import run_legacy_prediction
import logging
import argparse

# Set up logging
logger = logging.getLogger("equipment_prediction")
logger.setLevel(logging.INFO)

# Create arguments
args = argparse.Namespace()
args.model_path = "model.pkl"
args.input_file = "data.csv"
args.output_file = "results.csv"
args.description_column = "Description"
args.service_life_column = "Service Life"
args.asset_tag_column = "Asset Tag"

# Run legacy prediction
run_legacy_prediction(args, logger)
```

**Notes:**

- This function implements the same prediction logic as the original predict.py script
- It loads the model, processes the input data, makes predictions, and saves the results
- It provides detailed logging throughout the process

### `run_orchestrator_prediction(args, logger: logging.Logger) -> None`

Run the prediction using the pipeline orchestrator.

**Parameters:**

- `args`: Command-line arguments.
- `logger` (logging.Logger): Logger instance for logging messages.

**Raises:**

- SystemExit: If an error occurs during prediction.

**Example:**
```python
from nexusml.predict_v2 import run_orchestrator_prediction
import logging
import argparse

# Set up logging
logger = logging.getLogger("equipment_prediction")
logger.setLevel(logging.INFO)

# Create arguments
args = argparse.Namespace()
args.model_path = "model.pkl"
args.input_file = "data.csv"
args.output_file = "results.csv"
args.feature_config_path = "features.yml"
args.description_column = "Description"
args.service_life_column = "Service Life"
args.asset_tag_column = "Asset Tag"

# Run orchestrator prediction
run_orchestrator_prediction(args, logger)
```

**Notes:**

- This function uses the pipeline orchestrator to manage the prediction workflow
- It creates an orchestrator, loads the input data, makes predictions, and logs the results
- It provides detailed execution time metrics for each component in the pipeline

### `main() -> None`

Main function to run the prediction script.

**Example:**
```python
from nexusml.predict_v2 import main

# Run the prediction script
main()
```

**Notes:**

- This function is called when the script is run directly
- It parses command-line arguments, sets up logging, and runs the appropriate prediction implementation based on the feature flag
- It handles exceptions and exits with a non-zero status code if an error occurs

## Input Format

The script accepts CSV files with various column structures, similar to the original predict.py script. It can handle:

1. **Standard Format**: A CSV file with a description column, service life column, and asset tag column
2. **Fake Data Format**: A CSV file with "equipment_tag", "manufacturer", and "model" columns
3. **Custom Format**: A CSV file with custom column names specified via command-line arguments

Example input CSV (standard format):
```csv
Description,Service Life,Asset Tag
500 ton water-cooled centrifugal chiller,20,CH-01
10000 CFM air handler with MERV 13 filters,15,AH-01
100 GPM circulation pump,10,P-01
```

## Output Format

The output is a CSV file containing the prediction results for each equipment item, similar to the original predict.py script. The exact columns depend on the model's output and the prediction implementation used.

## Pipeline Orchestrator

The pipeline orchestrator is a new feature in predict_v2.py that provides a more modular and extensible way to manage the prediction workflow. It consists of the following components:

1. **Component Registry**: Manages the registration and retrieval of pipeline components
2. **Dependency Injection Container**: Manages the creation and injection of dependencies
3. **Pipeline Factory**: Creates pipeline instances based on the registry and container
4. **Pipeline Context**: Shares data between pipeline components
5. **Pipeline Orchestrator**: Coordinates the execution of the pipeline

The orchestrator approach offers several advantages over the legacy implementation:

1. **Modularity**: Each step in the prediction process is a separate component that can be developed, tested, and maintained independently
2. **Extensibility**: New components can be added to the pipeline without modifying existing code
3. **Testability**: Components can be tested in isolation, making it easier to write unit tests
4. **Performance Metrics**: The orchestrator provides detailed execution time metrics for each component
5. **Dependency Management**: The dependency injection container manages the creation and injection of dependencies

## Process Flow

### Legacy Implementation

1. **Parse Arguments**: The script parses command-line arguments to determine input file, model path, output file, and other options.
2. **Set Up Logging**: It configures logging to output to both a file and the console.
3. **Load Model**: It loads the pre-trained model from the specified path.
4. **Load Input Data**: It loads the input data from the specified CSV file.
5. **Check Columns**: It checks if the input data has the expected columns.
6. **Apply Feature Engineering**:
   - It maps staging data columns to the model input format.
   - It applies feature engineering to the input data.
7. **Make Predictions**:
   - For each row in the processed data:
     - It extracts the description, service life, and asset tag.
     - It makes a prediction using the model.
     - It adds the original description, service life, and asset tag to the prediction results.
   - It tracks progress and logs updates.
8. **Save Results**: It converts the prediction results to a DataFrame and saves it to the specified CSV file.
9. **Print Summary**: It logs a summary of the prediction process and a sample of the predictions.

### Orchestrator Implementation

1. **Parse Arguments**: The script parses command-line arguments to determine input file, model path, output file, and other options.
2. **Set Up Logging**: It configures logging to output to both a file and the console.
3. **Create Orchestrator**: It creates a pipeline orchestrator with all required components.
4. **Load Input Data**: It loads the input data from the specified CSV file.
5. **Make Predictions**: It uses the orchestrator to make predictions on the input data.
6. **Print Summary**: It logs a summary of the prediction process, including execution time metrics for each component.
7. **Print Sample Predictions**: It logs a sample of the predictions.

## Dependencies

- **logging**: Used for logging
- **sys**: Used for system operations
- **pathlib**: Used for path manipulation
- **typing**: Used for type hints
- **pandas**: Used for data manipulation
- **nexusml.core.cli.prediction_args**: Used for command-line argument parsing
- **nexusml.core.di.container**: Used for dependency injection
- **nexusml.core.pipeline.context**: Used for pipeline context
- **nexusml.core.pipeline.factory**: Used for pipeline factory
- **nexusml.core.pipeline.orchestrator**: Used for pipeline orchestration
- **nexusml.core.pipeline.registry**: Used for component registry
- **nexusml.core.model**: Used for the EquipmentClassifier (legacy implementation)
- **nexusml.core.data_mapper**: Used for mapping staging data to model input (legacy implementation)
- **nexusml.core.feature_engineering**: Used for feature engineering (legacy implementation)

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The input file must be in CSV format.
- The script expects a pre-trained model file. If the model file doesn't exist, the script will exit with an error.
- The script creates a "logs" directory in the current working directory for log files.
- The script creates the output directory if it doesn't exist.
- The `--use-orchestrator` flag determines which implementation to use:
  - If not specified, the script uses the legacy implementation (same as predict.py)
  - If specified, the script uses the new pipeline orchestrator implementation
- The orchestrator implementation provides detailed execution time metrics for each component in the pipeline.
- If an error occurs during the prediction process, the script logs the error and exits with a non-zero status code.

## Comparison with predict.py

While both `predict_v2.py` and `predict.py` are used for equipment classification prediction, `predict_v2.py` offers several advantages:

1. **Dual Implementation**: Supports both the legacy prediction approach and the new pipeline orchestrator approach
2. **Pipeline Orchestration**: Uses a pipeline orchestrator to manage the prediction workflow
3. **Dependency Injection**: Leverages a dependency injection container for better testability and modularity
4. **Component Registry**: Uses a component registry to manage pipeline components
5. **Performance Metrics**: Provides detailed execution time metrics for each component in the pipeline

Choose `predict_v2.py` when you want to use the new pipeline orchestrator approach or when you need detailed execution time metrics. Choose `predict.py` when you want a simpler implementation without the additional dependencies.
