# Command-Line Tool: predict.py

## Overview

The `predict.py` script is a command-line tool for making equipment classification predictions using a pre-trained model. It loads a trained model, processes input data, makes predictions, and saves the results to a CSV file.

Key features include:

1. **Pre-trained Model Usage**: Uses a previously trained model for predictions
2. **Flexible Input Handling**: Processes input data with various column structures
3. **Feature Engineering**: Applies feature engineering to input data before prediction
4. **Detailed Logging**: Provides comprehensive logging of the prediction process
5. **Progress Tracking**: Shows progress during prediction of large datasets
6. **Configurable Output**: Saves prediction results to a specified CSV file

## Usage

```bash
python predict.py --input-file INPUT_FILE [--model-path MODEL_PATH] [--output-file OUTPUT_FILE] 
                  [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] 
                  [--description-column DESCRIPTION_COLUMN] 
                  [--service-life-column SERVICE_LIFE_COLUMN] 
                  [--asset-tag-column ASSET_TAG_COLUMN]
```

### Arguments

- `--input-file`: Path to the input CSV file with equipment descriptions (required)
- `--model-path`: Path to the trained model file (default: "outputs/models/equipment_classifier_latest.pkl")
- `--output-file`: Path to save the prediction results (default: "prediction_results.csv")
- `--log-level`: Logging level (default: "INFO", choices: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
- `--description-column`: Column name containing equipment descriptions (default: "Description")
- `--service-life-column`: Column name containing service life values (default: "Service Life")
- `--asset-tag-column`: Column name containing asset tags (default: "Asset Tag")

### Examples

#### Basic Usage

```bash
python predict.py --input-file equipment_data.csv
```

This will load the default model, process `equipment_data.csv`, and save the results to `prediction_results.csv`.

#### Specifying Model and Output

```bash
python predict.py --input-file equipment_data.csv --model-path custom_model.pkl --output-file custom_results.csv
```

This will load the model from `custom_model.pkl`, process `equipment_data.csv`, and save the results to `custom_results.csv`.

#### Custom Column Names

```bash
python predict.py --input-file equipment_data.csv --description-column "Equipment Description" --service-life-column "Expected Life" --asset-tag-column "ID"
```

This will use the specified column names when processing the input data.

#### Detailed Logging

```bash
python predict.py --input-file equipment_data.csv --log-level DEBUG
```

This will provide more detailed logging information during the prediction process.

## Functions

### `setup_logging(log_level="INFO")`

Set up logging configuration.

**Parameters:**

- `log_level` (str, optional): Logging level. Default is "INFO".

**Returns:**

- Logger: Configured logger instance

**Example:**
```python
from nexusml.predict import setup_logging

# Set up logging with INFO level
logger = setup_logging("INFO")

# Set up logging with DEBUG level
logger = setup_logging("DEBUG")
```

**Notes:**

- Creates a "logs" directory if it doesn't exist
- Configures logging to output to both a file and the console
- Returns a logger named "equipment_prediction"

### `main()`

Main function to run the prediction script.

**Example:**
```python
from nexusml.predict import main

# Run the prediction script
main()
```

**Notes:**

- This function is called when the script is run directly
- It parses command-line arguments, sets up logging, loads the model, processes input data, makes predictions, and saves the results

## Input Format

The script accepts CSV files with various column structures. It can handle:

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

Example input CSV (fake data format):
```csv
equipment_tag,manufacturer,model,category_name,mcaa_system_category
CH-01,Carrier,30XA,Chiller,HVAC
AH-01,Trane,M-Series,Air Handler,HVAC
P-01,Grundfos,CR,Pump,Plumbing
```

## Output Format

The output is a CSV file containing the prediction results for each equipment item. The exact columns depend on the model's output, but typically include:

```csv
original_description,service_life,asset_tag,category_name,uniformat_code,mcaa_system_category,Equipment_Type,System_Subtype,MasterFormat_Class,Equipment_Category
500 ton water-cooled centrifugal chiller,20,CH-01,Chiller,D3010,HVAC,Chiller-Centrifugal,Cooling,23 64 16,Chiller
10000 CFM air handler with MERV 13 filters,15,AH-01,Air Handler,D3040,HVAC,Air Handler-VAV,Air Distribution,23 74 13,Air Handler
100 GPM circulation pump,10,P-01,Pump,D2010,Plumbing,Pump-Circulation,Water Distribution,22 11 23,Pump
```

## Process Flow

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

## Dependencies

- **argparse**: Used for command-line argument parsing
- **logging**: Used for logging
- **sys**: Used for system operations
- **pathlib**: Used for path manipulation
- **pandas**: Used for data manipulation
- **nexusml.core.model**: Used for the EquipmentClassifier
- **nexusml.core.data_mapper**: Used for mapping staging data to model input
- **nexusml.core.feature_engineering**: Used for feature engineering

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path.
- The input file must be in CSV format.
- The script expects a pre-trained model file. If the model file doesn't exist, the script will exit with an error.
- The script creates a "logs" directory in the current working directory for log files.
- The script creates the output directory if it doesn't exist.
- The script uses the `predict_from_row` method of the EquipmentClassifier, which is designed to work with rows that have already been processed by the feature engineering pipeline.
- For large input files, the script logs progress updates every 10 items.
- The script logs a sample of the predictions (first 3 items) at the end of the process.
- If an error occurs during the prediction process, the script logs the error and exits with a non-zero status code.

## Comparison with classify_equipment.py

While both `predict.py` and `classify_equipment.py` are used for equipment classification, they have some key differences:

1. **Model Usage**:
   - `predict.py` uses a pre-trained model loaded from a file.
   - `classify_equipment.py` trains a new model each time it is run.

2. **Input Handling**:
   - `predict.py` is more flexible with input column names and provides command-line options to specify them.
   - `classify_equipment.py` uses a DynamicFieldMapper to map input columns to the expected model format.

3. **Output Format**:
   - `predict.py` outputs a simple CSV file with prediction results.
   - `classify_equipment.py` can output either JSON or CSV, with the JSON format providing more detailed information.

4. **Logging**:
   - `predict.py` has more comprehensive logging.
   - `classify_equipment.py` has simpler progress reporting.

5. **Prediction Method**:
   - `predict.py` uses the `predict_from_row` method of the EquipmentClassifier.
   - `classify_equipment.py` uses the `predict_with_enhanced_model` function.

Choose `predict.py` when you have a pre-trained model and want to make predictions on new data with detailed logging. Choose `classify_equipment.py` when you want to train a new model and get more detailed output in JSON format.
