# Utility Script: train_model.sh

## Overview

The `train_model.sh` script is a Bash utility that provides a convenient wrapper around the NexusML model training pipeline. It simplifies the process of running the training pipeline with common options, handling command-line arguments, and providing sensible defaults. This script is particularly useful for users who prefer working with shell scripts or need to integrate model training into larger workflows or automation systems.

Key features include:

1. **Command-Line Interface**: Easy-to-use CLI with intuitive options
2. **Default Values**: Sensible defaults for common parameters
3. **Argument Validation**: Basic validation of required arguments
4. **Help Documentation**: Built-in help message with usage examples
5. **Pipeline Integration**: Seamless integration with the Python training pipeline

## Usage

```bash
./train_model.sh [options]
```

### Options

- `-d, --data-path PATH`: Path to the training data CSV file (required)
- `-f, --feature-config PATH`: Path to the feature configuration YAML file
- `-r, --reference-config PATH`: Path to the reference configuration YAML file
- `-o, --output-dir DIR`: Directory to save the trained model (default: outputs/models)
- `-n, --model-name NAME`: Base name for the saved model (default: equipment_classifier)
- `-t, --test-size SIZE`: Proportion of data to use for testing (default: 0.3)
- `-s, --random-state STATE`: Random state for reproducibility (default: 42)
- `-g, --sampling-strategy STR`: Sampling strategy for class imbalance (default: direct)
- `-l, --log-level LEVEL`: Logging level (default: INFO)
- `-p, --optimize`: Perform hyperparameter optimization
- `-v, --visualize`: Generate visualizations of model performance
- `-h, --help`: Show the help message

### Examples

#### Basic Usage

```bash
./train_model.sh -d files/training-data/equipment_data.csv
```

This will train a model using the specified data file with default settings.

#### With Hyperparameter Optimization and Visualizations

```bash
./train_model.sh -d files/training-data/equipment_data.csv -p -v
```

This will train a model with hyperparameter optimization and generate visualizations of model performance.

#### Custom Configuration and Output

```bash
./train_model.sh -d files/training-data/equipment_data.csv \
                -f config/custom_features.yml \
                -r config/custom_reference.yml \
                -o custom_models \
                -n custom_classifier
```

This will train a model using custom feature and reference configurations and save it to a custom output directory with a custom name.

#### Advanced Options

```bash
./train_model.sh -d files/training-data/equipment_data.csv \
                -t 0.2 \
                -s 123 \
                -g stratified \
                -l DEBUG \
                -p -v
```

This will train a model with a 20% test split, random state 123, stratified sampling, DEBUG logging level, hyperparameter optimization, and visualizations.

## Script Structure

The script is structured as follows:

1. **Default Values**: Sets default values for all parameters
2. **Help Function**: Defines a function to display the help message
3. **Argument Parsing**: Parses command-line arguments using a while loop
4. **Validation**: Checks if required arguments are provided
5. **Command Building**: Builds the Python command with all options
6. **Execution**: Prints and executes the command

## Default Values

The script sets the following default values:

```bash
DATA_PATH=""                    # No default, must be provided
FEATURE_CONFIG=""               # No default
REFERENCE_CONFIG=""             # No default
OUTPUT_DIR="outputs/models"     # Default output directory
MODEL_NAME="equipment_classifier" # Default model name
TEST_SIZE=0.3                   # Default test size
RANDOM_STATE=42                 # Default random state
SAMPLING_STRATEGY="direct"      # Default sampling strategy
LOG_LEVEL="INFO"                # Default log level
OPTIMIZE=false                  # Hyperparameter optimization disabled by default
VISUALIZE=false                 # Visualizations disabled by default
```

## Command Generation

The script generates a Python command to run the training pipeline with the specified options. The generated command looks like:

```bash
python -m nexusml.train_model_pipeline --data-path "files/training-data/equipment_data.csv" \
    --output-dir "outputs/models" --model-name "equipment_classifier" \
    --test-size 0.3 --random-state 42 --sampling-strategy direct \
    --log-level INFO
```

Additional options are added based on the provided arguments:

- If `FEATURE_CONFIG` is provided: `--feature-config "path/to/config.yml"`
- If `REFERENCE_CONFIG` is provided: `--reference-config "path/to/config.yml"`
- If `OPTIMIZE` is true: `--optimize`
- If `VISUALIZE` is true: `--visualize`

## Integration with train_model_pipeline.py

The script integrates with the `nexusml.train_model_pipeline` Python module, which is the actual implementation of the model training pipeline. The script simply provides a convenient shell interface to this Python module, handling argument parsing and command generation.

For detailed information about the training pipeline itself, see the [train_model_pipeline.py documentation](../cli/train_model_pipeline.md).

## Dependencies

- **Bash**: The script is written in Bash and requires a Bash-compatible shell
- **Python**: The script runs a Python module, so Python must be installed
- **nexusml.train_model_pipeline**: The Python module that implements the training pipeline

## Notes and Warnings

- The script must be run from a directory where the `nexusml` package is available in the Python path
- The script requires execute permissions (`chmod +x train_model.sh`)
- The data path argument (`-d, --data-path`) is required; the script will exit with an error if it's not provided
- The script does not validate that the specified files or directories exist; it passes them directly to the Python module
- The script uses double quotes around file paths to handle paths with spaces
- The script prints the generated command before executing it, which is useful for debugging
- The script passes all arguments directly to the Python module without additional validation
- The script does not capture or process the output of the Python module; it simply passes it through to the console
- The script does not handle signals or provide any way to interrupt the training process other than standard shell interrupts (e.g., Ctrl+C)

## Common Use Cases

### Training a Model with Default Settings

```bash
./train_model.sh -d files/training-data/equipment_data.csv
```

### Training a Model with Hyperparameter Optimization

```bash
./train_model.sh -d files/training-data/equipment_data.csv -p
```

### Training a Model with Visualizations

```bash
./train_model.sh -d files/training-data/equipment_data.csv -v
```

### Training a Model with Custom Configuration

```bash
./train_model.sh -d files/training-data/equipment_data.csv -f config/custom_features.yml
```

### Training a Model with Custom Output

```bash
./train_model.sh -d files/training-data/equipment_data.csv -o custom_models -n custom_classifier
```

### Training a Model with Reproducible Results

```bash
./train_model.sh -d files/training-data/equipment_data.csv -s 42
```

### Training a Model with Detailed Logging

```bash
./train_model.sh -d files/training-data/equipment_data.csv -l DEBUG
```

## Customization

The script can be customized by modifying the default values at the beginning of the script. For example, to change the default output directory:

```bash
# Default values
DATA_PATH=""
FEATURE_CONFIG=""
REFERENCE_CONFIG=""
OUTPUT_DIR="custom/output/directory"  # Changed default output directory
MODEL_NAME="equipment_classifier"
# ... other defaults ...
```

Additional options can be added by modifying the argument parsing section and the command building section. For example, to add a new option for specifying the number of cross-validation folds:

```bash
# Add to default values
CV_FOLDS=5

# Add to argument parsing
case "$1" in
    # ... existing cases ...
    -c|--cv-folds)
        CV_FOLDS="$2"
        shift 2
        ;;
    # ... existing cases ...
esac

# Add to command building
CMD="$CMD --cv-folds $CV_FOLDS"
```

## Error Handling

The script provides basic error handling:

- It checks if the required data path argument is provided
- It displays an error message and the help text if the data path is missing
- It exits with a non-zero status code if there's an error
- It passes unknown options to the help function with an error message

However, it does not handle errors from the Python module itself. If the Python module fails, the error will be displayed in the console, but the script does not provide any additional error handling or recovery.