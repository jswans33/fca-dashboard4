# Command-Line Arguments Documentation

## Overview

The training pipeline entry point provides a comprehensive set of command-line
arguments for configuring the training process. These arguments are defined in
`nexusml/core/cli/training_args.py` and are parsed using the `argparse` module.

## Argument Categories

The command-line arguments are organized into the following categories:

1. **Data Arguments**: Configure data sources
2. **Training Arguments**: Configure training parameters
3. **Optimization Arguments**: Configure hyperparameter optimization
4. **Output Arguments**: Configure output paths and formats
5. **Logging Arguments**: Configure logging behavior
6. **Visualization Arguments**: Configure visualization generation
7. **Feature Flags**: Toggle between implementations

## Argument Reference

### Data Arguments

| Argument             | Type   | Default    | Description                                   |
| -------------------- | ------ | ---------- | --------------------------------------------- |
| `--data-path`        | string | (required) | Path to the training data CSV file            |
| `--feature-config`   | string | None       | Path to the feature configuration YAML file   |
| `--reference-config` | string | None       | Path to the reference configuration YAML file |

### Training Arguments

| Argument              | Type   | Default  | Description                                        |
| --------------------- | ------ | -------- | -------------------------------------------------- |
| `--test-size`         | float  | 0.3      | Proportion of data to use for testing (0.0 to 1.0) |
| `--random-state`      | int    | 42       | Random state for reproducibility                   |
| `--sampling-strategy` | string | "direct" | Sampling strategy for handling class imbalance     |

### Optimization Arguments

| Argument     | Type | Default | Description                         |
| ------------ | ---- | ------- | ----------------------------------- |
| `--optimize` | flag | False   | Perform hyperparameter optimization |

### Output Arguments

| Argument       | Type   | Default                | Description                                     |
| -------------- | ------ | ---------------------- | ----------------------------------------------- |
| `--output-dir` | string | "outputs/models"       | Directory to save the trained model and results |
| `--model-name` | string | "equipment_classifier" | Base name for the saved model                   |

### Logging Arguments

| Argument      | Type   | Default | Description                                           |
| ------------- | ------ | ------- | ----------------------------------------------------- |
| `--log-level` | string | "INFO"  | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

### Visualization Arguments

| Argument      | Type | Default | Description                                  |
| ------------- | ---- | ------- | -------------------------------------------- |
| `--visualize` | flag | False   | Generate visualizations of model performance |

### Feature Flags

| Argument   | Type | Default | Description                                       |
| ---------- | ---- | ------- | ------------------------------------------------- |
| `--legacy` | flag | False   | Use legacy implementation instead of orchestrator |

## Validation

The command-line arguments are validated to ensure they meet the following
criteria:

- **Data Paths**: Must exist on the file system
- **Test Size**: Must be between 0.0 and 1.0 (exclusive)
- **Sampling Strategy**: Must be one of the supported strategies
- **Log Level**: Must be one of the supported log levels

## Examples

### Basic Usage

```bash
python nexusml/train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv
```

### Advanced Usage

```bash
python nexusml/train_model_pipeline_v2.py \
    --data-path files/training-data/equipment_data.csv \
    --feature-config configs/features.yml \
    --reference-config configs/references.yml \
    --test-size 0.2 \
    --random-state 123 \
    --optimize \
    --output-dir outputs/models/experiment1 \
    --model-name custom_model \
    --log-level DEBUG \
    --visualize
```

### Using Legacy Implementation

```bash
python nexusml/train_model_pipeline_v2.py \
    --data-path files/training-data/equipment_data.csv \
    --legacy
```

## Implementation Details

The command-line arguments are implemented using the `argparse` module and are
encapsulated in the `TrainingArguments` class. This class provides validation
and conversion of the arguments to a format suitable for use by the training
pipeline.

### TrainingArguments Class

The `TrainingArguments` class is a dataclass that encapsulates all the arguments
needed for training the model. It provides the following features:

- **Validation**: Validates arguments after initialization
- **Default Values**: Provides sensible default values for optional arguments
- **Conversion**: Converts arguments to the appropriate types
- **Dictionary Conversion**: Converts arguments to a dictionary for
  serialization

### Argument Parsing

The `parse_args` function parses command-line arguments using the `argparse`
module and returns a `TrainingArguments` object. It handles the following:

- **Required Arguments**: Enforces required arguments
- **Default Values**: Provides default values for optional arguments
- **Type Conversion**: Converts arguments to the appropriate types
- **Choices**: Restricts arguments to a set of valid choices
- **Help Text**: Provides help text for each argument
