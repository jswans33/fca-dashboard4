# NexusML Command-Line Interface Documentation

This directory contains documentation for the command-line interface (CLI) tools provided by the NexusML package. These tools enable users to perform common tasks from the command line without writing Python code.

## Overview

NexusML provides several command-line tools for tasks such as training models, making predictions, and validating reference data. These tools are designed to be easy to use and integrate into scripts and workflows.

## CLI Tools

- [classify_equipment.py](classify_equipment.md): Tool for classifying equipment from any input format
- [predict.py](predict.md): Tool for making predictions using the original pipeline
- [predict_v2.py](predict_v2.md): Tool for making predictions using the new pipeline architecture
- [train_model_pipeline.py](train_model_pipeline.md): Tool for training a model using the original pipeline
- [train_model_pipeline_v2.py](train_model_pipeline_v2.md): Tool for training a model using the new pipeline architecture
- [test_reference_validation.py](test_reference_validation.md): Tool for validating reference data

## Common Usage Patterns

### Training a Model

```bash
# Train a model using the new pipeline architecture
python -m nexusml.train_model_pipeline_v2 --data-path path/to/training_data.csv --optimize
```

### Making Predictions

```bash
# Make predictions using the new pipeline architecture
python -m nexusml.predict_v2 --model-path outputs/models/equipment_classifier_latest.pkl --input-file path/to/prediction_data.csv
```

### Classifying Equipment

```bash
# Classify equipment from any input format
python -m nexusml.classify_equipment path/to/input_file.csv --output path/to/output_file.json
```

## Command-Line Arguments

Most CLI tools support the following common arguments:

- `--help`: Show help message and exit
- `--verbose` or `-v`: Enable verbose output
- `--quiet` or `-q`: Suppress output
- `--log-file`: Specify a log file

Each tool also has specific arguments related to its functionality. Use the `--help` flag to see all available options for a specific tool.

## Configuration

CLI tools use the same configuration system as the rest of NexusML. Configuration can be provided through:

1. Command-line arguments
2. Environment variables
3. Configuration files

The precedence order is: command-line arguments > environment variables > configuration files.

## Error Handling

CLI tools provide meaningful error messages and exit codes:

- Exit code 0: Success
- Exit code 1: General error
- Exit code 2: Invalid arguments
- Exit code 3: File not found
- Exit code 4: Permission error

## Logging

CLI tools use the Python logging system to provide information about their execution. The log level can be controlled using the `--verbose` and `--quiet` flags.

## Next Steps

After reviewing the CLI documentation, you might want to:

1. Explore the [Examples](../examples/README.md) for practical usage examples
2. Check the [API Reference](../api_reference.md) for detailed information on the underlying classes and methods
3. Read the [Usage Guide](../usage_guide.md) for comprehensive usage documentation