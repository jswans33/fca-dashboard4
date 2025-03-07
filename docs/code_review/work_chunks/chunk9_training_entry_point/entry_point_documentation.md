# Training Pipeline Entry Point Documentation

## Overview

The updated training pipeline entry point (`train_model_pipeline_v2.py`)
provides a more modular, configurable, and testable approach to training
equipment classification models. It uses the pipeline orchestrator from Work
Chunk 8 while maintaining backward compatibility with the original
implementation through feature flags.

## Key Features

- **Orchestrator-based Pipeline**: Uses the pipeline orchestrator for improved
  modularity and testability
- **Feature Flags**: Maintains backward compatibility with the original
  implementation
- **Comprehensive Error Handling**: Provides detailed error messages and logging
- **Enhanced Configurability**: Offers more configuration options through
  command-line arguments
- **Improved Testability**: Facilitates unit and integration testing

## Usage

```bash
python nexusml/train_model_pipeline_v2.py --data-path PATH [options]
```

### Basic Example

```bash
python nexusml/train_model_pipeline_v2.py --data-path files/training-data/equipment_data.csv
```

### Advanced Example

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

## Architecture

The updated entry point follows a modular architecture:

1. **Command-Line Interface**: Parses and validates arguments using
   `training_args.py`
2. **Orchestrator Integration**: Uses the pipeline orchestrator to execute the
   training pipeline
3. **Feature Flags**: Provides backward compatibility through feature flags
4. **Error Handling**: Implements comprehensive error handling and logging

## Implementation Details

### Entry Point Flow

1. Parse command-line arguments
2. Set up logging
3. Load reference data
4. Validate training data
5. Train the model using either:
   - The orchestrator-based implementation (new)
   - The legacy implementation (backward compatibility)
6. Save the model
7. Generate visualizations (if requested)
8. Make a sample prediction

### Backward Compatibility

The entry point maintains backward compatibility through the `--legacy` flag,
which toggles between the new orchestrator-based implementation and the original
implementation. This allows existing scripts to continue working while new
scripts can take advantage of the improved architecture.

## Error Handling

The entry point implements comprehensive error handling:

- **Validation Errors**: Validates all inputs and provides detailed error
  messages
- **Runtime Errors**: Catches and logs runtime errors with stack traces
- **Component Errors**: Handles errors from individual pipeline components

## Logging

The entry point uses a structured logging approach:

- **Log Levels**: Supports DEBUG, INFO, WARNING, ERROR, and CRITICAL log levels
- **Log Format**: Includes timestamp, logger name, log level, and message
- **Log Output**: Logs to both console and file
- **Log Files**: Creates timestamped log files in the `logs` directory

## Integration Points

- **Pipeline Orchestrator**: Integrates with the pipeline orchestrator from Work
  Chunk 8
- **Configuration System**: Uses the configuration system from Work Chunk 1
- **Legacy Components**: Maintains compatibility with legacy components
