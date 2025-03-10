# NexusML Utility Scripts Documentation

This directory contains documentation for the utility scripts of the NexusML package. These scripts provide standalone functionality for specific tasks.

## Overview

The utility scripts in NexusML are standalone Python scripts or shell scripts that perform specific tasks related to model management, data processing, and system maintenance. These scripts are designed to be run from the command line or integrated into workflows.

## Utility Scripts

- [model_card_tool.py](model_card_tool.md): Tool for generating and managing model cards
- [train_model.sh](train_model.md): Shell script for training models with standard configurations

## Model Card Tool

The Model Card Tool is a utility for generating and managing model cards. Model cards are structured documents that provide information about machine learning models, including:

- Model details (name, version, type)
- Training data information
- Performance metrics
- Usage guidelines
- Limitations and biases
- Ethical considerations

The tool supports:

- Creating new model cards
- Updating existing model cards
- Validating model cards against a schema
- Exporting model cards to different formats (JSON, Markdown, HTML)

## Train Model Script

The Train Model Script is a shell script that simplifies the process of training models with standard configurations. It provides:

- Default configurations for common use cases
- Environment setup
- Logging and monitoring
- Error handling and reporting

## Usage Examples

### Model Card Tool

```bash
# Generate a model card for a trained model
python -m nexusml.scripts.model_card_tool generate --model-path outputs/models/equipment_classifier.pkl --output-path outputs/model_cards/equipment_classifier.json

# Validate a model card against the schema
python -m nexusml.scripts.model_card_tool validate --model-card-path outputs/model_cards/equipment_classifier.json

# Export a model card to Markdown format
python -m nexusml.scripts.model_card_tool export --model-card-path outputs/model_cards/equipment_classifier.json --format markdown --output-path outputs/model_cards/equipment_classifier.md
```

### Train Model Script

```bash
# Train a model with default configurations
./nexusml/scripts/train_model.sh --data-path data/training_data.csv

# Train a model with custom configurations
./nexusml/scripts/train_model.sh --data-path data/training_data.csv --config-path config/custom_config.yml --output-dir outputs/custom_models
```

## Integration with Other Components

The utility scripts integrate with other components in NexusML:

- They use the core modules for functionality
- They leverage the configuration system for flexible configuration
- They produce outputs that can be used by other components
- They can be called from other scripts or workflows

## Next Steps

After reviewing the utility scripts documentation, you might want to:

1. Explore the [Examples](../examples/README.md) for practical usage examples
2. Check the [CLI Documentation](../cli/README.md) for information on command-line tools
3. Read the [Core Modules Documentation](../modules/README.md) to understand the underlying functionality