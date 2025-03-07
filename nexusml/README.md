# NexusML

A Python machine learning package for equipment classification.

## Overview

NexusML is a Python package designed for classifying mechanical equipment based
on textual descriptions and metadata. It uses machine learning techniques to
categorize equipment into standardized classification systems like MasterFormat
and OmniClass.

## Features

- Data loading and preprocessing
- Feature engineering from textual descriptions
- Model training using random forest classifiers
- Model evaluation and validation
- Prediction on new equipment data
- Visualization of model performance

## Installation

```bash
pip install -e .
```

## Usage

### Training a Model

NexusML provides two ways to train a model:

#### 1. Using the Orchestrator-Based Pipeline (Recommended)

```bash
python nexusml/train_model_pipeline_v2.py --data-path PATH [options]
```

Example:

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

#### 2. Using the Legacy Pipeline

```bash
python nexusml/train_model_pipeline_v2.py --data-path PATH --legacy [options]
```

or

```bash
python nexusml/train_model_pipeline.py --data-path PATH [options]
```

### Making Predictions

```bash
python nexusml/predict.py --model-path PATH --data-path PATH [options]
```

Example:

```bash
python nexusml/predict.py \
    --model-path outputs/models/equipment_classifier_latest.pkl \
    --data-path files/test-data/equipment_data.csv \
    --output-path outputs/predictions.csv
```

## Command-Line Arguments

### Training Arguments

| Argument              | Type   | Default                | Description                                           |
| --------------------- | ------ | ---------------------- | ----------------------------------------------------- |
| `--data-path`         | string | (required)             | Path to the training data CSV file                    |
| `--feature-config`    | string | None                   | Path to the feature configuration YAML file           |
| `--reference-config`  | string | None                   | Path to the reference configuration YAML file         |
| `--test-size`         | float  | 0.3                    | Proportion of data to use for testing (0.0 to 1.0)    |
| `--random-state`      | int    | 42                     | Random state for reproducibility                      |
| `--sampling-strategy` | string | "direct"               | Sampling strategy for handling class imbalance        |
| `--optimize`          | flag   | False                  | Perform hyperparameter optimization                   |
| `--output-dir`        | string | "outputs/models"       | Directory to save the trained model and results       |
| `--model-name`        | string | "equipment_classifier" | Base name for the saved model                         |
| `--log-level`         | string | "INFO"                 | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--visualize`         | flag   | False                  | Generate visualizations of model performance          |
| `--legacy`            | flag   | False                  | Use legacy implementation instead of orchestrator     |

### Prediction Arguments

| Argument        | Type   | Default                   | Description                                           |
| --------------- | ------ | ------------------------- | ----------------------------------------------------- |
| `--model-path`  | string | (required)                | Path to the trained model file                        |
| `--data-path`   | string | (required)                | Path to the data file for prediction                  |
| `--output-path` | string | "outputs/predictions.csv" | Path to save the predictions                          |
| `--log-level`   | string | "INFO"                    | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

## Architecture

NexusML follows a modular architecture:

```
nexusml/
├── core/             # Core functionality and interfaces
│   ├── cli/          # Command-line interfaces
│   ├── di/           # Dependency injection
│   ├── pipeline/     # Pipeline components
│   └── reference/    # Reference data management
├── data/             # Data handling
├── examples/         # Example scripts
├── tests/            # Test suite
└── utils/            # Utility functions
```

## Development

### Testing

Run all tests:

```bash
pytest
```

Run tests with coverage:

```bash
pytest --cov=nexusml
```

### Code Quality

Format code:

```bash
black nexusml tests
```

Check types:

```bash
mypy nexusml
```

## Pipeline Factory

NexusML uses a factory pattern to create pipeline components with their proper
dependencies. The factory uses a component registry to look up implementations
and a dependency injection container to resolve dependencies.

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry

# Create a registry and container
registry = ComponentRegistry()
container = DIContainer()

# Register components
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataPreprocessor, "standard", StandardPreprocessor)
registry.register(FeatureEngineer, "text", TextFeatureEngineer)
registry.register(ModelBuilder, "random_forest", RandomForestModelBuilder)

# Set default implementations
registry.set_default_implementation(DataLoader, "csv")
registry.set_default_implementation(DataPreprocessor, "standard")
registry.set_default_implementation(FeatureEngineer, "text")
registry.set_default_implementation(ModelBuilder, "random_forest")

# Create a factory
factory = PipelineFactory(registry, container)

# Create components
data_loader = factory.create_data_loader()
preprocessor = factory.create_data_preprocessor()
feature_engineer = factory.create_feature_engineer()
model_builder = factory.create_model_builder()
```

The factory provides several advantages:

- **Centralized Component Creation**: All components are created through a
  single factory, making it easy to manage and configure.
- **Dependency Injection**: Components with dependencies are automatically wired
  together.
- **Customization**: You can select specific implementations or provide custom
  configuration.
- **Testability**: Components can be easily mocked or replaced for testing.

For more details, see the
[Pipeline Components README](nexusml/core/pipeline/README.md).

## Configuration

NexusML uses a unified configuration system that centralizes all settings in a
single file. The configuration system provides:

- Validation of configuration values using Pydantic
- Default values for all settings
- Loading from YAML files or environment variables
- Consistent access through a singleton provider

### Basic Configuration Usage

```python
from nexusml.core.config.provider import ConfigurationProvider

# Get the configuration
config_provider = ConfigurationProvider()
config = config_provider.config

# Access configuration values
feature_config = config.feature_engineering
data_config = config.data
```

### Custom Configuration

You can specify a custom configuration file using the `NEXUSML_CONFIG`
environment variable:

```bash
export NEXUSML_CONFIG=/path/to/your/config.yml
```

### Migration from Legacy Configuration

To migrate from legacy configuration files to the new unified format:

```python
from nexusml.core.config.migration import migrate_from_default_paths

# Migrate configurations and save to the default path
config = migrate_from_default_paths()
```

### Future Configuration Cleanup

The following legacy configuration files are maintained for backward
compatibility and are planned for removal in future work chunks:

- `classification_config.yml` - Will be replaced by the unified configuration
- `data_config.yml` - Will be replaced by the unified configuration
- `feature_config.yml` - Will be replaced by the unified configuration
- `reference_config.yml` - Will be replaced by the unified configuration
- `eav/equipment_attributes.json` - Will be replaced by the unified
  configuration
- `mappings/masterformat_primary.json` - Will be replaced by the unified
  configuration
- `mappings/masterformat_equipment.json` - Will be replaced by the unified
  configuration

Once all code is updated to use the new unified configuration system, these
files will be removed.
