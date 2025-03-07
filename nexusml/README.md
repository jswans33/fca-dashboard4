# NexusML

NexusML is a Python machine learning package for equipment classification. It
uses machine learning techniques to categorize equipment into standardized
classification systems like MasterFormat and OmniClass based on textual
descriptions and metadata.

## Features

- **Data Loading and Preprocessing**: Load data from various sources and
  preprocess it for machine learning
- **Feature Engineering**: Transform raw data into features suitable for machine
  learning
- **Model Training**: Train machine learning models for equipment classification
- **Model Evaluation**: Evaluate model performance with various metrics
- **Prediction**: Make predictions on new equipment data
- **Configuration**: Centralized configuration system for all settings
- **Extensibility**: Easy to extend with custom components

## Installation

```bash
pip install nexusml
```

## Quick Start

### Training a Model

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create the pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)

# Train a model
model, metrics = orchestrator.train_model(
    data_path="path/to/training_data.csv",
    test_size=0.3,
    random_state=42,
    optimize_hyperparameters=True,
    output_dir="outputs/models",
    model_name="equipment_classifier",
)

# Print metrics
print("Model training completed successfully")
print("Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

### Making Predictions

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create the pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)

# Load a trained model
model = orchestrator.load_model("outputs/models/equipment_classifier.pkl")

# Make predictions
predictions = orchestrator.predict(
    model=model,
    data_path="path/to/prediction_data.csv",
    output_path="outputs/predictions.csv",
)

# Print predictions
print("Predictions completed successfully")
print("Sample predictions:")
print(predictions.head())
```

## Architecture

NexusML follows a modular architecture with clear interfaces, dependency
injection, and a factory pattern. The key components are:

### Configuration System

The configuration system centralizes all settings in a single file, provides
validation through Pydantic models, supports loading from YAML files or
environment variables, and ensures consistent access through a singleton
provider.

### Pipeline Components

The pipeline components are responsible for the various stages of the machine
learning pipeline, from data loading to prediction. Each component has a clear
interface and is responsible for a specific part of the pipeline.

- **Data Loader**: Loads data from various sources
- **Data Preprocessor**: Cleans and prepares data
- **Feature Engineer**: Transforms raw data into features
- **Model Builder**: Creates and configures models
- **Model Trainer**: Trains models
- **Model Evaluator**: Evaluates models
- **Model Serializer**: Saves and loads models
- **Predictor**: Makes predictions

### Pipeline Management

The pipeline management components are responsible for creating, configuring,
and orchestrating the pipeline components.

- **Component Registry**: Registers component implementations and their default
  implementations
- **Pipeline Factory**: Creates pipeline components with proper dependencies
- **Pipeline Orchestrator**: Coordinates the execution of the pipeline
- **Pipeline Context**: Stores state and data during pipeline execution

### Dependency Injection

The dependency injection system provides a way to manage component dependencies,
making the system more testable and maintainable. It follows the Dependency
Inversion Principle from SOLID, allowing high-level modules to depend on
abstractions rather than concrete implementations.

## Documentation

For more detailed documentation, see the following:

- [Architecture Overview](docs/architecture/overview.md)
- [Configuration System](docs/architecture/configuration.md)
- [Pipeline Architecture](docs/architecture/pipeline.md)
- [Dependency Injection](docs/architecture/dependency_injection.md)
- [Migration Guide](docs/migration/overview.md)
- [Examples](docs/examples/)

## Examples

The `docs/examples/` directory contains example scripts demonstrating various
aspects of NexusML:

- [Basic Usage](docs/examples/basic_usage.py): Basic usage of NexusML for
  training and prediction
- [Custom Components](docs/examples/custom_components.py): Creating custom
  components for NexusML
- [Configuration](docs/examples/configuration.py): Using the configuration
  system
- [Dependency Injection](docs/examples/dependency_injection.py): Using the
  dependency injection system

You can run these examples using the following Makefile targets:

```bash
# Run all examples
make nexusml-examples

# Run individual examples
make nexusml-example-basic     # Basic usage example
make nexusml-example-custom    # Custom components example
make nexusml-example-config    # Configuration example
make nexusml-example-di        # Dependency injection example
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for
details.
