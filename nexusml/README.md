# NexusML

NexusML is a Python machine learning package for equipment classification. It
uses machine learning techniques to categorize equipment into standardized
classification systems like MasterFormat and OmniClass based on textual
descriptions and metadata.

## Features

- **Data Loading and Preprocessing**: Load data from various sources and
  preprocess it for machine learning
- **Feature Engineering**: Transform raw data into features suitable for machine
  learning using a configurable pipeline
- **Model Training**: Train machine learning models for equipment classification
  with support for hyperparameter optimization
- **Model Evaluation**: Evaluate model performance with comprehensive metrics
  and analysis tools
- **Prediction**: Make predictions on new equipment data with both batch and
  single-item support
- **Configuration**: Centralized configuration system with validation and
  environment variable support
- **Dependency Injection**: Flexible component system with dependency management
- **Pipeline Architecture**: Modular pipeline system for customizable ML
  workflows
- **Model Cards**: Generate standardized model cards for model documentation and
  governance

## Directory Structure

- `nexusml/config/` - Configuration files
- `nexusml/data/` - Data files
  - `nexusml/data/raw/` - Raw data files
  - `nexusml/data/processed/` - Processed data files
  - `nexusml/data/reference/` - Reference data files
- `nexusml/examples/` - Example scripts
- `nexusml/models/` - Saved models
  - `nexusml/models/baseline/` - Baseline models
  - `nexusml/models/production/` - Production models
- `nexusml/notebooks/` - Jupyter notebooks
- `nexusml/outputs/` - Output files
  - `nexusml/outputs/predictions/` - Prediction outputs
  - `nexusml/outputs/evaluation/` - Evaluation outputs
  - `nexusml/outputs/model_cards/` - Model cards
- `nexusml/scripts/` - Utility scripts
- `nexusml/src/` - Source code
  - `nexusml/src/data/` - Data loading and processing
  - `nexusml/src/features/` - Feature engineering
  - `nexusml/src/models/` - Model building and training
  - `nexusml/src/pipeline/` - Pipeline components
  - `nexusml/src/utils/` - Utility functions
- `nexusml/tests/` - Test files

## Installation

For basic installation:

```bash
pip install nexusml
```

For detailed installation instructions, system requirements, troubleshooting,
and configuration guidance, see the
[Installation Guide](docs/installation_guide.md).

### Basic Installation Options

```bash
# Standard installation
pip install nexusml

# Development installation
git clone https://github.com/your-org/nexusml.git
cd nexusml
pip install -e ".[dev]"

# With AI features
pip install "nexusml[ai]"
```

## Quick Start

### Training a Model

```python
from nexusml.src.pipeline.orchestrator import PipelineOrchestrator
from nexusml.src.pipeline.factory import PipelineFactory
from nexusml.src.pipeline.registry import ComponentRegistry
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.utils.di_container import DIContainer

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
from nexusml.src.pipeline.orchestrator import PipelineOrchestrator
from nexusml.src.pipeline.factory import PipelineFactory
from nexusml.src.pipeline.registry import ComponentRegistry
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.utils.di_container import DIContainer

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

### Using Command-Line Tools

NexusML provides several command-line tools for common tasks:

```bash
# Train a model
python -m nexusml.train_model_pipeline_v2 --data-path path/to/training_data.csv --optimize

# Make predictions
python -m nexusml.predict_v2 --model-path outputs/models/equipment_classifier_latest.pkl --input-file path/to/prediction_data.csv

# Classify equipment from any input format
python -m nexusml.classify_equipment path/to/input_file.csv --output path/to/output_file.json
```

## Architecture

NexusML follows a modular architecture with clear interfaces, dependency
injection, and a factory pattern. The key components are:

### Configuration System

The configuration system centralizes all settings in a single file, provides
validation through Pydantic models, supports loading from YAML files or
environment variables, and ensures consistent access through a singleton
provider.

```python
from nexusml.src.utils.config_provider import ConfigProvider

# Get configuration
config = ConfigProvider.get_config()
feature_config = config.feature_engineering
```

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

```python
from nexusml.src.utils.di_container import DIContainer
from nexusml.src.utils.di_decorators import inject

@inject
def my_function(data_loader=Inject(DataLoader)):
    # Use data_loader
    data = data_loader.load_data("path/to/data.csv")
    return data
```

## Model Verification and Testing

NexusML includes a comprehensive verification system to ensure the reliability
and correctness of the model training, evaluation, and prediction pipelines.
This verification system is a critical part of the refactoring work to establish
a production-ready model card system.

### Verification Components

The verification system tests the following components:

1. **Component Resolution**: Verifies that all required components can be
   resolved from the Dependency Injection container, ensuring proper
   registration and dependency management.

2. **Pipeline Factory**: Tests the creation of different pipeline types
   (training, prediction, evaluation) through the factory pattern, confirming
   the factory correctly assembles pipeline components.

3. **Feature Engineering**: Validates the feature engineering components by
   applying transformations to sample data and verifying the expected output
   columns are created.

4. **Model Building**: Ensures model builders can create valid models with the
   expected interfaces and behaviors.

5. **Model Card System**: Verifies the model card generation, saving, and
   loading functionality, which is essential for model governance and
   documentation.

6. **End-to-End Functionality**: Tests the complete workflow from data loading
   to model saving, ensuring all components work together correctly.

7. **Prediction Pipeline**: Validates the prediction pipeline by loading a
   trained model and making predictions on sample data.

### Current Status

As of March 2025, the verification system is fully operational and passing all
tests. Recent improvements include:

- Fixed issues with the StandardDataLoader class accessibility
- Enhanced the prediction pipeline to handle different model types
- Improved feature handling based on model's feature*names_in* attribute
- Added better error handling and logging throughout the pipeline
- Implemented comprehensive model card generation and validation

### Running Verification Tests

You can run the verification tests using the following command:

```bash
python -m nexusml.tests.verification_script
```

This script will execute all verification tests and provide detailed output on
the status of each component. A successful verification will show "✅ All
verification tests passed!" at the end of the output.

### Integration with Model Card System

The verification system is a key enabler for the production model card system,
which is the ultimate goal of the NexusML refactoring work. The model card
system provides:

- Standardized model training with proper validation and evaluation
- Consistent model metadata and documentation
- Reproducible model building and deployment processes
- Better tracking and versioning of models in production
- Improved model governance and compliance

By ensuring all components work correctly through verification, we can
confidently deploy models to production with proper documentation and
governance.

## Documentation

For more detailed documentation, see the following:

- [Installation Guide](docs/installation_guide.md)
- [Dependencies](docs/dependencies.md)
- [Usage Guide](docs/usage_guide.md)
- [API Reference](docs/api_reference.md)
- [Offline Resources](docs/offline_resources.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Configuration System](docs/architecture/configuration.md)
- [Pipeline Architecture](docs/architecture/pipeline.md)
- [Dependency Injection](docs/architecture/dependency_injection.md)
- [Feature Engineering](docs/architecture/feature_engineering.md)
- [Model Building](docs/architecture/model_building.md)
- [Migration Guide](docs/migration/overview.md)
- [Examples](docs/examples/)

## Examples

The `examples/` directory contains example scripts demonstrating various aspects
of NexusML:

- [Simple Example](examples/simple_example.py): Basic usage of NexusML
- [Advanced Example](examples/advanced_example.py): Advanced usage with custom
  components
- [Data Loading](examples/data_loader_example.py): Loading data from various
  sources
- [Feature Engineering](examples/feature_engineering_example.py): Custom feature
  engineering
- [Model Building](examples/model_building_example.py): Building and configuring
  models
- [Pipeline Orchestration](examples/pipeline_orchestrator_example.py): Using the
  pipeline orchestrator
- [OmniClass Generation](examples/omniclass_generator_example.py): Generating
  OmniClass descriptions

You can run these examples directly:

```bash
# Run a simple example
python -m nexusml.examples.simple_example

# Run an advanced example
python -m nexusml.examples.advanced_example
```

## Command-Line Tools

NexusML includes several command-line tools for common tasks:

- **train_model_pipeline.py**: Train a model using the original pipeline
- **train_model_pipeline_v2.py**: Train a model using the new pipeline
  architecture
- **predict.py**: Make predictions using the original pipeline
- **predict_v2.py**: Make predictions using the new pipeline architecture
- **classify_equipment.py**: Classify equipment from any input format
- **test_reference_validation.py**: Validate reference data

For detailed usage information, run any tool with the `--help` flag:

```bash
python -m nexusml.train_model_pipeline_v2 --help
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Clone the repository
2. Install development dependencies: `pip install -e ".[dev]"`
3. Run tests: `pytest`
4. Format code: `black .`
5. Check types: `mypy .`

## License

This project is licensed under the MIT License - see the LICENSE file for
details.
