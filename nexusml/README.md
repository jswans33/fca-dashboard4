# NexusML

A modern machine learning classification engine for equipment classification.

## Overview

NexusML is a standalone Python package that provides machine learning
capabilities for classifying equipment based on descriptions and other features.
It was extracted from the FCA Dashboard project to enable independent
development and reuse.

## Features

- Data preprocessing and cleaning
- Feature engineering for text data
- Hierarchical classification models
- Model evaluation and validation
- Visualization of results
- Easy-to-use API for predictions
- OmniClass data extraction and description generation
- Unified configuration system with validation

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

## Installation

### From Source

```bash
# Install with pip
pip install -e .

# Or install with uv (recommended)
uv pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

Note: The package is named 'core' in the current monorepo structure, so imports
should use:

```python
from core.model import ...
```

rather than:

```python
from nexusml.core.model import ...
```

## Usage

### Basic Example

```python
from core.model import train_enhanced_model, predict_with_enhanced_model

# Train a model
model, df = train_enhanced_model("path/to/training_data.csv")

# Make a prediction
description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
service_life = 20.0  # Example service life in years

prediction = predict_with_enhanced_model(model, description, service_life)
print(prediction)
```

### OmniClass Generator Usage

```python
from nexusml import extract_omniclass_data, generate_descriptions

# Extract OmniClass data from Excel files
df = extract_omniclass_data(
    input_dir="files/omniclass_tables",
    output_file="nexusml/ingest/generator/data/omniclass.csv",
    file_pattern="*.xlsx"
)

# Generate descriptions for OmniClass codes
result_df = generate_descriptions(
    input_file="nexusml/ingest/generator/data/omniclass.csv",
    output_file="nexusml/ingest/generator/data/omniclass_with_descriptions.csv",
    batch_size=50,
    description_column="Description"
)
```

### Advanced Usage

See the examples directory for more detailed usage examples:

- `simple_example.py`: Basic usage without visualizations
- `advanced_example.py`: Complete workflow with visualizations
- `omniclass_generator_example.py`: Example of using the OmniClass generator
- `advanced_example.py`: Complete workflow with visualizations

## Development

### Setup Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nexusml
```

## License

MIT
