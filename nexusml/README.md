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

### Advanced Usage

See the examples directory for more detailed usage examples:

- `simple_example.py`: Basic usage without visualizations
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
