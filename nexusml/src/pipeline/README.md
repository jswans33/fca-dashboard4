# Pipeline Components

This directory contains the core pipeline components for the NexusML suite.

## Overview

The pipeline components are designed to be modular, testable, and configurable.
They follow SOLID principles and use dependency injection to manage
dependencies.

## Components

### Interfaces (`interfaces.py`)

Defines the interfaces for all pipeline components. Each interface follows the
Interface Segregation Principle (ISP) from SOLID, defining a minimal set of
methods that components must implement.

### Base Implementations (`base.py`)

Provides base implementations of the interfaces that can be extended for
specific use cases.

### Component Registry (`registry.py`)

Manages registration and retrieval of component implementations. It allows
registering multiple implementations of the same component type and setting a
default implementation for each type.

### Pipeline Factory (`factory.py`)

Creates pipeline components with proper dependencies. It uses the
ComponentRegistry to look up component implementations and the DIContainer to
resolve dependencies.

### Adapters (`adapters/`)

Contains adapter implementations that provide backward compatibility with
existing code.

### Components (`components/`)

Contains concrete implementations of the pipeline components.

## Usage

### Component Registry

```python
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.interfaces import DataLoader
from nexusml.core.pipeline.components.data import CSVDataLoader, ExcelDataLoader

# Create a registry
registry = ComponentRegistry()

# Register components
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataLoader, "excel", ExcelDataLoader)

# Set a default implementation
registry.set_default_implementation(DataLoader, "csv")

# Get a specific implementation
csv_loader_class = registry.get_implementation(DataLoader, "csv")

# Get the default implementation
default_loader_class = registry.get_default_implementation(DataLoader)
```

### Pipeline Factory

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry

# Create a registry and container
registry = ComponentRegistry()
container = DIContainer()

# Register components (as shown above)
# ...

# Create a factory
factory = PipelineFactory(registry, container)

# Create components
data_loader = factory.create_data_loader()
preprocessor = factory.create_data_preprocessor()
feature_engineer = factory.create_feature_engineer()
model_builder = factory.create_model_builder()

# Create a component with a specific implementation
excel_loader = factory.create_data_loader("excel")

# Create a component with additional configuration
data_loader = factory.create_data_loader(config={"file_path": "data.csv"})
```

## Integration with Dependency Injection Container

The `PipelineFactory` class integrates with the Dependency Injection Container
(DI Container) to resolve dependencies for components. When creating a
component, the factory:

1. Looks up the component implementation in the registry
2. Analyzes the constructor parameters
3. Resolves dependencies from the DI container
4. Creates the component with resolved dependencies

This integration allows components to be created with their dependencies
automatically resolved, making the code more modular and testable.

Example of integration with the DI container:

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor

# Create a registry and container
registry = ComponentRegistry()
container = DIContainer()

# Register a component that depends on DataLoader
class MyPreprocessor(DataPreprocessor):
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def preprocess(self, data, **kwargs):
        # Implementation...
        return data

    def verify_required_columns(self, data):
        # Implementation...
        return data

# Register components
registry.register(DataLoader, "default", MyDataLoader)
registry.register(DataPreprocessor, "default", MyPreprocessor)

# Set default implementations
registry.set_default_implementation(DataLoader, "default")
registry.set_default_implementation(DataPreprocessor, "default")

# Create a factory
factory = PipelineFactory(registry, container)

# Create a preprocessor - the factory will automatically resolve the DataLoader dependency
preprocessor = factory.create_data_preprocessor()
```

## Customization Mechanisms

The factory provides several customization mechanisms:

### 1. Component Selection

You can select specific component implementations when creating components:

```python
# Create a component with a specific implementation
excel_loader = factory.create_data_loader("excel")
```

### 2. Configuration Parameters

You can pass configuration parameters to components:

```python
# Create a component with additional configuration
data_loader = factory.create_data_loader(config={"file_path": "data.csv"})
```

### 3. Custom Component Creation

You can create custom components using the generic `create` method:

```python
# Create a custom component
custom_component = factory.create(CustomComponentType, "implementation_name", **kwargs)
```

### 4. Dependency Override

You can override dependencies by providing them directly:

```python
# Create a component with a specific dependency
preprocessor = factory.create_data_preprocessor(data_loader=custom_loader)
```

## Complete Example

Here's a complete example of using the factory to create a pipeline:

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.interfaces import (
    DataLoader, DataPreprocessor, FeatureEngineer,
    ModelBuilder, ModelTrainer, ModelEvaluator
)
from nexusml.core.pipeline.components.data import CSVDataLoader
from nexusml.core.pipeline.components.preprocessing import StandardPreprocessor
from nexusml.core.pipeline.components.feature import TextFeatureEngineer
from nexusml.core.pipeline.components.model import RandomForestModelBuilder
from nexusml.core.pipeline.components.training import StandardModelTrainer
from nexusml.core.pipeline.components.evaluation import StandardModelEvaluator

# Create a registry and container
registry = ComponentRegistry()
container = DIContainer()

# Register components
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataPreprocessor, "standard", StandardPreprocessor)
registry.register(FeatureEngineer, "text", TextFeatureEngineer)
registry.register(ModelBuilder, "random_forest", RandomForestModelBuilder)
registry.register(ModelTrainer, "standard", StandardModelTrainer)
registry.register(ModelEvaluator, "standard", StandardModelEvaluator)

# Set default implementations
registry.set_default_implementation(DataLoader, "csv")
registry.set_default_implementation(DataPreprocessor, "standard")
registry.set_default_implementation(FeatureEngineer, "text")
registry.set_default_implementation(ModelBuilder, "random_forest")
registry.set_default_implementation(ModelTrainer, "standard")
registry.set_default_implementation(ModelEvaluator, "standard")

# Create a factory
factory = PipelineFactory(registry, container)

# Create pipeline components
data_loader = factory.create_data_loader(config={"file_path": "data.csv"})
preprocessor = factory.create_data_preprocessor()
feature_engineer = factory.create_feature_engineer()
model_builder = factory.create_model_builder(config={"n_estimators": 100})
model_trainer = factory.create_model_trainer()
model_evaluator = factory.create_model_evaluator()

# Use the components to build a pipeline
data = data_loader.load_data()
preprocessed_data = preprocessor.preprocess(data)
features = feature_engineer.engineer_features(preprocessed_data)
model = model_builder.build_model()
trained_model = model_trainer.train(model, features, preprocessed_data["target"])
evaluation = model_evaluator.evaluate(trained_model, features, preprocessed_data["target"])

print(f"Model evaluation: {evaluation}")
```
