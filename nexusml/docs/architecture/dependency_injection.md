# Dependency Injection

## Overview

NexusML uses a dependency injection (DI) system to manage component dependencies, making the system more testable and maintainable. The DI system follows the Dependency Inversion Principle from SOLID, allowing high-level modules to depend on abstractions rather than concrete implementations.

## Diagram

The following diagram illustrates the dependency injection system:

- [Dependency Injection System](../../diagrams/nexusml/dependency_injection.puml) - Components and relationships of the DI system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### DIContainer

Central container that manages instances and factories for dependencies.

```python
from nexusml.core.di.container import DIContainer

# Create container
container = DIContainer()

# Register instance
container.register_instance(DataLoader, csv_loader)

# Register factory
container.register_factory(FeatureEngineer, lambda: GenericFeatureEngineer())

# Get instance
data_loader = container.get(DataLoader)

# Check if container has implementation
if container.has(DataLoader):
    data_loader = container.get(DataLoader)
```

### Inject Annotation

Annotation for marking parameters to be injected.

```python
from nexusml.core.di.container import Inject

class Service:
    def __init__(self, data_loader: DataLoader = Inject(DataLoader)):
        self._data_loader = data_loader
```

### inject Decorator

Decorator for injecting dependencies into function parameters.

```python
from nexusml.core.di.decorators import inject
from nexusml.core.di.container import Inject

@inject
def process_data(data_path: str, 
                data_loader: DataLoader = Inject(DataLoader)):
    return data_loader.load_data(data_path)
```

### ComponentRegistry

Registry for mapping interfaces to implementations.

```python
from nexusml.core.pipeline.registry import ComponentRegistry

# Create registry
registry = ComponentRegistry()

# Register implementation
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataLoader, "excel", ExcelDataLoader)

# Set default implementation
registry.set_default_implementation(DataLoader, "csv")

# Get implementation class
loader_class = registry.get_implementation(DataLoader)  # Returns CSVDataLoader class
excel_loader_class = registry.get_implementation(DataLoader, "excel")  # Returns ExcelDataLoader class
```

### PipelineFactory

Factory that creates component instances using the registry and container.

```python
from nexusml.core.pipeline.factory import PipelineFactory

# Create factory
factory = PipelineFactory(registry, container)

# Create component
data_loader = factory.create(DataLoader)  # Returns CSVDataLoader instance
excel_loader = factory.create(DataLoader, "excel")  # Returns ExcelDataLoader instance

# Create with constructor parameters
custom_loader = factory.create(DataLoader, "csv", encoding="latin-1", delimiter=";")
```

## Dependency Injection Patterns

### Constructor Injection

Dependencies are injected through the constructor.

```python
class Service:
    def __init__(self, data_loader: DataLoader, feature_engineer: FeatureEngineer):
        self._data_loader = data_loader
        self._feature_engineer = feature_engineer
        
    def process_data(self, data_path: str):
        data = self._data_loader.load_data(data_path)
        return self._feature_engineer.engineer_features(data)
```

### Parameter Injection

Dependencies are injected into function parameters.

```python
@inject
def process_data(data_path: str, 
                data_loader: DataLoader = Inject(DataLoader),
                feature_engineer: FeatureEngineer = Inject(FeatureEngineer)):
    data = data_loader.load_data(data_path)
    return feature_engineer.engineer_features(data)
```

### Property Injection

Dependencies are injected into properties after object creation.

```python
class Service:
    def __init__(self):
        self._data_loader = None
        self._feature_engineer = None
        
    @property
    def data_loader(self):
        return self._data_loader
        
    @data_loader.setter
    def data_loader(self, value: DataLoader):
        self._data_loader = value
        
    @property
    def feature_engineer(self):
        return self._feature_engineer
        
    @feature_engineer.setter
    def feature_engineer(self, value: FeatureEngineer):
        self._feature_engineer = value
```

## Using with Pipeline Components

Pipeline components typically use constructor injection:

```python
class CustomFeatureEngineer(FeatureEngineer):
    def __init__(self, data_preprocessor: DataPreprocessor):
        self._preprocessor = data_preprocessor
        
    def engineer_features(self, data: pd.DataFrame, **kwargs):
        # Preprocess data first
        data = self._preprocessor.preprocess(data)
        # Then engineer features
        # ...
        return data
```

The PipelineFactory handles dependency resolution:

```python
# Register components
registry.register(DataPreprocessor, "standard", StandardPreprocessor)
registry.register(FeatureEngineer, "custom", CustomFeatureEngineer)

# Create factory
factory = PipelineFactory(registry, container)

# Create component with dependencies automatically resolved
feature_engineer = factory.create(FeatureEngineer, "custom")
# The factory will:
# 1. See that CustomFeatureEngineer needs a DataPreprocessor
# 2. Create a StandardPreprocessor instance
# 3. Inject it into the CustomFeatureEngineer constructor
```

## Testing with Dependency Injection

DI makes testing easier by allowing dependencies to be mocked:

```python
# Create mock
mock_data_loader = MagicMock(spec=DataLoader)
mock_data_loader.load_data.return_value = pd.DataFrame({"test": [1, 2, 3]})

# Register mock
container.register_instance(DataLoader, mock_data_loader)

# Test function with injected mock
result = process_data("dummy/path")

# Verify mock was called
mock_data_loader.load_data.assert_called_once_with("dummy/path")
```

## Advanced Usage

### Named Dependencies

Register and inject named dependencies:

```python
# Register named implementations
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataLoader, "excel", ExcelDataLoader)

# Inject specific implementation
@inject
def process_excel(file_path: str, 
                 excel_loader: DataLoader = Inject(DataLoader, "excel")):
    return excel_loader.load_data(file_path)
```

### Factories

Use factories for complex instance creation:

```python
def create_feature_engineer():
    # Complex creation logic
    config = ConfigProvider.get_config().feature_engineering
    return GenericFeatureEngineer(
        text_columns=config.text_columns,
        numerical_columns=config.numerical_columns,
        categorical_columns=config.categorical_columns
    )

# Register factory
container.register_factory(FeatureEngineer, create_feature_engineer)
```

### Scoped Instances

Create a new container for scoped instances:

```python
# Create request-scoped container
request_container = DIContainer()

# Register request-specific instances
request_container.register_instance(RequestContext, context)

# Use for request processing
process_request(request, request_container)