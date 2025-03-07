# NexusML Migration Overview

## Introduction

This document provides an overview of migrating from the old NexusML
architecture to the new refactored architecture. It explains the key differences
between the old and new architectures, the migration strategy, backward
compatibility mechanisms, and provides a migration checklist.

The refactoring of NexusML was undertaken to improve the system's testability,
configurability, and extensibility while maintaining backward compatibility.
This document will help you understand the changes and how to migrate your
existing code to the new architecture.

## Key Differences Between Old and New Architectures

### Configuration System

**Old Architecture**:

- Configuration scattered across multiple files
- Inconsistent loading mechanisms
- No validation of configuration values
- No centralized access point

**New Architecture**:

- Unified configuration system with Pydantic models
- Validation of configuration values
- Default values for all settings
- Loading from YAML files or environment variables
- Consistent access through a singleton provider

### Component Structure

**Old Architecture**:

- Tight coupling between components
- Inconsistent interfaces
- Dependencies created internally
- Difficult to test and extend

**New Architecture**:

- Clear interfaces for all components
- Loose coupling through dependency injection
- Components depend on abstractions, not concrete implementations
- Easy to test and extend

### Pipeline Execution

**Old Architecture**:

- Pipeline execution scattered across multiple files
- Inconsistent error handling
- No centralized orchestration
- Difficult to monitor and debug

**New Architecture**:

- Centralized pipeline orchestration
- Consistent error handling
- Pipeline context for sharing state
- Easy to monitor and debug

### Entry Points

**Old Architecture**:

- Monolithic entry points
- Hard-coded dependencies
- Limited configurability
- Difficult to extend

**New Architecture**:

- Modular entry points
- Dependency injection
- Highly configurable
- Easy to extend

## Migration Strategy

The migration strategy is designed to be incremental, allowing you to migrate
your code gradually while maintaining functionality. The strategy consists of
the following steps:

1. **Adapter Pattern**: Use adapters to maintain backward compatibility while
   introducing new components
2. **Feature Flags**: Use feature flags to toggle between old and new code paths
   for testing
3. **Incremental Integration**: Integrate the new architecture incrementally,
   with each integration step maintaining backward compatibility
4. **Comprehensive Testing**: Test each migration step to ensure functionality
   is maintained

### Adapter Pattern

The adapter pattern is used to wrap old components in new interfaces, allowing
them to be used in the new architecture. This approach enables gradual migration
without breaking existing functionality.

```python
# Old component
class OldDataLoader:
    def load(self, path):
        # Old implementation
        pass

# Adapter for old component
class OldDataLoaderAdapter(DataLoader):
    def __init__(self, old_loader):
        self.old_loader = old_loader

    def load_data(self, data_path=None, **kwargs):
        # Adapt old interface to new interface
        return self.old_loader.load(data_path)

    def get_config(self):
        # Provide configuration for old component
        return {}
```

### Feature Flags

Feature flags are used to toggle between old and new code paths, allowing you to
test the new architecture without affecting existing functionality.

```python
# Feature flag for using new architecture
USE_NEW_ARCHITECTURE = os.environ.get("NEXUSML_USE_NEW_ARCHITECTURE", "false").lower() == "true"

if USE_NEW_ARCHITECTURE:
    # Use new architecture
    orchestrator = PipelineOrchestrator(factory, context)
    model, metrics = orchestrator.train_model(data_path=data_path, **kwargs)
else:
    # Use old architecture
    model, metrics = train_model_legacy(data_path=data_path, **kwargs)
```

### Incremental Integration

The new architecture is designed to be integrated incrementally, with each
integration step maintaining backward compatibility. This approach allows you to
migrate your code gradually without breaking existing functionality.

```python
# Step 1: Use new configuration system with old components
config_provider = ConfigurationProvider()
config = config_provider.config

# Step 2: Use new data loader with old pipeline
data_loader = factory.create_data_loader()
data = data_loader.load_data(data_path)
# Use data with old pipeline

# Step 3: Use new feature engineer with old model builder
feature_engineer = factory.create_feature_engineer()
features = feature_engineer.engineer_features(data)
# Use features with old model builder

# Step 4: Use new model builder with old trainer
model_builder = factory.create_model_builder()
model = model_builder.build_model()
# Use model with old trainer

# Step 5: Use new trainer with old evaluator
model_trainer = factory.create_model_trainer()
trained_model = model_trainer.train(model, X_train, y_train)
# Use trained_model with old evaluator

# Step 6: Use new evaluator with old serializer
model_evaluator = factory.create_model_evaluator()
metrics = model_evaluator.evaluate(trained_model, X_test, y_test)
# Use metrics with old serializer

# Step 7: Use new serializer with old predictor
model_serializer = factory.create_model_serializer()
model_serializer.save_model(trained_model, model_path)
# Use model_path with old predictor

# Step 8: Use new predictor
predictor = factory.create_predictor()
predictions = predictor.predict(trained_model, data)
```

### Comprehensive Testing

Each migration step should be thoroughly tested to ensure functionality is
maintained. This includes unit tests, integration tests, and end-to-end tests.

```python
# Unit test for new data loader
def test_data_loader():
    data_loader = factory.create_data_loader()
    data = data_loader.load_data(data_path)
    assert data is not None
    assert len(data) > 0
    assert "description" in data.columns

# Integration test for new data loader with old pipeline
def test_data_loader_with_old_pipeline():
    data_loader = factory.create_data_loader()
    data = data_loader.load_data(data_path)
    # Use data with old pipeline
    result = old_pipeline.process(data)
    assert result is not None
    assert len(result) > 0

# End-to-end test for new architecture
def test_new_architecture():
    orchestrator = PipelineOrchestrator(factory, context)
    model, metrics = orchestrator.train_model(data_path=data_path, **kwargs)
    assert model is not None
    assert metrics is not None
    assert "accuracy" in metrics
```

## Backward Compatibility

The new architecture is designed to maintain backward compatibility with
existing code. This is achieved through the following mechanisms:

1. **Adapter Pattern**: Adapters are provided for all components, allowing old
   components to be used in the new architecture and new components to be used
   in the old architecture.

2. **Feature Flags**: Feature flags are used to toggle between old and new code
   paths, allowing you to test the new architecture without affecting existing
   functionality.

3. **Compatibility Layers**: Compatibility layers are provided for all entry
   points, allowing you to use the new architecture with existing code.

4. **Configuration Migration**: Tools are provided for migrating from old
   configuration files to the new unified format.

### Adapter Examples

```python
# Adapter for old data loader
class OldDataLoaderAdapter(DataLoader):
    def __init__(self, old_loader):
        self.old_loader = old_loader

    def load_data(self, data_path=None, **kwargs):
        # Adapt old interface to new interface
        return self.old_loader.load(data_path)

    def get_config(self):
        # Provide configuration for old component
        return {}

# Adapter for new data loader
class NewDataLoaderAdapter:
    def __init__(self, new_loader):
        self.new_loader = new_loader

    def load(self, path):
        # Adapt new interface to old interface
        return self.new_loader.load_data(path)
```

### Feature Flag Examples

```python
# Feature flag for using new architecture
USE_NEW_ARCHITECTURE = os.environ.get("NEXUSML_USE_NEW_ARCHITECTURE", "false").lower() == "true"

# Feature flag for using new configuration system
USE_NEW_CONFIG = os.environ.get("NEXUSML_USE_NEW_CONFIG", "false").lower() == "true"

# Feature flag for using new data loader
USE_NEW_DATA_LOADER = os.environ.get("NEXUSML_USE_NEW_DATA_LOADER", "false").lower() == "true"

# Feature flag for using new feature engineer
USE_NEW_FEATURE_ENGINEER = os.environ.get("NEXUSML_USE_NEW_FEATURE_ENGINEER", "false").lower() == "true"

# Feature flag for using new model builder
USE_NEW_MODEL_BUILDER = os.environ.get("NEXUSML_USE_NEW_MODEL_BUILDER", "false").lower() == "true"
```

### Compatibility Layer Examples

```python
# Compatibility layer for train_model function
def train_model(data_path, **kwargs):
    if USE_NEW_ARCHITECTURE:
        # Use new architecture
        orchestrator = PipelineOrchestrator(factory, context)
        model, metrics = orchestrator.train_model(data_path=data_path, **kwargs)
        return model, metrics
    else:
        # Use old architecture
        return train_model_legacy(data_path=data_path, **kwargs)

# Compatibility layer for predict function
def predict(model, data, **kwargs):
    if USE_NEW_ARCHITECTURE:
        # Use new architecture
        orchestrator = PipelineOrchestrator(factory, context)
        predictions = orchestrator.predict(model=model, data=data, **kwargs)
        return predictions
    else:
        # Use old architecture
        return predict_legacy(model, data, **kwargs)
```

### Configuration Migration Examples

```python
# Migrate from old configuration files to new unified format
from nexusml.core.config.migration import migrate_from_default_paths

# Migrate configurations and save to the default path
config = migrate_from_default_paths()
```

## Migration Checklist

Use the following checklist to guide your migration from the old architecture to
the new architecture:

### Configuration Migration

- [ ] Identify all configuration files used in your code
- [ ] Migrate configuration files to the new unified format
- [ ] Update code to use the new configuration system
- [ ] Test configuration loading and validation

### Component Migration

- [ ] Identify all components used in your code
- [ ] Create adapters for components that need to be migrated
- [ ] Update code to use the new component interfaces
- [ ] Test components with the new interfaces

### Pipeline Migration

- [ ] Identify all pipeline execution code in your code
- [ ] Create adapters for pipeline components that need to be migrated
- [ ] Update code to use the new pipeline orchestrator
- [ ] Test pipeline execution with the new orchestrator

### Entry Point Migration

- [ ] Identify all entry points used in your code
- [ ] Create compatibility layers for entry points that need to be migrated
- [ ] Update code to use the new entry points
- [ ] Test entry points with the new architecture

### Testing

- [ ] Write unit tests for all migrated components
- [ ] Write integration tests for all migrated pipelines
- [ ] Write end-to-end tests for all migrated entry points
- [ ] Verify that all tests pass with the new architecture

### Cleanup

- [ ] Remove feature flags once migration is complete
- [ ] Remove adapters once migration is complete
- [ ] Remove compatibility layers once migration is complete
- [ ] Remove old code once migration is complete

## Conclusion

Migrating from the old NexusML architecture to the new refactored architecture
is a gradual process that can be done incrementally while maintaining backward
compatibility. By following the migration strategy outlined in this document and
using the provided adapters, feature flags, and compatibility layers, you can
migrate your code with minimal disruption to existing functionality.

For more detailed information about specific migration tasks, see the following
documentation:

- [Configuration Migration](configuration.md)
- [Component Migration](components.md)
