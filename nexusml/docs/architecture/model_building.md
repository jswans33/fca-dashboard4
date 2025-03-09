# Model Building

## Overview

The model building system in NexusML creates and configures machine learning models for equipment classification. It supports various model types, hyperparameter optimization, and multi-target classification.

## Diagram

The following diagram illustrates the model building system:

- [Model Building System](../../diagrams/nexusml/model_building.puml) - Components and relationships of the model building system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### ModelBuilder Interface

The main interface for model building components in the pipeline.

```python
class ModelBuilder:
    def build_model(self, **kwargs) -> Any:
        """Build a machine learning model."""
        pass
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs) -> Any:
        """Optimize hyperparameters for the model."""
        pass
```

### RandomForestBuilder

Builds Random Forest classification models.

```python
from nexusml.core.model_building.builders import RandomForestBuilder

# Create builder
builder = RandomForestBuilder(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Build model
model = builder.build_model()

# Optimize hyperparameters
optimized_model = builder.optimize_hyperparameters(
    model, x_train, y_train
)
```

### GradientBoostingBuilder

Builds Gradient Boosting classification models.

```python
from nexusml.core.model_building.builders import GradientBoostingBuilder

# Create builder
builder = GradientBoostingBuilder(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Build model
model = builder.build_model()
```

### SVMBuilder

Builds Support Vector Machine classification models.

```python
from nexusml.core.model_building.builders import SVMBuilder

# Create builder
builder = SVMBuilder(
    C=1.0,
    kernel='rbf',
    random_state=42
)

# Build model
model = builder.build_model()
```

### ConfigDrivenModelBuilder

Creates model builders based on configuration settings.

```python
from nexusml.core.model_building import ConfigDrivenModelBuilder
from nexusml.core.config.provider import ConfigProvider

# Get configuration
config = ConfigProvider.get_config().model_building

# Create builder
builder = ConfigDrivenModelBuilder(config)

# Build model
model = builder.build_model()
```

### MultiTargetModelBuilder

Builds models for multiple target columns.

```python
from nexusml.core.model_building import MultiTargetModelBuilder
from nexusml.core.model_building.builders import RandomForestBuilder

# Create base builder
base_builder = RandomForestBuilder()

# Create multi-target builder
multi_builder = MultiTargetModelBuilder(
    base_builder=base_builder,
    target_columns=["category_name", "mcaa_system_category"]
)

# Build models for all targets
models = multi_builder.build_model()

# Access individual models
category_model = models["category_name"]
system_model = models["mcaa_system_category"]
```

### ModelRegistry

Registry for model builder implementations that can be created by name.

```python
from nexusml.core.model_building.registry import ModelRegistry

# Create registry
registry = ModelRegistry()

# Register builder
registry.register("random_forest", RandomForestBuilder)
registry.register("gradient_boosting", GradientBoostingBuilder)
registry.register("svm", SVMBuilder)

# Create builder by name
rf_builder = registry.create_builder("random_forest", n_estimators=200)
```

### HyperparameterOptimizer

Handles hyperparameter optimization using grid search or random search.

```python
from nexusml.core.model_building import HyperparameterOptimizer

# Create optimizer
optimizer = HyperparameterOptimizer(
    method="grid",  # or "random"
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

# Define parameter grid
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 10, 20, 30]
}

# Optimize model
optimized_model = optimizer.optimize(model, x_train, y_train, param_grid)
```

## Model Building Process

The model building process follows these steps:

1. **Configuration Validation**: Validate model building configuration
2. **Pipeline Creation**: Create a scikit-learn pipeline with preprocessing steps
3. **Model Configuration**: Configure the model with hyperparameters
4. **Pipeline Assembly**: Assemble the complete pipeline
5. **Hyperparameter Optimization**: Optimize hyperparameters if requested

```python
def build_model(self, **kwargs):
    """Build a machine learning model."""
    # Validate configuration
    self._validate_config()
    
    # Create pipeline
    pipeline = self._create_pipeline(**kwargs)
    
    return pipeline

def _create_pipeline(self, **kwargs):
    """Create a scikit-learn pipeline."""
    # Create preprocessing steps
    steps = []
    
    # Add classifier
    steps.append(("classifier", self._create_classifier(**kwargs)))
    
    # Create pipeline
    pipeline = Pipeline(steps)
    
    return pipeline
```

## Hyperparameter Optimization

Hyperparameter optimization is performed using scikit-learn's GridSearchCV or RandomizedSearchCV:

```python
def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
    """Optimize hyperparameters for the model."""
    # Get parameter grid
    param_grid = self._get_param_grid()
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        method=kwargs.get("method", "grid"),
        cv=kwargs.get("cv", 5),
        scoring=kwargs.get("scoring", "f1_macro"),
        n_jobs=kwargs.get("n_jobs", -1)
    )
    
    # Optimize model
    optimized_model = optimizer.optimize(model, x_train, y_train, param_grid)
    
    return optimized_model
```

## Configuration

Model building can be configured through:

### YAML Configuration

```yaml
# model_config.yml
model_building:
  model_type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
  optimization:
    method: grid
    cv: 5
    scoring: f1_macro
    param_grid:
      classifier__n_estimators: [50, 100, 200]
      classifier__max_depth: [null, 10, 20, 30]
```

### Code Configuration

```python
# Create builder with explicit configuration
builder = RandomForestBuilder(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Or use configuration-driven approach
from nexusml.core.config.provider import ConfigProvider

# Initialize with custom config
ConfigProvider.initialize("path/to/model_config.yml")

# Get configuration
config = ConfigProvider.get_config().model_building

# Create builder
builder = ConfigDrivenModelBuilder(config)
```

## Custom Model Builders

You can create custom model builders by implementing the ModelBuilder interface:

```python
from nexusml.core.model_building.base import BaseModelBuilder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline

class ExtraTreesBuilder(BaseModelBuilder):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def build_model(self, **kwargs):
        # Create pipeline
        pipeline = self._create_pipeline(**kwargs)
        
        return pipeline
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        # Get parameter grid
        param_grid = self._get_param_grid()
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            method=kwargs.get("method", "grid"),
            cv=kwargs.get("cv", 5),
            scoring=kwargs.get("scoring", "f1_macro"),
            n_jobs=kwargs.get("n_jobs", -1)
        )
        
        # Optimize model
        optimized_model = optimizer.optimize(model, x_train, y_train, param_grid)
        
        return optimized_model
        
    def _create_pipeline(self, **kwargs):
        # Create classifier
        classifier = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
        # Create pipeline
        pipeline = Pipeline([
            ("classifier", classifier)
        ])
        
        return pipeline
        
    def _get_param_grid(self):
        return {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30]
        }
```

Register your custom model builder:

```python
from nexusml.core.model_building.registry import ModelRegistry

# Get registry
registry = ModelRegistry()

# Register custom builder
registry.register("extra_trees", ExtraTreesBuilder)
```

Use in configuration:

```yaml
model_building:
  model_type: extra_trees
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42
```

## Pipeline Integration

Model building is integrated into the pipeline system:

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)

# Train model with model building configuration
model, metrics = orchestrator.train_model(
    data_path="data.csv",
    model_building_params={
        "model_type": "random_forest",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        },
        "optimize_hyperparameters": True
    }
)
```

## Advanced Usage

### Ensemble Models

```python
from nexusml.core.model_building import EnsembleBuilder
from nexusml.core.model_building.builders import RandomForestBuilder, GradientBoostingBuilder

# Create base builders
rf_builder = RandomForestBuilder(n_estimators=100)
gb_builder = GradientBoostingBuilder(n_estimators=100)

# Create ensemble builder
ensemble_builder = EnsembleBuilder(
    builders=[rf_builder, gb_builder],
    voting="soft"
)

# Build ensemble model
ensemble_model = ensemble_builder.build_model()
```

### Custom Preprocessing

```python
from nexusml.core.model_building.base import BaseModelBuilder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

class CustomPreprocessingBuilder(BaseModelBuilder):
    def __init__(self, n_estimators=100, pca_components=10):
        self.n_estimators = n_estimators
        self.pca_components = pca_components
        
    def _create_pipeline(self, **kwargs):
        # Create preprocessing steps
        steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=self.pca_components)),
            ("classifier", RandomForestClassifier(n_estimators=self.n_estimators))
        ]
        
        # Create pipeline
        pipeline = Pipeline(steps)
        
        return pipeline
```

### Model Stacking

```python
from nexusml.core.model_building import StackingBuilder
from nexusml.core.model_building.builders import RandomForestBuilder, GradientBoostingBuilder, SVMBuilder

# Create base builders
rf_builder = RandomForestBuilder(n_estimators=100)
gb_builder = GradientBoostingBuilder(n_estimators=100)
svm_builder = SVMBuilder(C=1.0)

# Create stacking builder
stacking_builder = StackingBuilder(
    builders=[rf_builder, gb_builder, svm_builder],
    final_builder=rf_builder,
    cv=5
)

# Build stacked model
stacked_model = stacking_builder.build_model()