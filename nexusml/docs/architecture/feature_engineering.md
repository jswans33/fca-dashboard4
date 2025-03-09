# Feature Engineering

## Overview

The feature engineering system in NexusML transforms raw data into features suitable for machine learning models. It handles text, numerical, and categorical data through a flexible transformer-based architecture.

## Diagram

The following diagram illustrates the feature engineering system:

- [Feature Engineering System](../../diagrams/nexusml/feature_engineering.puml) - Components and relationships of the feature engineering system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### FeatureEngineer Interface

The main interface for feature engineering components in the pipeline.

```python
class FeatureEngineer:
    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Engineer features from input data."""
        pass
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FeatureEngineer':
        """Fit the feature engineer to the input data."""
        pass
        
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Transform the input data using the fitted feature engineer."""
        pass
```

### Transformer Interface

Interface for individual feature transformers.

```python
class Transformer:
    def fit(self, X, y=None):
        """Fit the transformer to the input data."""
        pass
        
    def transform(self, X):
        """Transform the input data."""
        pass
        
    def fit_transform(self, X, y=None):
        """Fit and transform the input data."""
        pass
```

### GenericFeatureEngineer

Default implementation that handles text, numerical, and categorical features.

```python
from nexusml.core.feature_engineering import GenericFeatureEngineer

# Create feature engineer
engineer = GenericFeatureEngineer(
    text_columns=["description"],
    numerical_columns=["service_life"],
    categorical_columns=["category"]
)

# Fit to training data
engineer.fit(training_data)

# Transform new data
features = engineer.transform(new_data)
```

### ConfigDrivenFeatureEngineer

Creates transformers based on configuration settings.

```python
from nexusml.core.feature_engineering import ConfigDrivenFeatureEngineer
from nexusml.core.config.provider import ConfigProvider

# Get configuration
config = ConfigProvider.get_config().feature_engineering

# Create feature engineer
engineer = ConfigDrivenFeatureEngineer(config)

# Fit and transform
engineer.fit(training_data)
features = engineer.transform(new_data)
```

### TransformerRegistry

Registry for transformer implementations that can be created by name.

```python
from nexusml.core.feature_engineering.registry import TransformerRegistry

# Create registry
registry = TransformerRegistry()

# Register transformer
registry.register("text", TextTransformer)
registry.register("numerical", NumericalTransformer)
registry.register("categorical", CategoricalTransformer)

# Create transformer by name
text_transformer = registry.create_transformer("text", max_features=1000)
```

## Standard Transformers

### TextTransformer

Transforms text data using TF-IDF vectorization.

```python
from nexusml.core.feature_engineering.transformers import TextTransformer

# Create transformer
transformer = TextTransformer(max_features=1000, ngram_range=(1, 2))

# Fit and transform
text_features = transformer.fit_transform(data["description"])
```

### NumericalTransformer

Transforms numerical data using standardization.

```python
from nexusml.core.feature_engineering.transformers import NumericalTransformer

# Create transformer
transformer = NumericalTransformer()

# Fit and transform
numerical_features = transformer.fit_transform(data[["service_life", "cost"]])
```

### CategoricalTransformer

Transforms categorical data using one-hot encoding.

```python
from nexusml.core.feature_engineering.transformers import CategoricalTransformer

# Create transformer
transformer = CategoricalTransformer(handle_unknown="ignore")

# Fit and transform
categorical_features = transformer.fit_transform(data[["category", "type"]])
```

### ClassificationSystemMapper

Maps classification system codes to numerical IDs.

```python
from nexusml.core.feature_engineering.transformers import ClassificationSystemMapper

# Create mapper
mapper = ClassificationSystemMapper(name="omniclass")

# Fit and transform
mapped_codes = mapper.fit_transform(data["omniclass_code"])

# Get mapping
mapping = mapper.get_mapping()
```

## Feature Engineering Process

The feature engineering process follows these steps:

1. **Data Validation**: Validate input data structure and required columns
2. **Transformer Creation**: Create transformers for each feature type
3. **Fitting**: Fit transformers to training data
4. **Transformation**: Apply transformers to input data
5. **Feature Combination**: Combine transformed features into a single feature set

```python
def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Engineer features from input data."""
    # Validate data
    self._validate_data(data)
    
    # Create transformers if not already created
    if not hasattr(self, "_transformers") or self._transformers is None:
        self._transformers = self._create_transformers()
    
    # Transform text features
    text_features = self._transform_text_features(data)
    
    # Transform numerical features
    numerical_features = self._transform_numerical_features(data)
    
    # Transform categorical features
    categorical_features = self._transform_categorical_features(data)
    
    # Combine features
    combined_features = self._combine_features(
        text_features, numerical_features, categorical_features
    )
    
    return combined_features
```

## Configuration

Feature engineering can be configured through:

### YAML Configuration

```yaml
# feature_config.yml
feature_engineering:
  text_columns:
    - description
    - name
  numerical_columns:
    - service_life
    - cost
  categorical_columns:
    - category
    - type
  transformers:
    text:
      type: tfidf
      max_features: 1000
      ngram_range: [1, 2]
    numerical:
      type: standard_scaler
    categorical:
      type: one_hot_encoder
      handle_unknown: ignore
```

### Code Configuration

```python
# Create feature engineer with explicit configuration
engineer = GenericFeatureEngineer(
    text_columns=["description", "name"],
    numerical_columns=["service_life", "cost"],
    categorical_columns=["category", "type"]
)

# Or use configuration-driven approach
from nexusml.core.config.provider import ConfigProvider

# Initialize with custom config
ConfigProvider.initialize("path/to/feature_config.yml")

# Get configuration
config = ConfigProvider.get_config().feature_engineering

# Create feature engineer
engineer = ConfigDrivenFeatureEngineer(config)
```

## Custom Transformers

You can create custom transformers by implementing the Transformer interface:

```python
from nexusml.core.feature_engineering.base import BaseTransformer

class CustomTransformer(BaseTransformer):
    def __init__(self, param1=1, param2="default"):
        self.param1 = param1
        self.param2 = param2
        self._fitted = False
        
    def fit(self, X, y=None):
        # Implement fitting logic
        self._fitted = True
        return self
        
    def transform(self, X):
        # Validate that fit has been called
        if not self._fitted:
            raise ValueError("Transformer not fitted")
            
        # Implement transformation logic
        result = # ... your transformation ...
        
        return result
```

Register your custom transformer:

```python
from nexusml.core.feature_engineering.registry import TransformerRegistry

# Get registry
registry = TransformerRegistry()

# Register custom transformer
registry.register("custom", CustomTransformer)
```

Use in configuration:

```yaml
feature_engineering:
  transformers:
    custom:
      type: custom
      param1: 42
      param2: "custom_value"
```

## Pipeline Integration

Feature engineering is integrated into the pipeline system:

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

# Train model with feature engineering configuration
model, metrics = orchestrator.train_model(
    data_path="data.csv",
    feature_engineering_params={
        "text_columns": ["description"],
        "numerical_columns": ["service_life"],
        "categorical_columns": ["category"]
    }
)
```

## Advanced Usage

### Feature Selection

```python
from nexusml.core.feature_engineering import FeatureSelector

# Create feature selector
selector = FeatureSelector(k=100, method="chi2")

# Fit and transform
selected_features = selector.fit_transform(features, labels)
```

### Feature Importance Analysis

```python
from nexusml.core.feature_engineering import FeatureImportanceAnalyzer

# Create analyzer
analyzer = FeatureImportanceAnalyzer()

# Analyze feature importance
importance = analyzer.analyze(model, feature_names)

# Print top features
for name, score in importance[:10]:
    print(f"{name}: {score:.4f}")
```

### Custom Feature Combination

```python
from nexusml.core.feature_engineering import GenericFeatureEngineer

class CustomFeatureEngineer(GenericFeatureEngineer):
    def _combine_features(self, text_features, numerical_features, categorical_features):
        # Custom combination logic
        combined = super()._combine_features(
            text_features, numerical_features, categorical_features
        )
        
        # Add interaction features
        if numerical_features is not None and categorical_features is not None:
            for num_col in numerical_features.columns:
                for cat_col in categorical_features.columns:
                    combined[f"{num_col}_{cat_col}"] = (
                        numerical_features[num_col] * categorical_features[cat_col]
                    )
                    
        return combined