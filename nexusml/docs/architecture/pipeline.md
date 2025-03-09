# Pipeline Architecture

## Overview

NexusML uses a pipeline architecture to process data through distinct stages from loading to prediction. Each stage is handled by specialized components with clear interfaces.

## Diagrams

The following diagrams illustrate the pipeline architecture:

- [Architecture Overview](../../diagrams/nexusml/architecture_overview.puml) - High-level view of system components
- [Pipeline Flow](../../diagrams/nexusml/pipeline_flow.puml) - Data flow through training and prediction pipelines
- [Component Relationships](../../diagrams/nexusml/component_relationships.puml) - Relationships between pipeline components

To render these diagrams, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Pipeline Components

| Component | Interface | Responsibility |
|-----------|-----------|----------------|
| DataLoader | `load_data()` | Load data from various sources |
| DataPreprocessor | `preprocess()` | Clean and prepare data |
| FeatureEngineer | `engineer_features()`, `fit()`, `transform()` | Transform raw data into features |
| ModelBuilder | `build_model()`, `optimize_hyperparameters()` | Create and configure models |
| ModelTrainer | `train()`, `cross_validate()` | Train models with data |
| ModelEvaluator | `evaluate()`, `analyze_predictions()` | Calculate performance metrics |
| ModelSerializer | `save_model()`, `load_model()` | Save/load models to/from disk |
| Predictor | `predict()`, `predict_proba()` | Make predictions with trained models |

## Component Registry

The ComponentRegistry maps interfaces to implementations and manages default implementations.

```python
# Create registry
registry = ComponentRegistry()

# Register implementations
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataLoader, "excel", ExcelDataLoader)
registry.register(FeatureEngineer, "generic", GenericFeatureEngineer)
registry.register(FeatureEngineer, "text", TextFeatureEngineer)

# Set default implementations
registry.set_default_implementation(DataLoader, "csv")
registry.set_default_implementation(FeatureEngineer, "generic")
```

## Pipeline Factory

The PipelineFactory creates component instances using the registry and dependency injection.

```python
# Create factory
factory = PipelineFactory(registry, container)

# Create components
data_loader = factory.create(DataLoader)  # Returns default implementation
excel_loader = factory.create(DataLoader, "excel")  # Returns specific implementation
```

## Pipeline Context

The PipelineContext stores state and data during pipeline execution.

```python
# Create context
context = PipelineContext()

# Store data
context.set("raw_data", data)
context.set("features", features)

# Retrieve data
raw_data = context.get("raw_data")
features = context.get("features")

# Check if data exists
if context.has("features"):
    features = context.get("features")
```

## Pipeline Orchestrator

The PipelineOrchestrator coordinates the execution of pipeline components.

```python
# Create orchestrator
orchestrator = PipelineOrchestrator(factory, context)

# Training workflow
model, metrics = orchestrator.train_model(
    data_path="data.csv",
    test_size=0.3,
    random_state=42,
    optimize_hyperparameters=True,
    output_dir="outputs/models",
    model_name="equipment_classifier",
)

# Prediction workflow
predictions = orchestrator.predict(
    model=model,
    data_path="new_data.csv",
    output_path="outputs/predictions.csv",
)
```

## Training Pipeline Flow

1. **Data Loading**: `DataLoader.load_data()` loads data from source
2. **Data Preprocessing**: `DataPreprocessor.preprocess()` cleans data
3. **Train/Test Split**: Data is split into training and testing sets
4. **Feature Engineering**: `FeatureEngineer.fit()` and `transform()` create features
5. **Model Building**: `ModelBuilder.build_model()` creates model instance
6. **Hyperparameter Optimization**: `ModelBuilder.optimize_hyperparameters()` tunes model (optional)
7. **Model Training**: `ModelTrainer.train()` trains model with features
8. **Model Evaluation**: `ModelEvaluator.evaluate()` calculates metrics
9. **Model Serialization**: `ModelSerializer.save_model()` saves model to disk

## Prediction Pipeline Flow

1. **Model Loading**: `ModelSerializer.load_model()` loads model from disk (if needed)
2. **Data Loading**: `DataLoader.load_data()` loads new data
3. **Data Preprocessing**: `DataPreprocessor.preprocess()` cleans data
4. **Feature Engineering**: `FeatureEngineer.transform()` creates features
5. **Prediction**: `Predictor.predict()` generates predictions
6. **Output**: Predictions are saved or returned

## Creating Custom Pipeline Components

1. Implement the appropriate interface:

```python
from nexusml.core.pipeline.interfaces import DataLoader

class CustomDataLoader(DataLoader):
    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        # Custom implementation
        return data
```

2. Register with the ComponentRegistry:

```python
registry.register(DataLoader, "custom", CustomDataLoader)
```

3. Use in pipeline:

```python
# Create factory with registry
factory = PipelineFactory(registry, container)

# Create orchestrator
orchestrator = PipelineOrchestrator(factory, context)

# Use custom loader
orchestrator.train_model(
    data_path="data.csv",
    data_loader_name="custom",  # Specify custom implementation
    # Other parameters...
)
```

## Pipeline Configuration

Pipeline components can be configured through:

1. **Constructor Parameters**: Passed when creating component instances
2. **Method Parameters**: Passed when calling component methods
3. **Configuration System**: Global settings from configuration files

Example with constructor parameters:

```python
# Register with constructor parameters
registry.register(
    DataLoader, 
    "csv", 
    CSVDataLoader, 
    constructor_params={"encoding": "utf-8", "delimiter": ","}
)
```

Example with method parameters:

```python
# Pass parameters to method
orchestrator.train_model(
    data_path="data.csv",
    feature_engineering_params={
        "text_columns": ["description"],
        "numerical_columns": ["service_life"],
        "categorical_columns": ["category"]
    }
)
```

## Error Handling

The pipeline includes error handling mechanisms:

1. **Component-level Validation**: Components validate inputs
2. **Orchestrator Error Handling**: Orchestrator catches and logs errors
3. **Context Error State**: Context stores error information
4. **Execution Summary**: Summary includes error details

```python
try:
    result = orchestrator.train_model(...)
except Exception as e:
    # Handle error
    print(f"Pipeline error: {str(e)}")
    
    # Get execution summary
    summary = orchestrator.get_execution_summary()
    print(f"Status: {summary['status']}")
    print(f"Error: {summary.get('error', 'None')}")