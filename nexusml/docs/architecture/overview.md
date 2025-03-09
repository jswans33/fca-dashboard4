# NexusML Architecture Overview

## Core Components

### Pipeline Components
- **DataLoader**: Loads data from CSV, Excel, and databases
- **DataPreprocessor**: Cleans and prepares data
- **FeatureEngineer**: Transforms raw data into ML features
- **ModelBuilder**: Creates and configures ML models
- **ModelTrainer**: Trains models with data
- **ModelEvaluator**: Calculates performance metrics
- **ModelSerializer**: Saves/loads models to/from disk
- **Predictor**: Makes predictions with trained models

### Management Components
- **ComponentRegistry**: Maps interfaces to implementations
- **PipelineFactory**: Creates component instances
- **PipelineOrchestrator**: Coordinates pipeline execution
- **PipelineContext**: Stores state during execution
- **DIContainer**: Manages component dependencies

### Configuration System
- **ConfigProvider**: Singleton access to configuration
- **Configuration**: Validated settings via Pydantic
- **YAMLConfigLoader**: Loads settings from YAML files

## Data Flow

1. **Data Loading**: CSV/Excel → DataFrame
2. **Preprocessing**: Raw data → Clean data
3. **Feature Engineering**: Clean data → Feature vectors
4. **Model Building**: Parameters → Model instance
5. **Training**: Features + Labels → Trained model
6. **Evaluation**: Predictions + Ground truth → Metrics
7. **Serialization**: Model → Disk
8. **Prediction**: New data → Classifications

## Key Interfaces

```python
# Core interfaces in nexusml.core.pipeline.interfaces

class DataLoader:
    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame: ...

class FeatureEngineer:
    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame: ...
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FeatureEngineer': ...
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame: ...

class ModelBuilder:
    def build_model(self, **kwargs) -> Any: ...
    def optimize_hyperparameters(self, model: Any, x_train: pd.DataFrame, 
                                y_train: pd.DataFrame, **kwargs) -> Any: ...

class ModelTrainer:
    def train(self, model: Any, x_train: pd.DataFrame, 
             y_train: pd.DataFrame, **kwargs) -> Any: ...
    def cross_validate(self, model: Any, x: pd.DataFrame, 
                      y: pd.DataFrame, **kwargs) -> Dict[str, List[float]]: ...

class ModelEvaluator:
    def evaluate(self, model: Any, x_test: pd.DataFrame, 
                y_test: pd.DataFrame, **kwargs) -> Dict[str, float]: ...
    def analyze_predictions(self, model: Any, x_test: pd.DataFrame, 
                           y_test: pd.DataFrame, y_pred: pd.DataFrame, 
                           **kwargs) -> Dict[str, Any]: ...

class ModelSerializer:
    def save_model(self, model: Any, path: str, **kwargs) -> None: ...
    def load_model(self, path: str, **kwargs) -> Any: ...

class Predictor:
    def predict(self, model: Any, data: pd.DataFrame, **kwargs) -> pd.DataFrame: ...
    def predict_proba(self, model: Any, data: pd.DataFrame, 
                     **kwargs) -> Dict[str, pd.DataFrame]: ...
```

## Dependency Injection

```python
# Register components
registry = ComponentRegistry()
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataLoader, "excel", ExcelDataLoader)
registry.set_default_implementation(DataLoader, "csv")

# Create container
container = DIContainer()

# Create factory
factory = PipelineFactory(registry, container)

# Get component instance
data_loader = factory.create(DataLoader)  # Returns CSVDataLoader instance
excel_loader = factory.create(DataLoader, "excel")  # Returns ExcelDataLoader instance
```

## Pipeline Orchestration

```python
# Create orchestrator
orchestrator = PipelineOrchestrator(factory, context)

# Train model
model, metrics = orchestrator.train_model(
    data_path="data.csv",
    test_size=0.3,
    random_state=42,
    optimize_hyperparameters=True,
    output_dir="outputs/models",
    model_name="equipment_classifier",
)

# Make predictions
predictions = orchestrator.predict(
    model=model,
    data_path="new_data.csv",
    output_path="outputs/predictions.csv",
)
```

## Extension Points

1. **Custom Components**: Implement interfaces and register with ComponentRegistry
2. **Custom Transformers**: Extend base transformer classes for feature engineering
3. **Custom Models**: Use any scikit-learn compatible model
4. **Pipeline Hooks**: Register callbacks for pipeline events
5. **Configuration Overrides**: Override default settings via YAML or environment variables

## Design Patterns

- **Factory Pattern**: PipelineFactory creates components
- **Strategy Pattern**: Interchangeable algorithm implementations
- **Dependency Injection**: Components receive dependencies
- **Singleton**: ConfigProvider ensures single configuration instance
- **Observer**: Pipeline events notify registered callbacks
- **Adapter**: Compatibility layers for different APIs