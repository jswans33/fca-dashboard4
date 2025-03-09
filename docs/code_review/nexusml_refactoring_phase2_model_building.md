# NexusML Refactoring: Phase 2 - Model Building and Training

## Overview

This document provides a detailed summary of the Model Building and Training component implementation for Phase 2 of the NexusML refactoring. The implementation follows the SOLID principles and provides a flexible, extensible architecture for building and training machine learning models.

## Architecture

The Model Building and Training component is organized into the following modules:

1. **Interfaces**: Defines the contracts for model building and training components.
2. **Base Classes**: Provides base implementations of the interfaces with common functionality.
3. **Model Builders**: Implements specific model builders for different algorithms.
4. **Model Trainers**: Implements specific model trainers with different training strategies.
5. **Type Stubs**: Provides type hints for the interfaces to improve type safety.

### Component Diagram

```
nexusml/core/model_building/
├── interfaces.py           # Interface definitions
├── base.py                 # Base implementations
├── __init__.py             # Package initialization
└── builders/               # Model builder implementations
    ├── random_forest.py    # Random Forest model builder
    ├── gradient_boosting.py # Gradient Boosting model builder
    └── ensemble.py         # Ensemble model builder

nexusml/core/model_training/
├── __init__.py             # Package initialization
└── trainers/               # Model trainer implementations
    ├── standard.py         # Standard model trainer
    ├── cross_validation.py # Cross-validation trainer
    └── hyperparameter_optimizer.py # Hyperparameter optimizer

nexusml/types/model_building/
└── interfaces.py           # Type stubs for interfaces
```

## Key Components

### Interfaces

The interfaces define the contracts for model building and training components:

- **ModelBuilder**: Interface for model building components.
- **ConfigurableModelBuilder**: Interface for configurable model builders.
- **ModelTrainer**: Interface for model training components.
- **ConfigurableModelTrainer**: Interface for configurable model trainers.
- **HyperparameterOptimizer**: Interface for hyperparameter optimization components.
- **ModelEvaluator**: Interface for model evaluation components.
- **ModelSerializer**: Interface for model serialization components.

### Base Classes

The base classes provide default implementations of the interfaces:

- **BaseModelBuilder**: Base implementation of the ModelBuilder interface.
- **BaseConfigurableModelBuilder**: Base implementation of the ConfigurableModelBuilder interface.
- **BaseModelTrainer**: Base implementation of the ModelTrainer interface.
- **BaseConfigurableModelTrainer**: Base implementation of the ConfigurableModelTrainer interface.
- **BaseHyperparameterOptimizer**: Base implementation of the HyperparameterOptimizer interface.
- **BaseModelEvaluator**: Base implementation of the ModelEvaluator interface.
- **BaseModelSerializer**: Base implementation of the ModelSerializer interface.

### Model Builders

The model builders implement specific model building strategies:

- **RandomForestBuilder**: Builds Random Forest models.
- **GradientBoostingBuilder**: Builds Gradient Boosting models.
- **EnsembleBuilder**: Builds ensemble models that combine multiple base classifiers.

### Model Trainers

The model trainers implement specific training strategies:

- **StandardModelTrainer**: Trains models using standard training procedures.
- **CrossValidationTrainer**: Trains models using cross-validation procedures.
- **GridSearchOptimizer**: Optimizes hyperparameters using grid search.
- **RandomizedSearchOptimizer**: Optimizes hyperparameters using randomized search.

## Implementation Details

### Dependency Injection

The implementation uses dependency injection to decouple components and improve testability. The `@inject` and `@injectable` decorators from the DI container are used to inject dependencies into components.

```python
@injectable
class RandomForestBuilder(BaseConfigurableModelBuilder):
    @inject
    def __init__(
        self,
        name: str = "RandomForestBuilder",
        description: str = "Random Forest model builder using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        super().__init__(name, description, config_provider)
        logger.info(f"Initialized {name}")
```

### Configuration

The implementation uses the configuration system from Phase 1 to provide configuration-driven behavior. Components can be configured using the ConfigurationProvider or directly through their constructors.

```python
def get_default_parameters(self) -> Dict[str, Any]:
    """
    Get the default parameters for the Random Forest model.
    
    Returns:
        Dictionary of default parameters.
    """
    return {
        "tfidf": {
            "max_features": 5000,
            "ngram_range": [1, 3],
            "min_df": 2,
            "max_df": 0.9,
            "use_idf": True,
            "sublinear_tf": True,
        },
        "random_forest": {
            "n_estimators": 200,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": 42,
        },
    }
```

### Type Safety

The implementation uses type hints throughout to improve type safety. Type stubs are provided for the interfaces to enable static type checking.

```python
class ModelBuilder(Protocol):
    """
    Interface for model building components.
    
    Responsible for creating and configuring machine learning models.
    """
    
    def build_model(self, **kwargs: Any) -> Pipeline: ...
    
    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs: Any
    ) -> Pipeline: ...
    
    def get_default_parameters(self) -> Dict[str, Any]: ...
    
    def get_param_grid(self) -> Dict[str, List[Any]]: ...
```

### Error Handling

The implementation includes comprehensive error handling to provide clear error messages and prevent crashes. Exceptions are caught and logged, and appropriate error messages are provided.

```python
try:
    logger.info(f"Training model with {len(x_train)} samples")
    
    # Extract training parameters from config and kwargs
    verbose = kwargs.get("verbose", self.config.get("verbose", 1))
    
    # Log training information
    logger.info(f"X_train shape: {x_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_train columns: {x_train.columns.tolist()}")
    logger.info(f"y_train columns: {y_train.columns.tolist()}")
    
    # Train the model
    if verbose:
        print(f"Training model with {len(x_train)} samples...")
        print(f"X_train shape: {x_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    
    # Call the parent class's train method to fit the model
    trained_model = super().train(model, x_train, y_train, **kwargs)
    
    if verbose:
        print("Model training completed")
    
    logger.info("Model training completed successfully")
    return trained_model

except Exception as e:
    logger.error(f"Error training model: {str(e)}")
    raise ValueError(f"Error training model: {str(e)}") from e
```

### Logging

The implementation includes comprehensive logging to provide visibility into the model building and training process. Logs are provided at different levels (info, debug, warning, error) to provide appropriate information.

```python
logger.info(f"Training model with cross-validation on {len(x_train)} samples")

# Extract cross-validation parameters from config and kwargs
cv = kwargs.get("cv", self.config.get("cv", 5))
scoring = kwargs.get("scoring", self.config.get("scoring", "accuracy"))
verbose = kwargs.get("verbose", self.config.get("verbose", 1))
return_train_score = kwargs.get("return_train_score", self.config.get("return_train_score", True))

# Log training information
logger.info(f"X_train shape: {x_train.shape}")
logger.info(f"y_train shape: {y_train.shape}")
logger.info(f"X_train columns: {x_train.columns.tolist()}")
logger.info(f"y_train columns: {y_train.columns.tolist()}")
logger.info(f"Cross-validation folds: {cv}")
logger.info(f"Scoring metric: {scoring}")
```

## Testing

The implementation includes comprehensive unit tests to ensure the correctness of the components. Tests are provided for all interfaces, base classes, and implementations.

```python
class TestBaseModelBuilder(unittest.TestCase):
    """Tests for the BaseModelBuilder class."""
    
    def test_init(self):
        """Test initialization."""
        builder = BaseModelBuilder(name="TestBuilder", description="Test description")
        self.assertEqual(builder.get_name(), "TestBuilder")
        self.assertEqual(builder.get_description(), "Test description")
    
    def test_get_default_parameters(self):
        """Test get_default_parameters method."""
        builder = BaseModelBuilder()
        self.assertEqual(builder.get_default_parameters(), {})
    
    def test_get_param_grid(self):
        """Test get_param_grid method."""
        builder = BaseModelBuilder()
        self.assertEqual(builder.get_param_grid(), {})
    
    def test_build_model_not_implemented(self):
        """Test build_model method raises NotImplementedError."""
        builder = BaseModelBuilder()
        with self.assertRaises(NotImplementedError):
            builder.build_model()
    
    def test_optimize_hyperparameters(self):
        """Test optimize_hyperparameters method."""
        builder = BaseModelBuilder()
        model = MagicMock(spec=Pipeline)
        x_train = pd.DataFrame({"A": [1, 2, 3]})
        y_train = pd.DataFrame({"B": [4, 5, 6]})
        
        result = builder.optimize_hyperparameters(model, x_train, y_train)
        
        self.assertIs(result, model)
```

## Example Usage

The implementation includes an example script to demonstrate how to use the model building and training components.

```python
# Example 1: Building and training a Random Forest model
print("Example 1: Building and training a Random Forest model")
print("-----------------------------------------------------")

# Create a Random Forest model builder
rf_builder = RandomForestBuilder()

# Build the model
rf_model = rf_builder.build_model()
print("Random Forest model built")

# Create a standard model trainer
trainer = StandardModelTrainer()

# Train the model
trained_rf_model = trainer.train(rf_model, X_train, y_train)
print("Random Forest model trained\n")
```

## SOLID Principles

The implementation follows the SOLID principles:

1. **Single Responsibility Principle**: Each class has a single responsibility. For example, the RandomForestBuilder is responsible only for building Random Forest models.

2. **Open/Closed Principle**: The implementation is open for extension but closed for modification. New model builders and trainers can be added without modifying existing code.

3. **Liskov Substitution Principle**: Subclasses can be substituted for their base classes. For example, any ModelBuilder can be used where a BaseModelBuilder is expected.

4. **Interface Segregation Principle**: Interfaces are focused and minimal. For example, the ModelBuilder interface defines only the methods needed for building models.

5. **Dependency Inversion Principle**: High-level modules depend on abstractions, not concrete implementations. For example, the StandardModelTrainer depends on the ModelBuilder interface, not a specific implementation.

## Conclusion

The Model Building and Training component provides a flexible, extensible architecture for building and training machine learning models. It follows the SOLID principles and provides a clean, well-documented API for users. The implementation is type-safe, error-resistant, and well-tested, ensuring reliability and maintainability.