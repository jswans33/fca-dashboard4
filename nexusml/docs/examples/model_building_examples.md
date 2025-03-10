# Model Building Examples

This document provides documentation for the model building examples in NexusML, which demonstrate different approaches to building, training, and evaluating machine learning models.

## Model Building Example

### Overview

The `model_building_example.py` script demonstrates how to use the model building and training components in NexusML to create, train, evaluate, and save machine learning models. It showcases different model types, training approaches, and evaluation methods.

### Key Features

- Multiple model builders (Random Forest, Gradient Boosting, Ensemble)
- Standard and advanced training methods
- Model evaluation with comprehensive metrics
- Model serialization (saving and loading)
- Synthetic data generation for demonstration

### Usage

```python
# Run the example
python -m nexusml.examples.model_building_example
```

### Code Walkthrough

#### Data Loading

The example first sets up sample data for demonstration:

```python
def load_sample_data():
    """Load sample data for demonstration."""
    # Try to load from standard locations
    data_paths = [
        "data/sample_data.csv",
        "examples/sample_data.csv",
        "nexusml/data/sample_data.csv",
        "nexusml/examples/sample_data.csv",
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"Loading data from {path}")
            df = pd.read_csv(path)
            break
    else:
        # If no file is found, create synthetic data
        print("Creating synthetic data")
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=3,
            random_state=42,
        )
        
        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_df["combined_features"] = X_df.apply(
            lambda row: " ".join([f"{col}:{val:.2f}" for col, val in row.items()]), axis=1
        )
        X_df["service_life"] = np.random.uniform(10, 30, size=X.shape[0])
        
        # Convert y to DataFrame with multiple target columns
        y_df = pd.DataFrame({
            "category_name": [f"Category_{i}" for i in y],
            "uniformat_code": [f"U{i}" for i in y],
            "mcaa_system_category": [f"System_{i}" for i in y],
            "Equipment_Type": [f"Type_{i}" for i in y],
            "System_Subtype": [f"Subtype_{i}" for i in y],
        })
        
        return X_df, y_df
```

#### Example 1: Random Forest Model

The first example demonstrates building and training a Random Forest model:

```python
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

#### Example 2: Gradient Boosting Model

The second example demonstrates building and training a Gradient Boosting model:

```python
# Create a Gradient Boosting model builder
gb_builder = GradientBoostingBuilder()

# Build the model
gb_model = gb_builder.build_model()
print("Gradient Boosting model built")

# Create a standard model trainer
gb_trainer = StandardModelTrainer()

# Train the model
trained_gb_model = gb_trainer.train(gb_model, X_train, y_train)
print("Gradient Boosting model trained\n")
```

#### Example 3: Ensemble Model

The third example demonstrates building and training an Ensemble model:

```python
# Create an Ensemble model builder
ensemble_builder = EnsembleBuilder()

# Build the model
ensemble_model = ensemble_builder.build_model()
print("Ensemble model built")

# Create a standard model trainer
ensemble_trainer = StandardModelTrainer()

# Train the model
optimized_model = ensemble_trainer.train(ensemble_model, X_train, y_train)
print("Ensemble model trained\n")
```

#### Example 4: Model Evaluation

The fourth example demonstrates evaluating the trained models:

```python
# Create a model evaluator
evaluator = BaseModelEvaluator()

# Evaluate the Random Forest model
rf_metrics = evaluator.evaluate(trained_rf_model, X_test, y_test)
print("Random Forest model evaluated")
print(f"Overall accuracy: {rf_metrics['overall']['accuracy_mean']:.4f}")
print(f"Overall F1 score: {rf_metrics['overall']['f1_macro_mean']:.4f}\n")

# Evaluate the Gradient Boosting model
gb_metrics = evaluator.evaluate(trained_gb_model, X_test, y_test)
print("Gradient Boosting model evaluated")
print(f"Overall accuracy: {gb_metrics['overall']['accuracy_mean']:.4f}")
print(f"Overall F1 score: {gb_metrics['overall']['f1_macro_mean']:.4f}\n")

# Evaluate the Ensemble model
ensemble_metrics = evaluator.evaluate(optimized_model, X_test, y_test)
print("Ensemble model evaluated")
print(f"Overall accuracy: {ensemble_metrics['overall']['accuracy_mean']:.4f}")
print(f"Overall F1 score: {ensemble_metrics['overall']['f1_macro_mean']:.4f}\n")
```

#### Example 5: Model Serialization

The fifth example demonstrates saving and loading models:

```python
# Create a model serializer
serializer = BaseModelSerializer()

# Create output directory if it doesn't exist
os.makedirs("outputs/models", exist_ok=True)

# Save the best model
best_model_path = "outputs/models/best_model.pkl"
serializer.save_model(optimized_model, best_model_path)
print(f"Best model saved to {best_model_path}")

# Load the model
loaded_model = serializer.load_model(best_model_path)
print("Best model loaded")

# Evaluate the loaded model
loaded_metrics = evaluator.evaluate(loaded_model, X_test, y_test)
print("Loaded model evaluated")
print(f"Overall accuracy: {loaded_metrics['overall']['accuracy_mean']:.4f}")
print(f"Overall F1 score: {loaded_metrics['overall']['f1_macro_mean']:.4f}\n")
```

### Dependencies

- nexusml.core.model_building: For model builders
- nexusml.core.model_training: For model trainers
- nexusml.core.model_building.base: For model evaluation and serialization
- scikit-learn: For data splitting and synthetic data generation
- pandas: For data manipulation
- os: For file system operations

### Notes and Warnings

- The example creates synthetic data if no sample data file is found
- The model evaluation metrics may vary depending on the random state and data
- The example saves models to the "outputs/models" directory
- The EnsembleBuilder combines multiple models for potentially better performance
- For production use, you should tune hyperparameters and perform cross-validation

## Training Pipeline Example

### Overview

The `training_pipeline_example.py` script demonstrates how to use the updated training pipeline entry point with the pipeline orchestrator. It shows various configuration options, error handling examples, and feature flags for backward compatibility.

### Key Features

- Complete training pipeline with orchestrator
- Basic and advanced training configurations
- Hyperparameter optimization
- Visualization generation
- Error handling and validation
- Feature flags for backward compatibility
- Sample prediction with trained models

### Usage

```python
# Run the example
python -m nexusml.examples.training_pipeline_example
```

### Code Walkthrough

#### Setup Example Data

The example first sets up sample data for demonstration:

```python
def setup_example_data():
    """Set up example data for the training pipeline."""
    # Create example data directory if it doesn't exist
    example_data_dir = Path("nexusml/examples/data")
    example_data_dir.mkdir(parents=True, exist_ok=True)

    # Path to the example data file
    data_path = example_data_dir / "example_training_data.csv"

    # Create example data if it doesn't exist
    if not data_path.exists():
        # Create a simple DataFrame with example data
        data = pd.DataFrame(
            {
                "equipment_tag": [f"EQ-{i:03d}" for i in range(1, 101)],
                "manufacturer": ["Trane", "Carrier", "York", "Daikin", "Lennox"] * 20,
                # ... other columns
                "description": [
                    "Air Handling Unit with cooling coil",
                    "Centrifugal Chiller for HVAC system",
                    "Centrifugal Pump for chilled water",
                    "Supply Fan for air distribution",
                    "Hot Water Boiler for heating",
                ] * 20,
                "service_life": [15, 20, 25, 30, 35] * 20,
            }
        )

        # Save the example data
        data.to_csv(data_path, index=False)
        print(f"Created example data file: {data_path}")

    return data_path
```

#### Basic Example

The basic example demonstrates a simple training configuration:

```python
def basic_example(logger):
    """Basic example of using the training pipeline."""
    logger.info("Running basic example")

    # Set up example data
    data_path = setup_example_data()

    # Create training arguments
    args = TrainingArguments(
        data_path=str(data_path),
        test_size=0.3,
        random_state=42,
        output_dir="nexusml/examples/output/models",
        model_name="example_model",
    )

    # Train the model
    try:
        model, metrics, _ = train_with_orchestrator(args, logger)

        logger.info("Model training completed successfully")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        return model

    except Exception as e:
        logger.error(f"Error in basic example: {e}")
        return None
```

#### Advanced Example

The advanced example demonstrates hyperparameter optimization and visualizations:

```python
def advanced_example(logger):
    """Advanced example with hyperparameter optimization and visualizations."""
    logger.info("Running advanced example")

    # Set up example data
    data_path = setup_example_data()

    # Create training arguments
    args = TrainingArguments(
        data_path=str(data_path),
        test_size=0.2,
        random_state=123,
        optimize_hyperparameters=True,
        output_dir="nexusml/examples/output/models",
        model_name="advanced_model",
        visualize=True,
    )

    # Train the model
    try:
        model, metrics, viz_paths = train_with_orchestrator(args, logger)

        logger.info("Model training completed successfully")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        logger.info("Visualizations:")
        if viz_paths:
            for key, path in viz_paths.items():
                logger.info(f"  {key}: {path}")

        return model

    except Exception as e:
        logger.error(f"Error in advanced example: {e}")
        return None
```

#### Prediction Example

The prediction example demonstrates making predictions with a trained model:

```python
def prediction_example(model, logger):
    """Example of making predictions with a trained model."""
    logger.info("Running prediction example")

    # Create orchestrator
    orchestrator = create_orchestrator(logger)

    # Create sample data for prediction
    data = pd.DataFrame(
        {
            "equipment_tag": ["TEST-01", "TEST-02", "TEST-03"],
            "manufacturer": ["Trane", "Carrier", "York"],
            "model": ["Model-X", "Model-Y", "Model-Z"],
            "description": [
                "Air Handling Unit with cooling coil and variable frequency drive",
                "Water-cooled centrifugal chiller with high efficiency",
                "Vertical inline pump for condenser water system",
            ],
            "service_life": [20, 25, 30],
        }
    )

    # Make predictions
    try:
        predictions = orchestrator.predict(model=model, data=data)

        logger.info("Predictions:")
        for i, row in predictions.iterrows():
            logger.info(f"Item {i+1}:")
            logger.info(f"  Equipment Tag: {data.iloc[i]['equipment_tag']}")
            logger.info(f"  Description: {data.iloc[i]['description']}")
            for col in row.index:
                logger.info(f"  {col}: {row[col]}")

        return predictions

    except Exception as e:
        logger.error(f"Error in prediction example: {e}")
        return None
```

#### Error Handling Example

The error handling example demonstrates validation and error handling:

```python
def error_handling_example(logger):
    """Example of error handling in the training pipeline."""
    logger.info("Running error handling example")

    # Create training arguments with non-existent data path
    args = TrainingArguments(
        data_path="non_existent_file.csv",  # This will be caught by validation
        test_size=0.3,
        random_state=42,
    )

    # This should raise a ValueError
    try:
        train_with_orchestrator(args, logger)
    except ValueError as e:
        logger.info(f"Expected error caught: {e}")
        logger.info("Error handling worked correctly")
```

#### Feature Flags Example

The feature flags example demonstrates backward compatibility:

```python
def feature_flags_example(logger):
    """Example of using feature flags for backward compatibility."""
    logger.info("Running feature flags example")

    # Set up example data
    data_path = setup_example_data()

    # Example with orchestrator (new implementation)
    logger.info("Using orchestrator (new implementation)")
    args_new = TrainingArguments(
        data_path=str(data_path),
        use_orchestrator=True,
    )

    # Example with legacy implementation
    logger.info("Using legacy implementation")
    args_legacy = TrainingArguments(
        data_path=str(data_path),
        use_orchestrator=False,
    )

    # Note: In a real example, we would call the main function with these arguments
    # For this example, we'll just show the different configurations
    logger.info(f"New implementation args: {args_new.to_dict()}")
    logger.info(f"Legacy implementation args: {args_legacy.to_dict()}")
```

### Dependencies

- nexusml.core.cli.training_args: For training arguments and logging setup
- nexusml.train_model_pipeline_v2: For orchestrator creation and training functions
- pandas: For data manipulation
- logging: For logging
- os, sys, pathlib: For file system operations

### Notes and Warnings

- The example creates a CSV file in the "nexusml/examples/data" directory
- Models are saved to the "nexusml/examples/output/models" directory
- The example demonstrates error handling for invalid inputs
- Feature flags allow for backward compatibility with legacy implementations
- For production use, you should provide your own data rather than using the synthetic data

## Key Components

### Model Builders

- **RandomForestBuilder**: Creates Random Forest classifier models
- **GradientBoostingBuilder**: Creates Gradient Boosting classifier models
- **EnsembleBuilder**: Creates ensemble models combining multiple classifiers

### Model Trainers

- **StandardModelTrainer**: Basic trainer for single-step training
- **CrossValidationTrainer**: Trainer with cross-validation
- **GridSearchOptimizer**: Trainer with hyperparameter optimization

### Pipeline Components

- **TrainingArguments**: Configuration for the training pipeline
- **create_orchestrator**: Creates a pipeline orchestrator
- **train_with_orchestrator**: Trains a model using the orchestrator
- **make_sample_prediction_with_orchestrator**: Makes predictions with a trained model

## Best Practices

### Model Selection

When selecting a model type:

1. **Random Forest**: Good for general-purpose classification with moderate dataset sizes
2. **Gradient Boosting**: Often provides better performance but may require more tuning
3. **Ensemble**: Combines multiple models for potentially better performance but at the cost of increased complexity

### Training Configuration

When configuring training:

1. **test_size**: Typically 0.2-0.3 (20-30% of data for testing)
2. **random_state**: Set for reproducibility
3. **optimize_hyperparameters**: Enable for better performance (but slower training)
4. **visualize**: Enable to generate visualizations for model analysis

### Error Handling

Always use try-except blocks when working with the training pipeline to catch and handle errors gracefully:

```python
try:
    model, metrics, _ = train_with_orchestrator(args, logger)
    # Process results
except Exception as e:
    logger.error(f"Error in training: {e}")
    # Handle error
```

## Next Steps

After understanding model building, you might want to explore:

1. **Pipeline Examples**: See how model building fits into the complete machine learning pipeline
2. **Domain-Specific Examples**: Explore model building techniques specific to equipment classification