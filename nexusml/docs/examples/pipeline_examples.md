# Pipeline Examples

This document provides documentation for the pipeline examples in NexusML, which demonstrate different approaches to creating, configuring, and executing machine learning pipelines.

## Pipeline Factory Example

### Overview

The `pipeline_factory_example.py` script demonstrates how to use the Pipeline Factory to create and configure pipeline components with proper dependencies. It showcases the component registry, dependency injection container, and factory pattern used in NexusML.

### Key Features

- Component registration and default implementation selection
- Dependency injection for component creation
- Factory pattern for creating pipeline components
- Complete pipeline workflow from data loading to prediction
- Automatic dependency resolution

### Usage

```python
# Run the example
python -m nexusml.examples.pipeline_factory_example
```

### Code Walkthrough

#### Component Implementations

The example first defines mock implementations for all the pipeline components:

```python
# Mock implementations for demonstration purposes
class CSVDataLoader(DataLoader):
    """Example CSV data loader."""
    # Implementation...

class StandardPreprocessor(DataPreprocessor):
    """Example standard preprocessor."""
    # Implementation...

class TextFeatureEngineer(FeatureEngineer):
    """Example text feature engineer."""
    # Implementation...

class RandomForestModelBuilder(ModelBuilder):
    """Example random forest model builder."""
    # Implementation...

class StandardModelTrainer(ModelTrainer):
    """Example standard model trainer."""
    # Implementation...

class StandardModelEvaluator(ModelEvaluator):
    """Example standard model evaluator."""
    # Implementation...

class PickleModelSerializer(ModelSerializer):
    """Example pickle model serializer."""
    # Implementation...

class StandardPredictor(Predictor):
    """Example standard predictor."""
    # Implementation...
```

#### Component Registration

It then registers these components with the registry:

```python
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
registry.register(ModelSerializer, "pickle", PickleModelSerializer)
registry.register(Predictor, "standard", StandardPredictor)

# Set default implementations
registry.set_default_implementation(DataLoader, "csv")
registry.set_default_implementation(DataPreprocessor, "standard")
# ... other default implementations
```

#### Factory Creation

It creates a factory using the registry and container:

```python
# Create a factory
factory = PipelineFactory(registry, container)
```

#### Component Creation

It creates pipeline components using the factory:

```python
# Create pipeline components
data_loader = factory.create_data_loader(file_path="data.csv")
preprocessor = factory.create_data_preprocessor()
feature_engineer = factory.create_feature_engineer()
model_builder = factory.create_model_builder(n_estimators=200)
model_trainer = factory.create_model_trainer()
model_evaluator = factory.create_model_evaluator()
model_serializer = factory.create_model_serializer()
predictor = factory.create_predictor()
```

#### Pipeline Execution

It executes the pipeline components in sequence:

```python
# Use the components to build a pipeline
print("\n=== Loading and Preprocessing Data ===")
data = data_loader.load_data()
preprocessed_data = preprocessor.preprocess(data)

print("\n=== Feature Engineering ===")
features = feature_engineer.engineer_features(preprocessed_data)

# Split the data
print("\n=== Splitting Data ===")
X = features.drop("target", axis=1)
y = features["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== Building and Training Model ===")
model = model_builder.build_model()
trained_model = model_trainer.train(model, X_train, y_train)

print("\n=== Evaluating Model ===")
evaluation = model_evaluator.evaluate(trained_model, X_test, y_test)

print("\n=== Saving Model ===")
model_serializer.save_model(trained_model, "model.pkl")

print("\n=== Making Predictions ===")
new_data = pd.DataFrame(
    {"description": ["New Heat Exchanger"], "service_life": [22.0]}
)
predictions = predictor.predict(trained_model, new_data)
```

#### Dependency Resolution

It demonstrates automatic dependency resolution:

```python
# Create a component with dependencies
class ComponentWithDependencies:
    def __init__(self, data_loader, preprocessor):
        self.data_loader = data_loader
        self.preprocessor = preprocessor

# Register the component
registry.register(ComponentWithDependencies, "default", ComponentWithDependencies)
registry.set_default_implementation(ComponentWithDependencies, "default")

# Create the component - the factory will automatically resolve dependencies
component = factory.create(ComponentWithDependencies)
```

### Dependencies

- nexusml.core.di.container: For dependency injection
- nexusml.core.pipeline.factory: For the pipeline factory
- nexusml.core.pipeline.interfaces: For component interfaces
- nexusml.core.pipeline.registry: For component registration
- scikit-learn: For data splitting and model implementation
- pandas: For data manipulation

### Notes and Warnings

- The example uses mock implementations for demonstration purposes
- In a real application, you would use actual implementations
- The factory automatically resolves dependencies based on the component constructor
- The registry allows multiple implementations of the same interface
- The default implementation is used when no specific implementation is requested

## Pipeline Orchestrator Example

### Overview

The `pipeline_orchestrator_example.py` script demonstrates how to use the PipelineOrchestrator to train a model, make predictions, and evaluate the model's performance. It showcases the high-level API for executing complete machine learning workflows.

### Key Features

- Complete pipeline orchestration
- Training, prediction, and evaluation workflows
- Error handling and logging
- Execution summary and metrics
- Model saving and loading

### Usage

```python
# Run the example
python -m nexusml.examples.pipeline_orchestrator_example
```

### Code Walkthrough

#### Orchestrator Creation

The example first creates a PipelineOrchestrator with all necessary components:

```python
def create_orchestrator():
    """Create a PipelineOrchestrator instance."""
    # Create a component registry
    registry = ComponentRegistry()

    # Register component implementations
    # ... (component implementations)

    # Register the components
    registry.register(DataLoader, "standard", StandardDataLoader)
    registry.register(DataPreprocessor, "standard", StandardPreprocessor)
    # ... (other registrations)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "standard")
    registry.set_default_implementation(DataPreprocessor, "standard")
    # ... (other defaults)

    # Create a dependency injection container
    container = DIContainer()

    # Create a pipeline factory
    factory = PipelineFactory(registry, container)

    # Create a pipeline context
    context = PipelineContext()

    # Create a pipeline orchestrator
    orchestrator = PipelineOrchestrator(factory, context)

    return orchestrator
```

#### Training Example

It demonstrates training a model:

```python
def train_model_example(orchestrator, logger):
    """Example of training a model using the orchestrator."""
    logger.info("Training model example")

    # Define paths
    data_path = "examples/sample_data.xlsx"
    feature_config_path = "nexusml/config/feature_config.yml"
    output_dir = "outputs/models"

    # Train the model
    try:
        model, metrics = orchestrator.train_model(
            data_path=data_path,
            feature_config_path=feature_config_path,
            test_size=0.3,
            random_state=42,
            optimize_hyperparameters=True,
            output_dir=output_dir,
            model_name="equipment_classifier",
        )

        logger.info("Model training completed successfully")
        logger.info(f"Model saved to: {orchestrator.context.get('model_path')}")
        logger.info(f"Metadata saved to: {orchestrator.context.get('metadata_path')}")
        logger.info("Metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

        # Get execution summary
        summary = orchestrator.get_execution_summary()
        logger.info("Execution summary:")
        # ... (log summary details)

        return model

    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None
```

#### Prediction Example

It demonstrates making predictions:

```python
def predict_example(orchestrator, model, logger):
    """Example of making predictions using the orchestrator."""
    logger.info("Prediction example")

    # Create sample data for prediction
    data = pd.DataFrame(
        {
            "equipment_tag": ["AHU-01", "CHW-01", "P-01"],
            "manufacturer": ["Trane", "Carrier", "Armstrong"],
            "model": ["M-1000", "C-2000", "A-3000"],
            "description": [
                "Air Handling Unit with cooling coil",
                "Centrifugal Chiller for HVAC system",
                "Centrifugal Pump for chilled water",
            ],
        }
    )

    # Make predictions
    try:
        predictions = orchestrator.predict(
            model=model,
            data=data,
            output_path="outputs/predictions.csv",
        )

        logger.info("Predictions completed successfully")
        logger.info(f"Predictions saved to: {orchestrator.context.get('output_path')}")
        logger.info("Sample predictions:")
        # ... (log prediction details)

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None
```

#### Evaluation Example

It demonstrates evaluating a model:

```python
def evaluate_example(orchestrator, model, logger):
    """Example of evaluating a model using the orchestrator."""
    logger.info("Evaluation example")

    # Define paths
    data_path = "examples/sample_data.xlsx"
    output_path = "outputs/evaluation_results.json"

    # Evaluate the model
    try:
        results = orchestrator.evaluate(
            model=model,
            data_path=data_path,
            output_path=output_path,
        )

        logger.info("Evaluation completed successfully")
        logger.info(f"Evaluation results saved to: {output_path}")
        logger.info("Metrics:")
        for key, value in results["metrics"].items():
            logger.info(f"  {key}: {value}")

        # ... (log execution summary)

        return results

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None
```

#### Error Handling Example

It demonstrates error handling:

```python
def error_handling_example(orchestrator, logger):
    """Example of error handling in the orchestrator."""
    logger.info("Error handling example")

    # Try to train a model with a nonexistent data path
    try:
        model, metrics = orchestrator.train_model(
            data_path="nonexistent_path.csv",
            feature_config_path="nexusml/config/feature_config.yml",
        )
    except Exception as e:
        logger.info(f"Expected error caught: {e}")
        logger.info("Context status: " + orchestrator.context.status)
        logger.info("Error handling worked correctly")
```

### Dependencies

- nexusml.core.di.container: For dependency injection
- nexusml.core.pipeline.context: For pipeline context
- nexusml.core.pipeline.factory: For pipeline factory
- nexusml.core.pipeline.orchestrator: For pipeline orchestration
- nexusml.core.pipeline.registry: For component registration
- nexusml.core.pipeline.interfaces: For component interfaces
- pandas: For data manipulation
- logging: For logging
- os, sys, pathlib: For file system operations

### Notes and Warnings

- The orchestrator provides a high-level API for executing complete workflows
- It handles component creation, execution, and error handling
- The context stores state and data during pipeline execution
- The execution summary provides timing and status information
- Error handling is built into the orchestrator

## Pipeline Stages Example

### Overview

The `pipeline_stages_example.py` script demonstrates how to use the pipeline stages to create a complete machine learning pipeline for equipment classification. It showcases the stage-based approach to pipeline construction.

### Key Features

- Stage-based pipeline construction
- Configuration-driven stages
- Pipeline context for state management
- Error handling and execution summary
- Model card generation

### Usage

```python
# Run the example
python -m nexusml.examples.pipeline_stages_example
```

### Code Walkthrough

#### Pipeline Context Creation

The example first creates a pipeline context:

```python
# Create a pipeline context
context = PipelineContext()
context.start()

# Create a configuration manager
config_manager = ConfigurationManager()
```

#### Pipeline Stages Definition

It defines the pipeline stages:

```python
# Define the pipeline stages
stages = [
    ConfigurableDataLoadingStage(
        config={"loader_type": "csv"},
        config_manager=config_manager,
    ),
    ConfigDrivenValidationStage(
        config={"config_name": "production_data_config"},
        config_manager=config_manager,
    ),
    SimpleFeatureEngineeringStage(),
    RandomSplittingStage(
        config={
            "test_size": 0.3,
            "random_state": 42,
        }
    ),
    ConfigDrivenModelBuildingStage(
        config={
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
    ),
    StandardModelTrainingStage(),
    ClassificationEvaluationStage(),
    ModelCardSavingStage(
        config={
            "model_name": "Equipment Classifier",
            "model_version": "1.0.0",
            "model_description": "A random forest model for classifying equipment based on descriptions.",
            "model_authors": ["NexusML Team"],
            "model_license": "Proprietary",
        }
    ),
    StandardPredictionStage(),
]
```

#### Stage Execution

It executes each stage:

```python
# Execute each stage
for stage in stages:
    print(f"Executing stage: {stage.get_name()}")
    
    # Skip stages that require data we don't have yet
    if stage.get_name() == "ModelCardSaving" and not context.has("trained_model"):
        print("Skipping model saving stage (no trained model)")
        continue
        
    if stage.get_name() == "StandardPrediction" and not context.has("trained_model"):
        print("Skipping prediction stage (no trained model)")
        continue

    # Execute the stage
    if stage.get_name() == "ConfigurableDataLoading":
        # Pass the data path to the data loading stage
        stage.execute(context, data_path=data_path)
        
        # Print the column names for debugging
        if context.has("data"):
            data = context.get("data")
            print("\nAvailable columns in the loaded data:")
            for col in data.columns:
                print(f"  - {col}")
            print()
    elif stage.get_name() == "RandomSplitting":
        # Pass target columns to the data splitting stage
        stage.execute(
            context,
            target_columns=[
                "category_name",
                "uniformat_code",
                "mcaa_system_category",
                "System_Type_ID",
                "Equip_Name_ID",
            ],
        )
    elif stage.get_name() == "ModelCardSaving":
        # Pass the output path to the model saving stage
        output_path = os.path.join(
            Path(__file__).resolve().parent.parent,
            "output",
            "models",
            "equipment_classifier.pkl",
        )
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get the model and metadata
        model = context.get("trained_model")
        metadata = {
            "evaluation_results": context.get("evaluation_results", {}),
            "created_at": pd.Timestamp.now().isoformat(),
        }
        
        # Execute the stage
        stage.save_model(model, output_path, metadata)
    else:
        # Execute the stage normally
        stage.execute(context)
```

#### Execution Summary

It prints the execution summary:

```python
# Print the execution summary
print("\nExecution Summary:")
summary = context.get_execution_summary()
for key, value in summary.items():
    if key == "component_execution_times":
        print(f"Component Execution Times:")
        for component, time in value.items():
            print(f"  {component}: {time:.2f} seconds")
    elif key in ["accessed_keys", "modified_keys"]:
        print(f"{key}: {', '.join(value)}")
    else:
        print(f"{key}: {value}")

# Print evaluation results if available
if context.has("evaluation_results"):
    print("\nEvaluation Results:")
    evaluation_results = context.get("evaluation_results")
    if "overall" in evaluation_results:
        print("Overall Metrics:")
        for metric, value in evaluation_results["overall"].items():
            print(f"  {metric}: {value}")

# End the pipeline execution
context.end("completed")
print("\nPipeline execution completed successfully.")
```

### Dependencies

- nexusml.config.manager: For configuration management
- nexusml.core.pipeline.context: For pipeline context
- nexusml.core.pipeline.stages: For pipeline stages
- pandas: For data manipulation
- os, pathlib: For file system operations

### Notes and Warnings

- The stage-based approach provides more flexibility than the orchestrator
- Each stage is responsible for a specific part of the pipeline
- The context stores state and data during pipeline execution
- Stages can be configured with different parameters
- Error handling is built into the pipeline execution

## Integrated Classifier Example

### Overview

The `integrated_classifier_example.py` script demonstrates the comprehensive equipment classification model that integrates multiple classification systems, EAV (Entity-Attribute-Value) structure for flexible equipment attributes, and ML capabilities to fill in missing attribute data.

### Key Features

- Multiple classification systems (OmniClass, MasterFormat, Uniformat)
- EAV structure for flexible equipment attributes
- ML capabilities to fill in missing attribute data
- Attribute validation and template generation
- Attribute extraction from descriptions

### Usage

```python
# Run the example
python -m nexusml.examples.integrated_classifier_example
```

### Code Walkthrough

#### Classifier Initialization and Training

The example first initializes and trains the equipment classifier:

```python
# Initialize the equipment classifier
print("\nInitializing Equipment Classifier...")
classifier = EquipmentClassifier()

# Train the model
print("\nTraining the model...")
classifier.train()
```

#### Making Predictions

It makes predictions for example equipment descriptions:

```python
# Example equipment descriptions
examples = [
    {
        "description": "Centrifugal chiller with 500 tons cooling capacity, 0.6 kW/ton efficiency, using R-134a refrigerant",
        "service_life": 20.0,
    },
    # ... other examples
]

# Make predictions for each example
print("\nMaking predictions and generating EAV templates...")
results = []

for i, example in enumerate(examples):
    print(f"\nExample {i+1}: {example['description'][:50]}...")

    # Make prediction
    prediction = classifier.predict(example["description"], example["service_life"])

    # Extract basic classification results
    basic_result = {
        "description": example["description"],
        "service_life": example["service_life"],
        "Equipment_Category": prediction["Equipment_Category"],
        "Uniformat_Class": prediction["Uniformat_Class"],
        "System_Type": prediction["System_Type"],
        "MasterFormat_Class": prediction["MasterFormat_Class"],
        "OmniClass_ID": prediction.get("OmniClass_ID", ""),
        "Uniformat_ID": prediction.get("Uniformat_ID", ""),
    }

    print(f"Predicted Equipment Category: {basic_result['Equipment_Category']}")
    print(f"Predicted MasterFormat Class: {basic_result['MasterFormat_Class']}")
    print(f"Predicted OmniClass ID: {basic_result['OmniClass_ID']}")

    # Get the attribute template
    template = prediction.get("attribute_template", {})

    # Try to extract attributes from the description
    equipment_type = prediction["Equipment_Category"]
    extracted_attributes = {}

    if hasattr(classifier, "predict_attributes"):
        extracted_attributes = classifier.predict_attributes(
            equipment_type, example["description"]
        )

        if extracted_attributes:
            print("\nExtracted attributes from description:")
            for attr, value in extracted_attributes.items():
                print(f"  {attr}: {value}")

    # Add results to the list
    basic_result["extracted_attributes"] = extracted_attributes
    basic_result["attribute_template"] = template
    results.append(basic_result)
```

#### EAV Template Generation

It generates EAV templates for different equipment types:

```python
# Generate a complete EAV template example
print("\nGenerating complete EAV template example...")
eav_manager = EAVManager()

# Get templates for different equipment types
equipment_types = ["Chiller", "Air Handler", "Boiler", "Pump", "Cooling Tower"]
templates = {}

for eq_type in equipment_types:
    templates[eq_type] = eav_manager.generate_attribute_template(eq_type)

# Save templates to JSON file
templates_file = output_dir / "equipment_templates.json"
with open(templates_file, "w") as f:
    json.dump(templates, f, indent=2)

print(f"Equipment templates saved to {templates_file}")
```

#### Attribute Validation

It demonstrates attribute validation:

```python
# Demonstrate attribute validation
print("\nDemonstrating attribute validation...")

# Example: Valid attributes for a chiller
valid_attributes = {
    "cooling_capacity_tons": 500,
    "efficiency_kw_per_ton": 0.6,
    "refrigerant_type": "R-134a",
    "chiller_type": "Centrifugal",
}

# Example: Invalid attributes for a chiller (missing required, has unknown)
invalid_attributes = {
    "cooling_capacity_tons": 500,
    "unknown_attribute": "value",
    "chiller_type": "Centrifugal",
}

# Validate attributes
valid_result = eav_manager.validate_attributes("Chiller", valid_attributes)
invalid_result = eav_manager.validate_attributes("Chiller", invalid_attributes)

print("\nValid attributes validation result:")
print(f"  Missing required: {valid_result['missing_required']}")
print(f"  Unknown attributes: {valid_result['unknown']}")

print("\nInvalid attributes validation result:")
print(f"  Missing required: {invalid_result['missing_required']}")
print(f"  Unknown attributes: {invalid_result['unknown']}")
```

#### Filling Missing Attributes

It demonstrates filling missing attributes:

```python
# Demonstrate filling missing attributes
print("\nDemonstrating filling missing attributes...")

# Example: Partial attributes for a chiller
partial_attributes = {"cooling_capacity_tons": 500, "chiller_type": "Centrifugal"}

# Description with additional information
description = "Centrifugal chiller with 500 tons cooling capacity, 0.6 kW/ton efficiency, using R-134a refrigerant"

# Fill missing attributes
filled_attributes = eav_manager.fill_missing_attributes(
    "Chiller", partial_attributes, description, classifier
)

print("\nPartial attributes:")
print(json.dumps(partial_attributes, indent=2))

print("\nFilled attributes:")
print(json.dumps(filled_attributes, indent=2))
```

### Dependencies

- nexusml.core.eav_manager: For EAV management
- nexusml.core.model: For the EquipmentClassifier
- pandas: For data manipulation
- json, os, sys, pathlib: For file system operations

### Notes and Warnings

- The integrated classifier combines multiple classification systems
- The EAV structure provides flexibility for different equipment types
- Attribute extraction uses ML to extract values from descriptions
- Attribute validation ensures data quality
- Attribute filling uses ML to fill in missing values

## Common Patterns

Across all four pipeline examples, you can observe these common patterns:

1. **Component-Based Architecture**: All examples use a component-based architecture where each component is responsible for a specific part of the pipeline.

2. **Dependency Injection**: Components receive their dependencies through constructor injection, making them more testable and maintainable.

3. **Factory Pattern**: The factory pattern is used to create components with proper dependencies.

4. **Context Management**: The pipeline context stores state and data during pipeline execution.

5. **Configuration-Driven Behavior**: Many components and stages are configured through configuration objects or files.

6. **Error Handling**: All examples include error handling to gracefully handle failures.

7. **Execution Summary**: The pipeline provides an execution summary with timing and status information.

## Best Practices

### When to Use Each Approach

1. **Pipeline Factory**: Use when you need fine-grained control over component creation and configuration.

2. **Pipeline Orchestrator**: Use for high-level workflows with standard components and configurations.

3. **Pipeline Stages**: Use when you need more flexibility in pipeline construction and execution.

4. **Integrated Classifier**: Use when you need a complete classification system with attribute management.

### Pipeline Design

When designing a pipeline:

1. **Define Clear Interfaces**: Each component should have a clear interface.

2. **Use Dependency Injection**: Components should receive their dependencies through constructor injection.

3. **Make Components Configurable**: Components should be configurable through configuration objects or files.

4. **Handle Errors Gracefully**: The pipeline should handle errors gracefully and provide useful error messages.

5. **Provide Execution Summary**: The pipeline should provide an execution summary with timing and status information.

### Integration with Other Components

The pipeline components can be integrated with other components:

1. **Data Loading**: Use the data loading components to load data from various sources.

2. **Feature Engineering**: Use the feature engineering components to transform raw data into features.

3. **Model Building**: Use the model building components to create and configure models.

4. **Model Training**: Use the model training components to train models.

5. **Model Evaluation**: Use the model evaluation components to evaluate models.

6. **Prediction**: Use the prediction components to make predictions.

## Next Steps

After understanding pipeline examples, you might want to explore:

1. **Domain-Specific Examples**: Explore examples specific to equipment classification.

2. **Usage Guide**: Learn how to use NexusML in your own projects.

3. **API Reference**: Explore the complete API reference for NexusML.