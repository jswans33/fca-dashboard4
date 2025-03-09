# Pipeline Stages

This package provides implementations of pipeline stages for the NexusML pipeline system. Each stage represents a distinct step in the pipeline execution process and follows the Single Responsibility Principle (SRP) from SOLID.

## Overview

The pipeline stages are organized into the following categories:

1. **Data Loading Stages**: Load data from various sources
2. **Validation Stages**: Validate data against requirements
3. **Feature Engineering Stages**: Transform raw data into features
4. **Data Splitting Stages**: Split data into training and testing sets
5. **Model Building Stages**: Create and configure machine learning models
6. **Model Training Stages**: Train machine learning models
7. **Model Evaluation Stages**: Evaluate trained models
8. **Model Saving Stages**: Save trained models and metadata
9. **Prediction Stages**: Make predictions using trained models

Each category has multiple implementations that can be used interchangeably, and each implementation follows a common interface.

## Architecture

The pipeline stages follow a consistent architecture:

- **Interfaces**: Define the contract for each stage type
- **Base Implementations**: Provide common functionality for all stages
- **Concrete Implementations**: Implement specific functionality for each stage type

### Interfaces

All pipeline stages implement the `PipelineStage` interface, which defines the following methods:

- `execute(context, **kwargs)`: Execute the stage
- `get_name()`: Get the name of the stage
- `get_description()`: Get a description of the stage
- `validate_context(context)`: Validate that the context contains all required data

Each stage type also has a specific interface that extends `PipelineStage` and adds methods specific to that stage type.

### Base Implementations

Base implementations provide common functionality for all stages of a particular type. They implement the stage interface and provide default behavior where appropriate.

### Concrete Implementations

Concrete implementations provide specific functionality for each stage type. They extend the base implementation and override methods as needed.

## Usage

### Basic Usage

To use a pipeline stage, you need to:

1. Create an instance of the stage
2. Create a pipeline context
3. Execute the stage with the context

```python
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages import ConfigurableDataLoadingStage

# Create a pipeline context
context = PipelineContext()
context.start()

# Create a data loading stage
data_loading_stage = ConfigurableDataLoadingStage()

# Execute the stage
data_loading_stage.execute(context, data_path="path/to/data.csv")

# Access the loaded data
data = context.get("data")
```

### Creating a Pipeline

You can create a complete pipeline by chaining multiple stages together:

```python
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages import (
    ConfigurableDataLoadingStage,
    ConfigDrivenValidationStage,
    SimpleFeatureEngineeringStage,
    RandomSplittingStage,
    RandomForestModelBuildingStage,
    StandardModelTrainingStage,
    ClassificationEvaluationStage,
)

# Create a pipeline context
context = PipelineContext()
context.start()

# Define the pipeline stages
stages = [
    ConfigurableDataLoadingStage(),
    ConfigDrivenValidationStage(),
    SimpleFeatureEngineeringStage(),
    RandomSplittingStage(),
    RandomForestModelBuildingStage(),
    StandardModelTrainingStage(),
    ClassificationEvaluationStage(),
]

# Execute each stage
for stage in stages:
    stage.execute(context)

# Access the results
model = context.get("trained_model")
evaluation_results = context.get("evaluation_results")
```

### Configuration

Most stages accept a configuration dictionary that can be used to customize their behavior:

```python
from nexusml.core.pipeline.stages import RandomForestModelBuildingStage

# Create a model building stage with custom configuration
model_building_stage = RandomForestModelBuildingStage(
    config={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    }
)
```

### Dependency Injection

Stages can also accept dependencies through constructor injection:

```python
from nexusml.config.manager import ConfigurationManager
from nexusml.core.pipeline.stages import ConfigDrivenValidationStage

# Create a configuration manager
config_manager = ConfigurationManager()

# Create a validation stage with the configuration manager
validation_stage = ConfigDrivenValidationStage(
    config_manager=config_manager
)
```

## Available Stages

### Data Loading Stages

- `CSVDataLoadingStage`: Loads data from CSV files
- `ExcelDataLoadingStage`: Loads data from Excel files
- `SQLiteDataLoadingStage`: Loads data from SQLite databases
- `ConfigurableDataLoadingStage`: Loads data from various sources based on configuration

### Validation Stages

- `ColumnValidationStage`: Validates specific columns
- `DataTypeValidationStage`: Validates data types
- `CompositeValidationStage`: Combines multiple validators
- `DataFrameValidationStage`: Validates the entire DataFrame
- `ConfigDrivenValidationStage`: Validates data against configuration-defined rules

### Feature Engineering Stages

- `TextFeatureEngineeringStage`: Engineers text features
- `NumericFeatureEngineeringStage`: Engineers numeric features
- `HierarchicalFeatureEngineeringStage`: Engineers hierarchical features
- `CompositeFeatureEngineeringStage`: Combines multiple feature engineers
- `SimpleFeatureEngineeringStage`: Performs simplified feature engineering
- `ConfigDrivenFeatureEngineeringStage`: Engineers features based on configuration

### Data Splitting Stages

- `RandomSplittingStage`: Splits data randomly
- `StratifiedSplittingStage`: Splits data with stratification
- `TimeSeriesSplittingStage`: Splits time series data
- `CrossValidationSplittingStage`: Splits data for cross-validation
- `ConfigDrivenDataSplittingStage`: Splits data based on configuration

### Model Building Stages

- `RandomForestModelBuildingStage`: Builds Random Forest models
- `GradientBoostingModelBuildingStage`: Builds Gradient Boosting models
- `EnsembleModelBuildingStage`: Builds ensemble models
- `ConfigDrivenModelBuildingStage`: Builds models based on configuration

### Model Training Stages

- `StandardModelTrainingStage`: Trains models using standard training
- `CrossValidationTrainingStage`: Trains models using cross-validation
- `GridSearchTrainingStage`: Trains models using grid search
- `RandomizedSearchTrainingStage`: Trains models using randomized search
- `ConfigDrivenModelTrainingStage`: Trains models based on configuration

### Model Evaluation Stages

- `ClassificationEvaluationStage`: Evaluates classification models
- `DetailedClassificationEvaluationStage`: Performs detailed evaluation of classification models
- `ConfigDrivenModelEvaluationStage`: Evaluates models based on configuration

### Model Saving Stages

- `PickleModelSavingStage`: Saves models using pickle
- `ModelCardSavingStage`: Saves models with model cards
- `ConfigDrivenModelSavingStage`: Saves models based on configuration

### Prediction Stages

- `StandardPredictionStage`: Makes standard predictions
- `ProbabilityPredictionStage`: Makes probability predictions
- `ThresholdPredictionStage`: Makes predictions with custom thresholds
- `ConfigDrivenPredictionStage`: Makes predictions based on configuration

## Examples

See the `nexusml/examples/pipeline_stages_example.py` file for a complete example of how to use the pipeline stages.