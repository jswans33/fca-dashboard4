# NexusML API Reference

This document provides detailed information about the public API and interfaces of NexusML. It covers all the public classes, methods, and functions that users can interact with.

## Table of Contents

1. [Core Pipeline](#core-pipeline)
   - [Orchestrator](#orchestrator)
   - [Factory](#factory)
   - [Registry](#registry)
   - [Context](#context)
   - [Interfaces](#interfaces)
   - [Stages](#stages)
2. [Dependency Injection](#dependency-injection)
   - [Container](#container)
   - [Decorators](#decorators)
3. [Configuration](#configuration)
   - [Provider](#provider)
   - [Loader](#loader)
   - [Schema](#schema)
4. [Feature Engineering](#feature-engineering)
   - [Base Classes](#feature-engineering-base-classes)
   - [Transformers](#transformers)
   - [Configuration](#feature-engineering-configuration)
5. [Model Building](#model-building)
   - [Base Classes](#model-building-base-classes)
   - [Model Builders](#model-builders)
   - [Model Trainers](#model-trainers)
   - [Model Evaluators](#model-evaluators)
   - [Model Serializers](#model-serializers)
6. [Data Loading](#data-loading)
   - [Data Loaders](#data-loaders)
   - [Data Preprocessors](#data-preprocessors)
7. [Prediction](#prediction)
   - [Predictors](#predictors)
8. [Validation](#validation)
   - [Rules](#validation-rules)
   - [Validators](#validators)
   - [Reports](#validation-reports)
9. [Reference Data](#reference-data)
   - [Reference Manager](#reference-manager)
10. [Model Cards](#model-cards)
    - [Generator](#model-card-generator)
    - [Schema](#model-card-schema)
11. [Utilities](#utilities)
    - [CSV Utilities](#csv-utilities)
    - [Excel Utilities](#excel-utilities)
    - [Path Utilities](#path-utilities)
    - [Logging Utilities](#logging-utilities)
    - [Verification Utilities](#verification-utilities)
12. [Command-Line Tools](#command-line-tools)
    - [Train Model Pipeline](#train-model-pipeline)
    - [Predict](#predict)
    - [Classify Equipment](#classify-equipment)
    - [Test Reference Validation](#test-reference-validation)

## Core Pipeline

### Orchestrator

The `PipelineOrchestrator` class coordinates the execution of the pipeline components.

```python
class PipelineOrchestrator:
    """
    Coordinates the execution of pipeline components.
    
    The orchestrator is responsible for creating and executing pipeline components
    in the correct order, handling errors, and providing a high-level API for
    common tasks like training models and making predictions.
    """
    
    def __init__(self, factory, context):
        """
        Initialize the orchestrator.
        
        Args:
            factory: PipelineFactory instance for creating components
            context: PipelineContext instance for storing state
        """
        
    def train_model(self, data_path=None, feature_config_path=None, test_size=0.3,
                   random_state=42, optimize_hyperparameters=False, output_dir=None,
                   model_name=None, **kwargs):
        """
        Train a model using the pipeline.
        
        Args:
            data_path: Path to the training data
            feature_config_path: Path to the feature configuration file
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            optimize_hyperparameters: Whether to optimize hyperparameters
            output_dir: Directory to save the model
            model_name: Name of the model
            **kwargs: Additional arguments for pipeline components
            
        Returns:
            tuple: (trained_model, metrics)
        """
        
    def predict(self, model=None, model_path=None, data=None, data_path=None,
               output_path=None, **kwargs):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model instance
            model_path: Path to a saved model
            data: DataFrame with prediction data
            data_path: Path to prediction data
            output_path: Path to save predictions
            **kwargs: Additional arguments for pipeline components
            
        Returns:
            DataFrame: Predictions
        """
        
    def evaluate(self, model=None, model_path=None, data=None, data_path=None,
                output_path=None, **kwargs):
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model instance
            model_path: Path to a saved model
            data: DataFrame with evaluation data
            data_path: Path to evaluation data
            output_path: Path to save evaluation results
            **kwargs: Additional arguments for pipeline components
            
        Returns:
            dict: Evaluation results
        """
        
    def save_model(self, model, path, **kwargs):
        """
        Save a trained model.
        
        Args:
            model: Trained model instance
            path: Path to save the model
            **kwargs: Additional arguments for the serializer
            
        Returns:
            str: Path to the saved model
        """
        
    def load_model(self, path, **kwargs):
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments for the serializer
            
        Returns:
            object: Loaded model
        """
        
    def get_execution_summary(self):
        """
        Get a summary of the pipeline execution.
        
        Returns:
            dict: Execution summary
        """
```

### Factory

The `PipelineFactory` class creates pipeline components with proper dependencies.

```python
class PipelineFactory:
    """
    Creates pipeline components with proper dependencies.
    
    The factory is responsible for creating instances of pipeline components
    with their dependencies resolved from the registry and container.
    """
    
    def __init__(self, registry, container):
        """
        Initialize the factory.
        
        Args:
            registry: ComponentRegistry instance
            container: DIContainer instance
        """
        
    def create(self, interface_type, implementation_name=None, **kwargs):
        """
        Create a component instance.
        
        Args:
            interface_type: Interface class
            implementation_name: Name of the implementation to create
            **kwargs: Additional arguments for the component constructor
            
        Returns:
            object: Component instance
        """
        
    def create_data_loader(self, **kwargs):
        """Create a data loader instance."""
        
    def create_data_preprocessor(self, **kwargs):
        """Create a data preprocessor instance."""
        
    def create_feature_engineer(self, **kwargs):
        """Create a feature engineer instance."""
        
    def create_model_builder(self, **kwargs):
        """Create a model builder instance."""
        
    def create_model_trainer(self, **kwargs):
        """Create a model trainer instance."""
        
    def create_model_evaluator(self, **kwargs):
        """Create a model evaluator instance."""
        
    def create_model_serializer(self, **kwargs):
        """Create a model serializer instance."""
        
    def create_predictor(self, **kwargs):
        """Create a predictor instance."""
```

### Registry

The `ComponentRegistry` class registers component implementations and their default implementations.

```python
class ComponentRegistry:
    """
    Registers component implementations and their default implementations.
    
    The registry maps interface types to their implementations and keeps track
    of the default implementation for each interface.
    """
    
    def register(self, interface_type, implementation_name, implementation_class):
        """
        Register an implementation for an interface.
        
        Args:
            interface_type: Interface class
            implementation_name: Name of the implementation
            implementation_class: Implementation class
        """
        
    def set_default_implementation(self, interface_type, implementation_name):
        """
        Set the default implementation for an interface.
        
        Args:
            interface_type: Interface class
            implementation_name: Name of the default implementation
        """
        
    def get_implementation(self, interface_type, implementation_name=None):
        """
        Get an implementation for an interface.
        
        Args:
            interface_type: Interface class
            implementation_name: Name of the implementation to get
            
        Returns:
            class: Implementation class
        """
        
    def get_default_implementation(self, interface_type):
        """
        Get the default implementation for an interface.
        
        Args:
            interface_type: Interface class
            
        Returns:
            class: Default implementation class
        """
```

### Context

The `PipelineContext` class stores state and data during pipeline execution.

```python
class PipelineContext:
    """
    Stores state and data during pipeline execution.
    
    The context provides a way to share data between pipeline components
    and track the execution of the pipeline.
    """
    
    def __init__(self):
        """Initialize the context."""
        
    def start(self):
        """Start the pipeline execution."""
        
    def end(self, status):
        """
        End the pipeline execution.
        
        Args:
            status: Status of the pipeline execution
        """
        
    def set(self, key, value):
        """
        Set a value in the context.
        
        Args:
            key: Key to store the value under
            value: Value to store
        """
        
    def get(self, key, default=None):
        """
        Get a value from the context.
        
        Args:
            key: Key to get the value for
            default: Default value to return if the key doesn't exist
            
        Returns:
            object: Value for the key
        """
        
    def has(self, key):
        """
        Check if a key exists in the context.
        
        Args:
            key: Key to check
            
        Returns:
            bool: True if the key exists, False otherwise
        """
        
    def log(self, level, message):
        """
        Log a message.
        
        Args:
            level: Log level
            message: Message to log
        """
        
    def get_execution_summary(self):
        """
        Get a summary of the pipeline execution.
        
        Returns:
            dict: Execution summary
        """
```

### Interfaces

The core pipeline interfaces define the contract for pipeline components.

```python
class DataLoader:
    """Interface for data loading components."""
    
    def load_data(self, data_path=None, **kwargs):
        """
        Load data from a source.
        
        Args:
            data_path: Path to the data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Loaded data
        """
        
    def get_config(self):
        """
        Get the configuration for the data loader.
        
        Returns:
            dict: Configuration
        """

class DataPreprocessor:
    """Interface for data preprocessing components."""
    
    def preprocess(self, data, **kwargs):
        """
        Preprocess the input data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Preprocessed data
        """
        
    def verify_required_columns(self, data):
        """
        Verify that all required columns exist in the DataFrame.
        
        Args:
            data: Input data
            
        Returns:
            DataFrame: Verified data
        """

class FeatureEngineer:
    """Interface for feature engineering components."""
    
    def engineer_features(self, data, **kwargs):
        """
        Engineer features from the input data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Data with engineered features
        """
        
    def fit(self, data, **kwargs):
        """
        Fit the feature engineer to the input data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            FeatureEngineer: Fitted feature engineer
        """
        
    def transform(self, data, **kwargs):
        """
        Transform the input data using the fitted feature engineer.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Transformed data
        """

class ModelBuilder:
    """Interface for model building components."""
    
    def build_model(self, **kwargs):
        """
        Build a machine learning model.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            object: Model instance
        """
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """
        Optimize hyperparameters for the model.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Optimized model instance
        """

class ModelTrainer:
    """Interface for model training components."""
    
    def train(self, model, x_train, y_train, **kwargs):
        """
        Train a model on the provided data.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Trained model instance
        """
        
    def cross_validate(self, model, x, y, **kwargs):
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model instance
            x: Features
            y: Targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Cross-validation results
        """

class ModelEvaluator:
    """Interface for model evaluation components."""
    
    def evaluate(self, model, x_test, y_test, **kwargs):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model instance
            x_test: Test features
            y_test: Test targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Evaluation metrics
        """
        
    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        """
        Analyze model predictions in detail.
        
        Args:
            model: Trained model instance
            x_test: Test features
            y_test: Test targets
            y_pred: Predicted targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Analysis results
        """

class ModelSerializer:
    """Interface for model serialization components."""
    
    def save_model(self, model, path, **kwargs):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model instance
            path: Path to save the model
            **kwargs: Additional arguments
        """
        
    def load_model(self, path, **kwargs):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments
            
        Returns:
            object: Loaded model instance
        """

class Predictor:
    """Interface for prediction components."""
    
    def predict(self, model, data, **kwargs):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model instance
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Predictions
        """
        
    def predict_proba(self, model, data, **kwargs):
        """
        Make probability predictions using a trained model.
        
        Args:
            model: Trained model instance
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            dict: Probability predictions
        """
```

### Stages

The pipeline stages provide a more granular approach to pipeline construction.

```python
class PipelineStage:
    """Base class for pipeline stages."""
    
    def get_name(self):
        """
        Get the name of the stage.
        
        Returns:
            str: Stage name
        """
        
    def get_description(self):
        """
        Get the description of the stage.
        
        Returns:
            str: Stage description
        """
        
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            object: Stage result
        """

class ConfigurableDataLoadingStage(PipelineStage):
    """Stage for loading data with configuration."""
    
    def __init__(self, config=None, config_manager=None):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
            config_manager: ConfigurationManager instance
        """
        
    def execute(self, context, data_path=None, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            data_path: Path to the data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Loaded data
        """

class ConfigDrivenValidationStage(PipelineStage):
    """Stage for validating data with configuration."""
    
    def __init__(self, config=None, config_manager=None):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
            config_manager: ConfigurationManager instance
        """
        
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            ValidationReport: Validation report
        """

class SimpleFeatureEngineeringStage(PipelineStage):
    """Stage for feature engineering."""
    
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Data with engineered features
        """

class RandomSplittingStage(PipelineStage):
    """Stage for splitting data into training and testing sets."""
    
    def __init__(self, config=None):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
        """
        
    def execute(self, context, target_columns=None, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            target_columns: List of target columns
            **kwargs: Additional arguments
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """

class ConfigDrivenModelBuildingStage(PipelineStage):
    """Stage for building models with configuration."""
    
    def __init__(self, config=None):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
        """
        
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            object: Model instance
        """

class StandardModelTrainingStage(PipelineStage):
    """Stage for training models."""
    
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            object: Trained model instance
        """

class ClassificationEvaluationStage(PipelineStage):
    """Stage for evaluating classification models."""
    
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            dict: Evaluation metrics
        """

class ModelCardSavingStage(PipelineStage):
    """Stage for saving model cards."""
    
    def __init__(self, config=None):
        """
        Initialize the stage.
        
        Args:
            config: Configuration dictionary
        """
        
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            str: Path to the saved model card
        """
        
    def save_model(self, model, path, metadata=None):
        """
        Save a model with metadata.
        
        Args:
            model: Trained model instance
            path: Path to save the model
            metadata: Model metadata
            
        Returns:
            str: Path to the saved model
        """

class StandardPredictionStage(PipelineStage):
    """Stage for making predictions."""
    
    def execute(self, context, **kwargs):
        """
        Execute the stage.
        
        Args:
            context: PipelineContext instance
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Predictions
        """
```

## Dependency Injection

### Container

The `DIContainer` class manages component dependencies.

```python
class DIContainer:
    """
    Manages component dependencies.
    
    The container provides a way to register and resolve dependencies
    for components.
    """
    
    def register(self, interface_type, implementation):
        """
        Register an implementation for an interface.
        
        Args:
            interface_type: Interface class
            implementation: Implementation instance
        """
        
    def resolve(self, interface_type):
        """
        Resolve an implementation for an interface.
        
        Args:
            interface_type: Interface class
            
        Returns:
            object: Implementation instance
        """
```

### Decorators

The dependency injection decorators provide a way to inject dependencies into functions and methods.

```python
def inject(func):
    """
    Decorator for injecting dependencies into functions and methods.
    
    Args:
        func: Function to inject dependencies into
        
    Returns:
        function: Wrapped function
    """

class Inject:
    """
    Marker class for injecting dependencies.
    
    Usage:
        @inject
        def my_function(data_loader=Inject(DataLoader)):
            # Use data_loader
    """
    
    def __init__(self, interface_type):
        """
        Initialize the marker.
        
        Args:
            interface_type: Interface class to inject
        """
```

## Configuration

### Provider

The `ConfigProvider` class provides access to the configuration.

```python
class ConfigProvider:
    """
    Provides access to the configuration.
    
    The provider ensures that only one configuration instance exists
    throughout the application.
    """
    
    @classmethod
    def get_config(cls):
        """
        Get the configuration instance.
        
        Returns:
            Configuration: Configuration instance
        """
        
    @classmethod
    def initialize(cls, config_path=None):
        """
        Initialize the configuration provider.
        
        Args:
            config_path: Path to the configuration file
        """
        
    @classmethod
    def reset(cls):
        """Reset the configuration provider."""
```

### Loader

The `YAMLConfigLoader` class loads configuration from YAML files.

```python
class YAMLConfigLoader:
    """
    Loads configuration from YAML files.
    
    The loader supports loading configuration from YAML files and
    applying environment variable overrides.
    """
    
    def load(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            dict: Configuration dictionary
        """
```

### Schema

The `ConfigSchema` class provides schema validation for configuration.

```python
class ConfigSchema:
    """
    Provides schema validation for configuration.
    
    The schema ensures that the configuration is valid and contains
    all required fields.
    """
    
    @classmethod
    def get_schema(cls):
        """
        Get the configuration schema.
        
        Returns:
            dict: Schema dictionary
        """
        
    @classmethod
    def validate(cls, config):
        """
        Validate a configuration against the schema.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if the configuration is valid, False otherwise
        """
```

## Feature Engineering

### Feature Engineering Base Classes

The feature engineering base classes provide the foundation for feature engineering components.

```python
class BaseFeatureTransformer:
    """Base class for feature transformers."""
    
    def fit(self, X, y=None, **kwargs):
        """
        Fit the transformer to the input data.
        
        Args:
            X: Input data
            y: Target data
            **kwargs: Additional arguments
            
        Returns:
            BaseFeatureTransformer: Fitted transformer
        """
        
    def transform(self, X, **kwargs):
        """
        Transform the input data.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Transformed data
        """
        
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the transformer to the input data and transform it.
        
        Args:
            X: Input data
            y: Target data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Transformed data
        """
        
    def get_name(self):
        """
        Get the name of the transformer.
        
        Returns:
            str: Transformer name
        """
        
    def get_description(self):
        """
        Get the description of the transformer.
        
        Returns:
            str: Transformer description
        """

class BaseColumnTransformer(BaseFeatureTransformer):
    """Base class for column transformers."""
    
    def __init__(self, input_columns, output_columns, name=None):
        """
        Initialize the transformer.
        
        Args:
            input_columns: List of input columns
            output_columns: List of output columns
            name: Transformer name
        """
        
    def _transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame: Transformed data
        """

class BaseFeatureEngineer:
    """Base class for feature engineers."""
    
    def add_transformer(self, transformer):
        """
        Add a transformer to the feature engineer.
        
        Args:
            transformer: Transformer instance
        """
        
    def fit(self, X, y=None, **kwargs):
        """
        Fit the feature engineer to the input data.
        
        Args:
            X: Input data
            y: Target data
            **kwargs: Additional arguments
            
        Returns:
            BaseFeatureEngineer: Fitted feature engineer
        """
        
    def transform(self, X, **kwargs):
        """
        Transform the input data.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Transformed data
        """
        
    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the feature engineer to the input data and transform it.
        
        Args:
            X: Input data
            y: Target data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Transformed data
        """
        
    def engineer_features(self, data, **kwargs):
        """
        Engineer features from the input data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Data with engineered features
        """
```

### Transformers

The feature engineering transformers provide specific transformations for different types of data.

```python
class TextCombiner(BaseColumnTransformer):
    """Combines multiple text columns into a single column."""
    
    def __init__(self, columns, separator=' ', new_column='combined_text', name=None):
        """
        Initialize the transformer.
        
        Args:
            columns: List of columns to combine
            separator: Separator to use between column values
            new_column: Name of the new column
            name: Transformer name
        """
        
    def _transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame: Transformed data
        """

class NumericCleaner(BaseColumnTransformer):
    """Cleans and standardizes numeric columns."""
    
    def __init__(self, column, new_name=None, fill_value=0, dtype='float', name=None):
        """
        Initialize the transformer.
        
        Args:
            column: Column to clean
            new_name: Name of the new column
            fill_value: Value to use for missing values
            dtype: Data type for the column
            name: Transformer name
        """
        
    def _transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame: Transformed data
        """

class HierarchyBuilder(BaseColumnTransformer):
    """Builds hierarchical features from parent columns."""
    
    def __init__(self, parent_columns, new_column='hierarchy', separator='/', name=None):
        """
        Initialize the transformer.
        
        Args:
            parent_columns: List of parent columns
            new_column: Name of the new column
            separator: Separator to use between parent values
            name: Transformer name
        """
        
    def _transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame: Transformed data
        """

class ColumnMapper(BaseColumnTransformer):
    """Maps columns from one name to another."""
    
    def __init__(self, source, target, name=None):
        """
        Initialize the transformer.
        
        Args:
            source: Source column name
            target: Target column name
            name: Transformer name
        """
        
    def _transform(self, X):
        """
        Transform the input data.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame: Transformed data
        """
```

### Feature Engineering Configuration

The feature engineering configuration components provide configuration-driven feature engineering.

```python
class ConfigDrivenFeatureEngineer(BaseFeatureEngineer):
    """Feature engineer driven by configuration."""
    
    def __init__(self, config=None):
        """
        Initialize the feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        
    def engineer_features(self, data, **kwargs):
        """
        Engineer features from the input data.
        
        Args:
            data: Input data
            **kwargs: Additional arguments
            
        Returns:
            DataFrame: Data with engineered features
        """

def register_transformer(name, transformer_class):
    """
    Register a transformer class with a name.
    
    Args:
        name: Transformer name
        transformer_class: Transformer class
    """

def create_transformer(name, **kwargs):
    """
    Create a transformer instance by name.
    
    Args:
        name: Transformer name
        **kwargs: Additional arguments for the transformer constructor
        
    Returns:
        BaseFeatureTransformer: Transformer instance
    """

def enhance_features(data, feature_engineer, **kwargs):
    """
    Enhance features using a fitted feature engineer.
    
    Args:
        data: Input data
        feature_engineer: Fitted feature engineer
        **kwargs: Additional arguments
        
    Returns:
        DataFrame: Data with enhanced features
    """
```

## Model Building

### Model Building Base Classes

The model building base classes provide the foundation for model building components.

```python
class BaseModelBuilder:
    """Base class for model builders."""
    
    def build_model(self, **kwargs):
        """
        Build a machine learning model.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            object: Model instance
        """
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """
        Optimize hyperparameters for the model.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Optimized model instance
        """

class BaseModelTrainer:
    """Base class for model trainers."""
    
    def train(self, model, x_train, y_train, **kwargs):
        """
        Train a model on the provided data.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Trained model instance
        """
        
    def cross_validate(self, model, x, y, **kwargs):
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model instance
            x: Features
            y: Targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Cross-validation results
        """

class BaseModelEvaluator:
    """Base class for model evaluators."""
    
    def evaluate(self, model, x_test, y_test, **kwargs):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model instance
            x_test: Test features
            y_test: Test targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Evaluation metrics
        """
        
    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        """
        Analyze model predictions in detail.
        
        Args:
            model: Trained model instance
            x_test: Test features
            y_test: Test targets
            y_pred: Predicted targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Analysis results
        """

class BaseModelSerializer:
    """Base class for model serializers."""
    
    def save_model(self, model, path, **kwargs):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model instance
            path: Path to save the model
            **kwargs: Additional arguments
        """
        
    def load_model(self, path, **kwargs):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            **kwargs: Additional arguments
            
        Returns:
            object: Loaded model instance
        """
```

### Model Builders

The model builders provide specific implementations for different types of models.

```python
class RandomForestBuilder(BaseModelBuilder):
    """Builds random forest models."""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """
        Initialize the model builder.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
        """
        
    def build_model(self, **kwargs):
        """
        Build a random forest model.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            object: Random forest model instance
        """
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """
        Optimize hyperparameters for the random forest model.
        
        Args:
            model: Random forest model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Optimized random forest model instance
        """

class GradientBoostingBuilder(BaseModelBuilder):
    """Builds gradient boosting models."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=None):
        """
        Initialize the model builder.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of the trees
            random_state: Random seed for reproducibility
        """
        
    def build_model(self, **kwargs):
        """
        Build a gradient boosting model.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            object: Gradient boosting model instance
        """
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """
        Optimize hyperparameters for the gradient boosting model.
        
        Args:
            model: Gradient boosting model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Optimized gradient boosting model instance
        """

class EnsembleBuilder(BaseModelBuilder):
    """Builds ensemble models."""
    
    def __init__(self, models=None):
        """
        Initialize the model builder.
        
        Args:
            models: List of models to include in the ensemble
        """
        
    def build_model(self, **kwargs):
        """
        Build an ensemble model.
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            object: Ensemble model instance
        """
        
    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """
        Optimize hyperparameters for the ensemble model.
        
        Args:
            model: Ensemble model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Optimized ensemble model instance
        """
```

### Model Trainers

The model trainers provide specific implementations for different training approaches.

```python
class StandardModelTrainer(BaseModelTrainer):
    """Standard model trainer."""
    
    def train(self, model, x_train, y_train, **kwargs):
        """
        Train a model on the provided data.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Trained model instance
        """
        
    def cross_validate(self, model, x, y, **kwargs):
        """
        Perform cross-validation on the model.
        
        Args:
            model: Model instance
            x: Features
            y: Targets
            **kwargs: Additional arguments
            
        Returns:
            dict: Cross-validation results
        """

class CrossValidationTrainer(BaseModelTrainer):
    """Model trainer with cross-validation."""
    
    def __init__(self, cv=5, scoring=None):
        """
        Initialize the model trainer.
        
        Args:
            cv: Number of cross-validation folds
            scoring: Scoring metric
        """
        
    def train(self, model, x_train, y_train, **kwargs):
        """
        Train a model on the provided data with cross-validation.
        
        Args:
            model: Model instance
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments
            
        Returns:
            object: Trained model instance
        """
        
    def cross_validate(self, model, x, y, **kwargs):
        """
        Perform cross-validation on the model.
        
        Args:
