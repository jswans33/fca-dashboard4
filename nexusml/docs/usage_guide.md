# NexusML Usage Guide

This guide provides comprehensive documentation for using NexusML, a Python machine learning package for equipment classification. It covers all aspects of using the package, from basic to advanced usage.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Building and Training](#model-building-and-training)
7. [Prediction and Evaluation](#prediction-and-evaluation)
8. [Pipeline Orchestration](#pipeline-orchestration)
9. [Configuration](#configuration)
10. [Command-Line Tools](#command-line-tools)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)

## Introduction

NexusML is a Python machine learning package designed for equipment classification. It uses machine learning techniques to categorize equipment into standardized classification systems like MasterFormat and OmniClass based on textual descriptions and metadata.

### Key Features

- **Data Loading and Preprocessing**: Load data from various sources and preprocess it for machine learning
- **Feature Engineering**: Transform raw data into features suitable for machine learning using a configurable pipeline
- **Model Training**: Train machine learning models for equipment classification with support for hyperparameter optimization
- **Model Evaluation**: Evaluate model performance with comprehensive metrics and analysis tools
- **Prediction**: Make predictions on new equipment data with both batch and single-item support
- **Configuration**: Centralized configuration system with validation and environment variable support
- **Dependency Injection**: Flexible component system with dependency management
- **Pipeline Architecture**: Modular pipeline system for customizable ML workflows
- **Model Cards**: Generate standardized model cards for model documentation and governance

### Use Cases

NexusML is particularly useful for:

- Classifying equipment based on textual descriptions
- Mapping equipment to standardized classification systems
- Extracting attributes from equipment descriptions
- Generating standardized descriptions for equipment
- Validating equipment data against reference standards

## Installation

For detailed installation instructions, see the [Installation Guide](installation_guide.md).

### Quick Installation

```bash
# Basic installation
pip install nexusml

# With AI features
pip install "nexusml[ai]"

# Development installation
git clone https://github.com/your-org/nexusml.git
cd nexusml
pip install -e ".[dev]"
```

## Basic Usage

### Training a Model

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create the pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)

# Train a model
model, metrics = orchestrator.train_model(
    data_path="path/to/training_data.csv",
    test_size=0.3,
    random_state=42,
    optimize_hyperparameters=True,
    output_dir="outputs/models",
    model_name="equipment_classifier",
)

# Print metrics
print("Model training completed successfully")
print("Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

### Making Predictions

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create the pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)

# Load a trained model
model = orchestrator.load_model("outputs/models/equipment_classifier.pkl")

# Make predictions
predictions = orchestrator.predict(
    model=model,
    data_path="path/to/prediction_data.csv",
    output_path="outputs/predictions.csv",
)

# Print predictions
print("Predictions completed successfully")
print("Sample predictions:")
print(predictions.head())
```

### Using Command-Line Tools

```bash
# Train a model
python -m nexusml.train_model_pipeline_v2 --data-path path/to/training_data.csv --optimize

# Make predictions
python -m nexusml.predict_v2 --model-path outputs/models/equipment_classifier_latest.pkl --input-file path/to/prediction_data.csv

# Classify equipment from any input format
python -m nexusml.classify_equipment path/to/input_file.csv --output path/to/output_file.json
```

## Data Loading and Preprocessing

### Data Loading

NexusML supports loading data from various sources:

```python
from nexusml.core.pipeline.components.data_loader import StandardDataLoader

# Create a data loader
data_loader = StandardDataLoader()

# Load data from a CSV file
data = data_loader.load_data("path/to/data.csv")

# Load data from an Excel file
data = data_loader.load_data("path/to/data.xlsx")

# Load data with automatic discovery
data = data_loader.load_data(discover_files=True)
```

### Data Preprocessing

NexusML provides preprocessing capabilities for cleaning and preparing data:

```python
from nexusml.core.pipeline.components.data_preprocessor import StandardPreprocessor

# Create a preprocessor
preprocessor = StandardPreprocessor()

# Preprocess data
preprocessed_data = preprocessor.preprocess(data)

# Verify required columns
preprocessor.verify_required_columns(preprocessed_data)
```

### Data Validation

NexusML includes validation capabilities for ensuring data quality:

```python
from nexusml.core.validation import (
    ValidationRule,
    ColumnExistenceRule,
    NonNullRule,
    ValueRangeRule,
    BaseValidator,
)

# Create a validator with multiple rules
validator = BaseValidator("DataValidator")
validator.add_rule(ColumnExistenceRule('description'))
validator.add_rule(ColumnExistenceRule('service_life'))
validator.add_rule(NonNullRule('description'))
validator.add_rule(ValueRangeRule('service_life', min_value=0, max_value=100))

# Validate data
report = validator.validate(data)
print(f"Validation passed: {report.is_valid()}")
```

## Feature Engineering

### Basic Feature Engineering

NexusML provides feature engineering capabilities for transforming raw data into features suitable for machine learning:

```python
from nexusml.core.feature_engineering import (
    TextCombiner,
    NumericCleaner,
    HierarchyBuilder,
    BaseFeatureEngineer,
)

# Create a feature engineer
feature_engineer = BaseFeatureEngineer()

# Add transformers
feature_engineer.add_transformer(TextCombiner(
    columns=['Asset Category', 'Equip Name ID'],
    separator=' - ',
    new_column='Equipment_Type'
))
feature_engineer.add_transformer(NumericCleaner(
    column='Service Life',
    new_name='service_life_years',
    fill_value=0,
    dtype='int'
))
feature_engineer.add_transformer(HierarchyBuilder(
    parent_columns=['Asset Category', 'Equip Name ID'],
    new_column='Equipment_Hierarchy',
    separator='/'
))

# Apply feature engineering
features = feature_engineer.fit_transform(data)
```

### Configuration-Driven Feature Engineering

NexusML supports configuration-driven feature engineering:

```python
from nexusml.core.feature_engineering import ConfigDrivenFeatureEngineer

# Create a configuration
config = {
    "text_combinations": [
        {
            "columns": ["Asset Category", "Equip Name ID"],
            "separator": " - ",
            "name": "Equipment_Type"
        }
    ],
    "numeric_columns": [
        {
            "name": "Service Life",
            "new_name": "service_life_years",
            "fill_value": 0,
            "dtype": "int"
        }
    ],
    "hierarchies": [
        {
            "parents": ["Asset Category", "Equip Name ID"],
            "new_col": "Equipment_Hierarchy",
            "separator": "/"
        }
    ],
    "column_mappings": [
        {
            "source": "Manufacturer",
            "target": "equipment_manufacturer"
        },
        {
            "source": "Model",
            "target": "equipment_model"
        }
    ]
}

# Create a configuration-driven feature engineer
config_driven_fe = ConfigDrivenFeatureEngineer(config=config)

# Apply feature engineering
features = config_driven_fe.fit_transform(data)
```

### Custom Transformers

NexusML allows creating custom transformers for specific feature engineering needs:

```python
from nexusml.core.feature_engineering import BaseColumnTransformer, register_transformer

# Define a custom transformer
class ManufacturerNormalizer(BaseColumnTransformer):
    """
    Normalizes manufacturer names by converting to uppercase and removing special characters.
    """
    
    def __init__(
        self,
        column: str = "Manufacturer",
        new_column: str = "normalized_manufacturer",
        name: str = "ManufacturerNormalizer",
    ):
        """Initialize the manufacturer normalizer."""
        super().__init__([column], [new_column], name)
        self.column = column
        self.new_column = new_column
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize manufacturer names."""
        import re
        
        # Convert to uppercase and remove special characters
        X[self.new_column] = X[self.column].str.upper()
        X[self.new_column] = X[self.new_column].apply(
            lambda x: re.sub(r'[^A-Z0-9]', '', x) if isinstance(x, str) else x
        )
        
        return X

# Register the custom transformer
register_transformer("manufacturer_normalizer", ManufacturerNormalizer)

# Create an instance of the custom transformer
manufacturer_normalizer = create_transformer("manufacturer_normalizer")

# Apply the custom transformer
df_transformed = manufacturer_normalizer.fit_transform(df)
```

## Model Building and Training

### Building Models

NexusML provides model building capabilities for creating and configuring machine learning models:

```python
from nexusml.core.model_building import RandomForestBuilder

# Create a model builder
model_builder = RandomForestBuilder(n_estimators=100)

# Build a model
model = model_builder.build_model()
```

### Training Models

NexusML provides model training capabilities:

```python
from nexusml.core.model_training import StandardModelTrainer

# Create a model trainer
trainer = StandardModelTrainer()

# Train the model
trained_model = trainer.train(model, X_train, y_train)
```

### Hyperparameter Optimization

NexusML supports hyperparameter optimization:

```python
from nexusml.core.model_building import RandomForestBuilder

# Create a model builder
model_builder = RandomForestBuilder()

# Build a model
model = model_builder.build_model()

# Optimize hyperparameters
optimized_model = model_builder.optimize_hyperparameters(model, X_train, y_train)
```

### Model Evaluation

NexusML provides model evaluation capabilities:

```python
from nexusml.core.model_building.base import BaseModelEvaluator

# Create a model evaluator
evaluator = BaseModelEvaluator()

# Evaluate the model
metrics = evaluator.evaluate(trained_model, X_test, y_test)
print(f"Overall accuracy: {metrics['overall']['accuracy_mean']:.4f}")
print(f"Overall F1 score: {metrics['overall']['f1_macro_mean']:.4f}")
```

### Model Serialization

NexusML provides model serialization capabilities:

```python
from nexusml.core.model_building.base import BaseModelSerializer

# Create a model serializer
serializer = BaseModelSerializer()

# Save the model
serializer.save_model(trained_model, "outputs/models/best_model.pkl")

# Load the model
loaded_model = serializer.load_model("outputs/models/best_model.pkl")
```

## Prediction and Evaluation

### Making Predictions

NexusML provides prediction capabilities:

```python
from nexusml.core.pipeline.components.predictor import StandardPredictor

# Create a predictor
predictor = StandardPredictor()

# Make predictions
predictions = predictor.predict(model, data)
```

### Batch Prediction

NexusML supports batch prediction:

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator

# Create an orchestrator
orchestrator = PipelineOrchestrator(factory, context)

# Make batch predictions
predictions = orchestrator.predict(
    model=model,
    data_path="path/to/prediction_data.csv",
    output_path="outputs/predictions.csv",
)
```

### Prediction Analysis

NexusML provides prediction analysis capabilities:

```python
from nexusml.core.model_building.base import BaseModelEvaluator

# Create a model evaluator
evaluator = BaseModelEvaluator()

# Analyze predictions
analysis = evaluator.analyze_predictions(model, X_test, y_test, y_pred)
```

## Pipeline Orchestration

### Creating a Pipeline

NexusML provides pipeline orchestration capabilities:

```python
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.di.container import DIContainer

# Create the pipeline components
registry = ComponentRegistry()
container = DIContainer()
factory = PipelineFactory(registry, container)
context = PipelineContext()
orchestrator = PipelineOrchestrator(factory, context)
```

### Training Pipeline

NexusML provides a training pipeline:

```python
# Train a model
model, metrics = orchestrator.train_model(
    data_path="path/to/training_data.csv",
    test_size=0.3,
    random_state=42,
    optimize_hyperparameters=True,
    output_dir="outputs/models",
    model_name="equipment_classifier",
)
```

### Prediction Pipeline

NexusML provides a prediction pipeline:

```python
# Make predictions
predictions = orchestrator.predict(
    model=model,
    data_path="path/to/prediction_data.csv",
    output_path="outputs/predictions.csv",
)
```

### Evaluation Pipeline

NexusML provides an evaluation pipeline:

```python
# Evaluate a model
results = orchestrator.evaluate(
    model=model,
    data_path="path/to/evaluation_data.csv",
    output_path="outputs/evaluation_results.json",
)
```

### Pipeline Stages

NexusML supports pipeline stages for more granular control:

```python
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages import (
    ConfigurableDataLoadingStage,
    ConfigDrivenValidationStage,
    SimpleFeatureEngineeringStage,
    RandomSplittingStage,
    ConfigDrivenModelBuildingStage,
    StandardModelTrainingStage,
    ClassificationEvaluationStage,
    ModelCardSavingStage,
    StandardPredictionStage,
)

# Create a pipeline context
context = PipelineContext()
context.start()

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

# Execute each stage
for stage in stages:
    stage.execute(context)
```

## Configuration

### Configuration System

NexusML uses a centralized configuration system:

```python
from nexusml.core.config.provider import ConfigProvider

# Get configuration
config = ConfigProvider.get_config()

# Access configuration sections
feature_config = config.feature_engineering
model_config = config.model_building
data_config = config.data_loading
```

### Configuration Files

NexusML supports YAML configuration files:

```yaml
# nexusml_config.yml
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
    numerical:
      type: standard_scaler
    categorical:
      type: one_hot_encoder

model_building:
  model_type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
  optimization:
    method: grid_search
    cv: 5
    scoring: f1_macro

data_loading:
  encoding: utf-8
  delimiter: ","
  required_columns:
    - description
    - service_life
    - category

paths:
  output_dir: outputs
  models_dir: outputs/models
  data_dir: data
```

### Environment Variables

NexusML supports environment variable overrides:

```bash
# Override feature engineering settings
export NEXUSML_FEATURE_ENGINEERING_TEXT_COLUMNS=description,name,manufacturer
export NEXUSML_MODEL_BUILDING_HYPERPARAMETERS_N_ESTIMATORS=200
export NEXUSML_PATHS_OUTPUT_DIR=/custom/output/path
```

## Command-Line Tools

### Training a Model

```bash
# Train a model using the original pipeline
python -m nexusml.train_model_pipeline --data-path path/to/training_data.csv

# Train a model using the new pipeline architecture
python -m nexusml.train_model_pipeline_v2 --data-path path/to/training_data.csv --optimize
```

### Making Predictions

```bash
# Make predictions using the original pipeline
python -m nexusml.predict --model-path outputs/models/equipment_classifier.pkl --input-file path/to/prediction_data.csv

# Make predictions using the new pipeline architecture
python -m nexusml.predict_v2 --model-path outputs/models/equipment_classifier_latest.pkl --input-file path/to/prediction_data.csv
```

### Classifying Equipment

```bash
# Classify equipment from any input format
python -m nexusml.classify_equipment path/to/input_file.csv --output path/to/output_file.json
```

### Validating Reference Data

```bash
# Validate reference data
python -m nexusml.test_reference_validation
```

## Advanced Usage

### Custom Components

NexusML supports creating custom components:

```python
from nexusml.core.pipeline.interfaces import DataLoader
from nexusml.core.pipeline.registry import ComponentRegistry

# Define a custom data loader
class CustomDataLoader(DataLoader):
    """Custom data loader for specific file formats."""
    
    def __init__(self, file_path=None):
        self.file_path = file_path
    
    def load_data(self, data_path=None, **kwargs):
        """Load data from a custom file format."""
        path = data_path or self.file_path
        # Custom loading logic
        return data
    
    def get_config(self):
        """Get the configuration for the data loader."""
        return {"file_path": self.file_path}

# Register the custom component
registry = ComponentRegistry()
registry.register(DataLoader, "custom", CustomDataLoader)
registry.set_default_implementation(DataLoader, "custom")
```

### Dependency Injection

NexusML uses dependency injection for component management:

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.di.decorators import inject

# Create a container
container = DIContainer()

# Register a component
container.register(DataLoader, CustomDataLoader())

# Use dependency injection
@inject
def process_data(data_loader=Inject(DataLoader)):
    data = data_loader.load_data("path/to/data.csv")
    return data
```

### Model Cards

NexusML supports model cards for model documentation and governance:

```python
from nexusml.core.model_card.generator import ModelCardGenerator

# Create a model card generator
generator = ModelCardGenerator()

# Generate a model card
model_card = generator.generate_model_card(
    model=model,
    model_name="Equipment Classifier",
    model_version="1.0.0",
    model_description="A random forest model for classifying equipment based on descriptions.",
    model_authors=["NexusML Team"],
    model_license="Proprietary",
    metrics=metrics,
    training_data_info={
        "source": "path/to/training_data.csv",
        "size": len(X_train) + len(X_test),
        "features": list(X_train.columns),
        "targets": list(y_train.columns),
    },
)

# Save the model card
model_card.save("outputs/model_cards/equipment_classifier.json")
```

### Domain-Specific Functionality

#### OmniClass Generator

NexusML includes an OmniClass generator:

```python
from nexusml import (
    OmniClassDescriptionGenerator,
    extract_omniclass_data,
    generate_descriptions,
)

# Extract OmniClass data from Excel files
df = extract_omniclass_data(input_dir="files/omniclass_tables", output_file="omniclass.csv", file_pattern="*.xlsx")

# Generate descriptions for OmniClass codes
result_df = generate_descriptions(
    input_file="omniclass.csv",
    output_file="omniclass_with_descriptions.csv",
    start_index=0,
    end_index=5,
    batch_size=5,
    description_column="Description",
)
```

#### Uniformat Keywords

NexusML includes Uniformat keyword functionality:

```python
from nexusml.core.reference.manager import ReferenceManager

# Initialize the reference manager
ref_manager = ReferenceManager()

# Load all reference data
ref_manager.load_all()

# Find Uniformat codes by keyword
results = ref_manager.find_uniformat_codes_by_keyword("Boilers")

# Enrich equipment data with Uniformat and MasterFormat information
enriched_df = ref_manager.enrich_equipment_data(df)
```

## Troubleshooting and Best Practices

### Common Issues

#### Package Not Found

If you encounter a "Package not found" error:

```
ERROR: Could not find a version that satisfies the requirement nexusml
ERROR: No matching distribution found for nexusml
```

Solutions:
- Ensure you're using Python 3.8 or higher
- Update pip: `pip install --upgrade pip`
- Check your internet connection
- If installing from a private repository, ensure you have proper authentication

#### Dependency Conflicts

If you encounter dependency conflicts:

```
ERROR: Cannot install nexusml due to conflicting dependencies
```

Solutions:
- Use a virtual environment for a clean installation
- Try installing with the `--ignore-installed` flag: `pip install --ignore-installed nexusml`
- Check for conflicting packages in your environment

#### Import Errors After Installation

If you can install but encounter import errors:

```python
>>> import nexusml
ImportError: No module named 'nexusml'
```

Solutions:
- Ensure you're using the same Python environment where you installed the package
- Check if the package is installed: `pip list | grep nexusml`
- Try reinstalling: `pip uninstall nexusml && pip install nexusml`

### Best Practices

#### Project Structure

Organize your project with a clear structure:

```
project/
├── config/
│   └── nexusml_config.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── equipment_classifier.pkl
├── outputs/
│   ├── predictions.csv
│   └── model_cards/
├── scripts/
│   ├── train.py
│   └── predict.py
└── README.md
```

#### Configuration Management

Use a centralized configuration file:

```yaml
# config/nexusml_config.yml
feature_engineering:
  # Feature engineering configuration
model_building:
  # Model building configuration
data_loading:
  # Data loading configuration
paths:
  # Path configuration
```

#### Error Handling

Implement proper error handling:

```python
try:
    # NexusML operations
    model, metrics = orchestrator.train_model(
        data_path="path/to/training_data.csv",
        test_size=0.3,
        random_state=42,
    )
except Exception as e:
    print(f"Error: {e}")
    # Handle error
```

#### Logging

Use logging for tracking operations:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("nexusml.log"),
        logging.StreamHandler(),
    ],
)

# Get logger
logger = logging.getLogger("nexusml")

# Log operations
logger.info("Training model...")
model, metrics = orchestrator.train_model(
    data_path="path/to/training_data.csv",
    test_size=0.3,
    random_state=42,
)
logger.info("Model training completed successfully")
```

#### Performance Optimization

Optimize performance for large datasets:

- Use batch processing for large datasets
- Use parallel processing for computationally intensive operations
- Use memory-efficient data structures
- Use incremental learning for very large datasets

#### Model Governance

Implement model governance practices:

- Use model cards for model documentation
- Track model versions and changes
- Validate models before deployment
- Monitor model performance in production
- Implement model retraining procedures

## Next Steps

After reading this guide, you might want to:

1. Explore the [Examples](examples/README.md) for practical usage examples
2. Check the [API Reference](api_reference.md) for detailed information on classes and methods
3. Read the [Architecture Documentation](architecture/README.md) for a deeper understanding of the system design