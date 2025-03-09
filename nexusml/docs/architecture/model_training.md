# Model Training

## Overview

The model training system in NexusML handles the training, evaluation, and serialization of machine learning models. It provides interfaces for training models, evaluating their performance, and saving/loading them to/from disk.

## Diagram

The following diagram illustrates the model training system:

- [Model Training System](../../diagrams/nexusml/model_training.puml) - Components and relationships of the model training system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### ModelTrainer Interface

The main interface for model training components in the pipeline.

```python
class ModelTrainer:
    def train(self, model, x_train, y_train, **kwargs) -> Any:
        """Train a model on the provided data."""
        pass
        
    def cross_validate(self, model, x, y, **kwargs) -> Dict[str, List[float]]:
        """Perform cross-validation on the model."""
        pass
```

### ModelEvaluator Interface

The main interface for model evaluation components in the pipeline.

```python
class ModelEvaluator:
    def evaluate(self, model, x_test, y_test, **kwargs) -> Dict[str, float]:
        """Evaluate a trained model on test data."""
        pass
        
    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs) -> Dict[str, Any]:
        """Analyze model predictions in detail."""
        pass
```

### ModelSerializer Interface

The main interface for model serialization components in the pipeline.

```python
class ModelSerializer:
    def save_model(self, model, path, **kwargs) -> None:
        """Save a trained model to disk."""
        pass
        
    def load_model(self, path, **kwargs) -> Any:
        """Load a trained model from disk."""
        pass
```

## Model Trainers

### StandardModelTrainer

Standard implementation for training scikit-learn models.

```python
from nexusml.core.model_training.trainers import StandardModelTrainer

# Create trainer
trainer = StandardModelTrainer(
    early_stopping=True,
    class_weight="balanced",
    sample_weight=None
)

# Train model
trained_model = trainer.train(
    model, x_train, y_train
)

# Cross-validate model
cv_results = trainer.cross_validate(
    model, x, y, cv=5
)
```

### MultiTargetModelTrainer

Trains models for multiple target columns.

```python
from nexusml.core.model_training.trainers import MultiTargetModelTrainer
from nexusml.core.model_training.trainers import StandardModelTrainer

# Create base trainer
base_trainer = StandardModelTrainer()

# Create multi-target trainer
multi_trainer = MultiTargetModelTrainer(
    base_trainer=base_trainer,
    target_columns=["category_name", "mcaa_system_category"]
)

# Train models for all targets
trained_models = multi_trainer.train(
    models, x_train, y_train
)

# Cross-validate models
cv_results = multi_trainer.cross_validate(
    models, x, y, cv=5
)
```

## Model Evaluators

### StandardModelEvaluator

Standard implementation for evaluating scikit-learn models.

```python
from nexusml.core.model_training.evaluators import StandardModelEvaluator

# Create evaluator
evaluator = StandardModelEvaluator(
    metrics=["accuracy", "f1_macro", "precision", "recall"]
)

# Evaluate model
metrics = evaluator.evaluate(
    model, x_test, y_test
)

# Analyze predictions
analysis = evaluator.analyze_predictions(
    model, x_test, y_test, y_pred
)
```

### MultiTargetModelEvaluator

Evaluates models for multiple target columns.

```python
from nexusml.core.model_training.evaluators import MultiTargetModelEvaluator
from nexusml.core.model_training.evaluators import StandardModelEvaluator

# Create base evaluator
base_evaluator = StandardModelEvaluator()

# Create multi-target evaluator
multi_evaluator = MultiTargetModelEvaluator(
    base_evaluator=base_evaluator,
    target_columns=["category_name", "mcaa_system_category"]
)

# Evaluate models for all targets
metrics = multi_evaluator.evaluate(
    models, x_test, y_test
)

# Analyze predictions
analysis = multi_evaluator.analyze_predictions(
    models, x_test, y_test, y_pred
)
```

## Model Serializers

### PickleModelSerializer

Serializes models using Python's pickle module.

```python
from nexusml.core.model_training.serializers import PickleModelSerializer

# Create serializer
serializer = PickleModelSerializer(protocol=4)

# Save model
serializer.save_model(
    model, "outputs/models/model.pkl"
)

# Load model
loaded_model = serializer.load_model(
    "outputs/models/model.pkl"
)
```

### JobLibModelSerializer

Serializes models using joblib, which is more efficient for large NumPy arrays.

```python
from nexusml.core.model_training.serializers import JobLibModelSerializer

# Create serializer
serializer = JobLibModelSerializer(compress=3)

# Save model
serializer.save_model(
    model, "outputs/models/model.joblib"
)

# Load model
loaded_model = serializer.load_model(
    "outputs/models/model.joblib"
)
```

## Training Process

The model training process follows these steps:

1. **Model Preparation**: Prepare the model for training
2. **Training**: Fit the model to the training data
3. **Evaluation**: Evaluate the model on test data
4. **Analysis**: Analyze model predictions
5. **Serialization**: Save the model to disk

```python
def train_model(model, x_train, y_train, x_test, y_test, output_path):
    """Train, evaluate, and save a model."""
    # Create components
    trainer = StandardModelTrainer()
    evaluator = StandardModelEvaluator()
    serializer = PickleModelSerializer()
    
    # Train model
    trained_model = trainer.train(model, x_train, y_train)
    
    # Evaluate model
    metrics = evaluator.evaluate(trained_model, x_test, y_test)
    
    # Make predictions
    y_pred = trained_model.predict(x_test)
    
    # Analyze predictions
    analysis = evaluator.analyze_predictions(
        trained_model, x_test, y_test, y_pred
    )
    
    # Save model
    serializer.save_model(trained_model, output_path)
    
    return trained_model, metrics, analysis
```

## Callbacks

The training system supports callbacks for monitoring and controlling the training process.

### ModelTrainingCallback

Base class for all training callbacks.

```python
class ModelTrainingCallback:
    def on_training_start(self, model, x_train, y_train, **kwargs):
        """Called at the start of training."""
        pass
        
    def on_training_end(self, model, x_train, y_train, **kwargs):
        """Called at the end of training."""
        pass
        
    def on_epoch_start(self, epoch, logs, **kwargs):
        """Called at the start of each epoch."""
        pass
        
    def on_epoch_end(self, epoch, logs, **kwargs):
        """Called at the end of each epoch."""
        pass
```

### EarlyStoppingCallback

Stops training when a monitored metric stops improving.

```python
from nexusml.core.model_training.callbacks import EarlyStoppingCallback

# Create callback
early_stopping = EarlyStoppingCallback(
    patience=10,
    min_delta=0.001,
    monitor="val_loss"
)

# Use in training
trainer = StandardModelTrainer()
trained_model = trainer.train(
    model, x_train, y_train,
    callbacks=[early_stopping]
)
```

## Model Cards

The model training system includes support for generating model cards, which document model details, performance, and usage.

### ModelCardGenerator

Generates model cards for model documentation and governance.

```python
from nexusml.core.model_card.generator import ModelCardGenerator

# Create generator
generator = ModelCardGenerator(
    template_path="templates/model_card_template.md",
    output_dir="outputs/model_cards"
)

# Generate model card
model_card = generator.generate_model_card(
    model=model,
    metrics=metrics,
    config=config,
    model_name="Equipment Classifier",
    model_version="1.0.0",
    model_type="Random Forest",
    author="NexusML Team",
    description="Classifies equipment into standardized categories",
    intended_use="Equipment classification for facility management",
    training_data="equipment_data.csv",
    evaluation_data="test_data.csv",
    ethical_considerations="No personally identifiable information used",
    caveats_and_recommendations="Model performs best on HVAC equipment"
)

# Save model card
generator.save_model_card(
    model_card, "outputs/model_cards/equipment_classifier_v1.md"
)
```

## Configuration

Model training can be configured through:

### YAML Configuration

```yaml
# training_config.yml
model_training:
  early_stopping: true
  class_weight: "balanced"
  sample_weight: null
  callbacks:
    - type: early_stopping
      patience: 10
      min_delta: 0.001
      monitor: val_loss
  evaluation:
    metrics:
      - accuracy
      - f1_macro
      - precision
      - recall
  serialization:
    format: pickle
    protocol: 4
```

### Code Configuration

```python
# Create components with explicit configuration
trainer = StandardModelTrainer(
    early_stopping=True,
    class_weight="balanced"
)

evaluator = StandardModelEvaluator(
    metrics=["accuracy", "f1_macro", "precision", "recall"]
)

serializer = PickleModelSerializer(protocol=4)

# Or use configuration-driven approach
from nexusml.core.config.provider import ConfigProvider

# Initialize with custom config
ConfigProvider.initialize("path/to/training_config.yml")

# Get configuration
config = ConfigProvider.get_config().model_training

# Create components based on configuration
# ...
```

## Custom Trainers

You can create custom trainers by implementing the ModelTrainer interface:

```python
from nexusml.core.model_training.base import BaseModelTrainer

class CustomModelTrainer(BaseModelTrainer):
    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def train(self, model, x_train, y_train, **kwargs):
        # Custom training logic
        # ...
        
        return trained_model
        
    def cross_validate(self, model, x, y, **kwargs):
        # Custom cross-validation logic
        # ...
        
        return cv_results
```

## Pipeline Integration

Model training is integrated into the pipeline system:

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

# Train model with training configuration
model, metrics = orchestrator.train_model(
    data_path="data.csv",
    model_training_params={
        "early_stopping": True,
        "class_weight": "balanced",
        "evaluation": {
            "metrics": ["accuracy", "f1_macro", "precision", "recall"]
        }
    }
)
```

## Advanced Usage

### Custom Metrics

```python
from nexusml.core.model_training.evaluators import StandardModelEvaluator
from sklearn.metrics import fbeta_score

def f2_score(y_true, y_pred):
    """Calculate F2 score (emphasizes recall over precision)."""
    return fbeta_score(y_true, y_pred, beta=2, average="macro")

# Create evaluator with custom metrics
evaluator = StandardModelEvaluator(
    metrics=["accuracy", "f1_macro", "precision", "recall", f2_score]
)

# Evaluate model
metrics = evaluator.evaluate(model, x_test, y_test)
```

### Training with Sample Weights

```python
import numpy as np
from nexusml.core.model_training.trainers import StandardModelTrainer

# Create custom sample weights
def create_sample_weights(y_train):
    """Create sample weights based on class frequency."""
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    return sample_weights

# Create trainer
trainer = StandardModelTrainer()

# Train model with sample weights
trained_model = trainer.train(
    model, x_train, y_train,
    sample_weight=create_sample_weights(y_train)
)
```

### Custom Callbacks

```python
from nexusml.core.model_training.callbacks import ModelTrainingCallback
import matplotlib.pyplot as plt

class LearningCurveCallback(ModelTrainingCallback):
    def __init__(self, output_path):
        self.output_path = output_path
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        
    def on_epoch_end(self, epoch, logs, **kwargs):
        self.epochs.append(epoch)
        self.train_losses.append(logs.get("train_loss", 0))
        self.val_losses.append(logs.get("val_loss", 0))
        
    def on_training_end(self, model, x_train, y_train, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label="Training Loss")
        plt.plot(self.epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.savefig(self.output_path)
        plt.close()

# Create callback
learning_curve = LearningCurveCallback("outputs/learning_curve.png")

# Use in training
trainer = StandardModelTrainer()
trained_model = trainer.train(
    model, x_train, y_train,
    callbacks=[learning_curve]
)