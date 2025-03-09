# Prediction

## Overview

The prediction system in NexusML handles making predictions with trained models. It provides interfaces for predicting with models, formatting predictions, and explaining prediction results.

## Diagram

The following diagram illustrates the prediction system:

- [Prediction System](../../diagrams/nexusml/prediction.puml) - Components and relationships of the prediction system

To render this diagram, use the PlantUML utilities as described in [SOP-004](../../SOPs/004-plantuml-utilities.md):

```bash
python -m fca_dashboard.utils.puml.cli render
```

## Key Components

### Predictor Interface

The main interface for prediction components in the pipeline.

```python
class Predictor:
    def predict(self, model, data, **kwargs) -> pd.DataFrame:
        """Make predictions using a trained model."""
        pass
        
    def predict_proba(self, model, data, **kwargs) -> Dict[str, pd.DataFrame]:
        """Make probability predictions using a trained model."""
        pass
```

### StandardPredictor

Standard implementation for making predictions with scikit-learn models.

```python
from nexusml.core.prediction import StandardPredictor

# Create predictor
predictor = StandardPredictor(
    threshold=0.5,
    output_columns=["category_name", "mcaa_system_category"]
)

# Make predictions
predictions = predictor.predict(
    model, data
)

# Get prediction probabilities
probabilities = predictor.predict_proba(
    model, data
)
```

### MultiTargetPredictor

Makes predictions for multiple target columns.

```python
from nexusml.core.prediction import MultiTargetPredictor
from nexusml.core.prediction import StandardPredictor

# Create base predictor
base_predictor = StandardPredictor()

# Create multi-target predictor
multi_predictor = MultiTargetPredictor(
    base_predictor=base_predictor,
    target_columns=["category_name", "mcaa_system_category"]
)

# Make predictions for all targets
predictions = multi_predictor.predict(
    models, data
)

# Access predictions for specific target
category_predictions = predictions["category_name"]
system_predictions = predictions["mcaa_system_category"]
```

### BatchPredictor

Makes predictions in batches to handle large datasets.

```python
from nexusml.core.prediction import BatchPredictor
from nexusml.core.prediction import StandardPredictor

# Create base predictor
base_predictor = StandardPredictor()

# Create batch predictor
batch_predictor = BatchPredictor(
    predictor=base_predictor,
    batch_size=1000
)

# Make predictions in batches
predictions = batch_predictor.predict(
    model, large_dataset
)
```

### PredictionPipeline

Coordinates the prediction process from loading to output.

```python
from nexusml.core.prediction import PredictionPipeline
from nexusml.core.model_training.serializers import PickleModelSerializer
from nexusml.core.data_loading import CSVDataLoader
from nexusml.core.feature_engineering import GenericFeatureEngineer
from nexusml.core.prediction import StandardPredictor

# Create pipeline components
model_serializer = PickleModelSerializer()
data_loader = CSVDataLoader()
feature_engineer = GenericFeatureEngineer()
predictor = StandardPredictor()

# Create prediction pipeline
pipeline = PredictionPipeline(
    model_serializer, data_loader, feature_engineer, predictor
)

# Make predictions from file
predictions = pipeline.predict_from_file(
    model_path="models/classifier.pkl",
    data_path="data/new_data.csv",
    output_path="outputs/predictions.csv"
)

# Make predictions from data
predictions = pipeline.predict_from_data(
    model, data,
    output_path="outputs/predictions.csv"
)
```

## Helper Classes

### PredictionFormatter

Formats prediction results for output.

```python
from nexusml.core.prediction import PredictionFormatter

# Create formatter
formatter = PredictionFormatter(
    column_mapping={
        "category_name": "Equipment Category",
        "mcaa_system_category": "System Type"
    },
    include_probabilities=True
)

# Format predictions
formatted_predictions = formatter.format_predictions(
    predictions, probabilities
)
```

### PredictionExplainer

Provides explanations for model predictions.

```python
from nexusml.core.prediction import PredictionExplainer

# Create explainer
explainer = PredictionExplainer(
    feature_names=["description", "service_life", "category"]
)

# Explain a single prediction
explanation = explainer.explain_prediction(
    model, data, prediction_idx=0
)

# Explain multiple predictions
explanations = explainer.explain_predictions(
    model, data, prediction_indices=[0, 1, 2]
)
```

### PredictionValidator

Validates prediction results.

```python
from nexusml.core.prediction import PredictionValidator

# Create validator
validator = PredictionValidator(
    required_columns=["category_name", "mcaa_system_category"],
    value_ranges={
        "confidence": (0.0, 1.0)
    }
)

# Validate predictions
validation_results = validator.validate_predictions(predictions)
```

## Prediction Process

The prediction process follows these steps:

1. **Model Loading**: Load the trained model from disk
2. **Data Loading**: Load the data to predict
3. **Feature Engineering**: Transform the data into features
4. **Prediction**: Make predictions with the model
5. **Formatting**: Format the predictions for output
6. **Validation**: Validate the prediction results
7. **Output**: Save or return the predictions

```python
def predict_from_file(self, model_path, data_path, output_path=None, **kwargs):
    """Make predictions from a file."""
    # Load model
    model = self._load_model(model_path)
    
    # Load data
    data = self._load_data(data_path)
    
    # Engineer features
    features = self._engineer_features(data)
    
    # Make predictions
    predictions = self._predictor.predict(model, features, **kwargs)
    
    # Save predictions if output path is provided
    if output_path:
        self.save_predictions(predictions, output_path, **kwargs)
    
    return predictions
```

## Command-Line Prediction

NexusML provides command-line tools for making predictions:

### predict.py

Original prediction script.

```bash
python -m nexusml.predict \
    --model-path outputs/models/equipment_classifier_latest.pkl \
    --input-file data/new_data.csv \
    --output-file outputs/predictions.csv \
    --description-column "Description" \
    --service-life-column "Service Life" \
    --asset-tag-column "Asset Tag"
```

### predict_v2.py

Updated prediction script using the pipeline architecture.

```bash
python -m nexusml.predict_v2 \
    --model-path outputs/models/equipment_classifier_latest.pkl \
    --input-file data/new_data.csv \
    --output-file outputs/predictions.csv \
    --description-column "Description" \
    --service-life-column "Service Life" \
    --asset-tag-column "Asset Tag" \
    --use-orchestrator  # Use the new pipeline architecture
```

### classify_equipment.py

Modular equipment classification script.

```bash
python -m nexusml.classify_equipment \
    data/equipment_data.csv \
    --output outputs/classified_equipment.json \
    --config config/classification_config.yml
```

## Configuration

Prediction can be configured through:

### YAML Configuration

```yaml
# prediction_config.yml
prediction:
  threshold: 0.5
  output_columns:
    - category_name
    - mcaa_system_category
    - Equipment_Type
  batch_size: 1000
  include_probabilities: true
  column_mapping:
    category_name: "Equipment Category"
    mcaa_system_category: "System Type"
    Equipment_Type: "Equipment Type"
```

### Code Configuration

```python
# Create predictor with explicit configuration
predictor = StandardPredictor(
    threshold=0.5,
    output_columns=["category_name", "mcaa_system_category", "Equipment_Type"]
)

# Or use configuration-driven approach
from nexusml.core.config.provider import ConfigProvider

# Initialize with custom config
ConfigProvider.initialize("path/to/prediction_config.yml")

# Get configuration
config = ConfigProvider.get_config().prediction

# Create predictor
predictor = StandardPredictor(
    threshold=config.threshold,
    output_columns=config.output_columns
)
```

## Custom Predictors

You can create custom predictors by implementing the Predictor interface:

```python
from nexusml.core.prediction.base import BasePredictor
import pandas as pd

class CustomPredictor(BasePredictor):
    def __init__(self, threshold=0.5, output_columns=None):
        self.threshold = threshold
        self.output_columns = output_columns or []
        
    def predict(self, model, data, **kwargs):
        # Validate model and data
        self._validate_model(model)
        self._validate_data(data)
        
        # Make predictions
        predictions = model.predict(data)
        
        # Format predictions
        result = self._format_predictions(predictions, self.output_columns)
        
        return result
        
    def predict_proba(self, model, data, **kwargs):
        # Validate model and data
        self._validate_model(model)
        self._validate_data(data)
        
        # Make probability predictions
        probabilities = model.predict_proba(data)
        
        # Format probabilities
        result = {}
        for i, col in enumerate(self.output_columns):
            result[col] = pd.DataFrame(
                probabilities[i],
                columns=[f"{col}_{c}" for c in model.classes_[i]]
            )
        
        return result
        
    def _format_predictions(self, predictions, column_names):
        # Create DataFrame from predictions
        result = pd.DataFrame()
        
        # Add prediction columns
        for i, col in enumerate(column_names):
            result[col] = predictions[:, i] if len(predictions.shape) > 1 else predictions
            
        return result
```

## Pipeline Integration

Prediction is integrated into the pipeline system:

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

# Make predictions
predictions = orchestrator.predict(
    model_path="models/classifier.pkl",
    data_path="data/new_data.csv",
    output_path="outputs/predictions.csv",
    prediction_params={
        "threshold": 0.5,
        "output_columns": ["category_name", "mcaa_system_category"],
        "include_probabilities": True
    }
)
```

## Advanced Usage

### Prediction with Confidence Scores

```python
from nexusml.core.prediction import StandardPredictor
import numpy as np

class ConfidencePredictor(StandardPredictor):
    def predict(self, model, data, **kwargs):
        # Get base predictions
        predictions = super().predict(model, data, **kwargs)
        
        # Get probabilities
        probabilities = self.predict_proba(model, data, **kwargs)
        
        # Calculate confidence scores
        for col in self.output_columns:
            if col in probabilities:
                # Get the probability for the predicted class
                pred_probs = probabilities[col].values
                pred_classes = predictions[col].values
                
                # Get the probability of the predicted class for each sample
                confidence = np.array([
                    pred_probs[i, np.where(model.classes_ == pred_classes[i])[0][0]]
                    for i in range(len(pred_classes))
                ])
                
                # Add confidence column
                predictions[f"{col}_confidence"] = confidence
                
        return predictions
```

### Ensemble Prediction

```python
from nexusml.core.prediction import StandardPredictor
import numpy as np

class EnsemblePredictor(StandardPredictor):
    def __init__(self, threshold=0.5, output_columns=None, weights=None):
        super().__init__(threshold, output_columns)
        self.weights = weights
        
    def predict(self, models, data, **kwargs):
        # Validate models
        if not isinstance(models, list):
            raise ValueError("Models must be a list for ensemble prediction")
            
        # Initialize predictions
        all_predictions = []
        
        # Get predictions from each model
        for model in models:
            predictions = super().predict(model, data, **kwargs)
            all_predictions.append(predictions)
            
        # Combine predictions
        result = pd.DataFrame()
        
        # Use voting or weighted voting
        for col in self.output_columns:
            # Get predictions for this column from all models
            col_predictions = [p[col].values for p in all_predictions]
            
            # Use weighted voting if weights are provided
            if self.weights:
                # Get weighted votes
                weighted_votes = {}
                for i, preds in enumerate(col_predictions):
                    weight = self.weights[i]
                    for j, pred in enumerate(preds):
                        if pred not in weighted_votes:
                            weighted_votes[pred] = 0
                        weighted_votes[pred] += weight
                        
                # Get the prediction with the highest weighted vote
                result[col] = [
                    max(weighted_votes.items(), key=lambda x: x[1])[0]
                    for _ in range(len(data))
                ]
            else:
                # Use majority voting
                result[col] = [
                    max(set([p[j] for p in col_predictions]), key=lambda x: sum([1 for p in col_predictions if p[j] == x]))
                    for j in range(len(data))
                ]
                
        return result
```

### Prediction with Explanation

```python
from nexusml.core.prediction import StandardPredictor, PredictionExplainer

class ExplainablePredictor(StandardPredictor):
    def __init__(self, threshold=0.5, output_columns=None, feature_names=None):
        super().__init__(threshold, output_columns)
        self.feature_names = feature_names
        self.explainer = PredictionExplainer(feature_names)
        
    def predict(self, model, data, **kwargs):
        # Get base predictions
        predictions = super().predict(model, data, **kwargs)
        
        # Add explanations if requested
        if kwargs.get("explain", False):
            explanations = self.explainer.explain_predictions(
                model, data, prediction_indices=kwargs.get("explain_indices")
            )
            
            # Add explanations to the result
            predictions["explanations"] = explanations
            
        return predictions