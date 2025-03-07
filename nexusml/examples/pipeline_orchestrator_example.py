"""
Example script demonstrating the usage of the PipelineOrchestrator and PipelineContext classes.

This example shows how to use the PipelineOrchestrator to train a model, make predictions,
and evaluate the model's performance.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry


# Define mock components for demonstration
class SimpleDataLoader:
    """Simple implementation of DataLoader for demonstration."""

    def load_data(self, data_path=None, **kwargs):
        """Load data from a CSV file or return a sample DataFrame."""
        if data_path and os.path.exists(data_path):
            return pd.read_csv(data_path)

        # Return a sample DataFrame if no path is provided or file doesn't exist
        return pd.DataFrame(
            {
                "equipment_tag": ["tag1", "tag2", "tag3"],
                "manufacturer": ["mfg1", "mfg2", "mfg3"],
                "model": ["model1", "model2", "model3"],
                "category_name": ["cat1", "cat2", "cat3"],
                "uniformat_code": ["code1", "code2", "code3"],
                "mcaa_system_category": ["sys1", "sys2", "sys3"],
                "Equipment_Type": ["type1", "type2", "type3"],
                "System_Subtype": ["subtype1", "subtype2", "subtype3"],
            }
        )


class SimpleDataPreprocessor:
    """Simple implementation of DataPreprocessor for demonstration."""

    def preprocess(self, data, **kwargs):
        """Preprocess the data."""
        # In a real implementation, this would clean the data, handle missing values, etc.
        return data


class SimpleFeatureEngineer:
    """Simple implementation of FeatureEngineer for demonstration."""

    def fit(self, data, **kwargs):
        """Fit the feature engineer to the data."""
        return self

    def transform(self, data, **kwargs):
        """Transform the data by adding engineered features."""
        # Add engineered features
        data["combined_text"] = (
            data["equipment_tag"] + " " + data["manufacturer"] + " " + data["model"]
        )
        data["service_life"] = 20.0
        return data


class SimpleModelBuilder:
    """Simple implementation of ModelBuilder for demonstration."""

    def build_model(self, **kwargs):
        """Build a model."""
        return Pipeline([("classifier", RandomForestClassifier(n_estimators=10))])


class SimpleModelTrainer:
    """Simple implementation of ModelTrainer for demonstration."""

    def train(self, model, x_train, y_train, **kwargs):
        """Train the model."""
        # Extract the first target column for simplicity
        target = y_train.iloc[:, 0]

        # Train the model using only numerical features
        # RandomForestClassifier can only handle numerical data
        model.fit(x_train[["service_life"]], target)
        return model

    def cross_validate(self, model, x, y, **kwargs):
        """Perform cross-validation on the model."""
        # In a real implementation, this would perform cross-validation
        # For this example, we'll just return some mock results
        return {
            "accuracy": [0.8, 0.9, 0.85],
            "f1_macro": [0.75, 0.85, 0.8],
            "precision": [0.82, 0.88, 0.85],
            "recall": [0.79, 0.86, 0.83],
        }


class SimpleModelEvaluator:
    """Simple implementation of ModelEvaluator for demonstration."""

    def evaluate(self, model, x_test, y_test, **kwargs):
        """Evaluate the model."""
        # Extract the first target column for simplicity
        target = y_test.iloc[:, 0]

        # Ensure we only use the service_life column for prediction
        # This is important because the model was trained only on service_life
        features = pd.DataFrame({"service_life": x_test["service_life"]})

        # Make predictions
        predictions = model.predict(features)

        # Calculate accuracy
        accuracy = sum(predictions == target) / len(target)

        return {
            "accuracy": accuracy,
            "f1_macro": 0.8,  # Placeholder value
            "precision": 0.82,  # Placeholder value
            "recall": 0.79,  # Placeholder value
        }

    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        """Analyze the predictions."""
        return {
            "confusion_matrix": [[10, 2], [3, 15]],  # Placeholder values
            "classification_report": "Sample classification report",
        }


class SimpleModelSerializer:
    """Simple implementation of ModelSerializer for demonstration."""

    def save_model(self, model, path, **kwargs):
        """Save the model to disk."""
        import pickle

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model
        with open(path, "wb") as f:
            pickle.dump(model, f)

        return path

    def load_model(self, path, **kwargs):
        """Load the model from disk."""
        import pickle

        with open(path, "rb") as f:
            model = pickle.load(f)

        return model


class SimplePredictor:
    """Simple implementation of Predictor for demonstration."""

    def predict(self, model, data, **kwargs):
        """Make predictions using the model."""
        # Extract only the numerical features for prediction
        # If service_life is in the data, use it; otherwise, use a default value
        if "service_life" in data.columns:
            features = pd.DataFrame({"service_life": data["service_life"]})
        else:
            features = pd.DataFrame({"service_life": [20.0] * len(data)})

        # Make predictions
        predictions = model.predict(features)

        # Create a DataFrame with predictions
        # Ensure the number of rows in the output matches the input
        num_rows = len(data)
        return pd.DataFrame(
            {
                "category_name": predictions[:num_rows],
                "uniformat_code": ["code1"] * num_rows,
                "mcaa_system_category": ["sys1"] * num_rows,
                "Equipment_Type": ["type1"] * num_rows,
                "System_Subtype": ["subtype1"] * num_rows,
            }
        )


def main():
    """Main function to demonstrate the PipelineOrchestrator."""
    # Create a registry and register components
    registry = ComponentRegistry()
    registry.register(DataLoader, "simple", SimpleDataLoader)
    registry.register(DataPreprocessor, "simple", SimpleDataPreprocessor)
    registry.register(FeatureEngineer, "simple", SimpleFeatureEngineer)
    registry.register(ModelBuilder, "simple", SimpleModelBuilder)
    registry.register(ModelTrainer, "simple", SimpleModelTrainer)
    registry.register(ModelEvaluator, "simple", SimpleModelEvaluator)
    registry.register(ModelSerializer, "simple", SimpleModelSerializer)
    registry.register(Predictor, "simple", SimplePredictor)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "simple")
    registry.set_default_implementation(DataPreprocessor, "simple")
    registry.set_default_implementation(FeatureEngineer, "simple")
    registry.set_default_implementation(ModelBuilder, "simple")
    registry.set_default_implementation(ModelTrainer, "simple")
    registry.set_default_implementation(ModelEvaluator, "simple")
    registry.set_default_implementation(ModelSerializer, "simple")
    registry.set_default_implementation(Predictor, "simple")

    # Create a container
    container = DIContainer()

    # Create a factory
    factory = PipelineFactory(registry, container)

    # Create a context
    context = PipelineContext()

    # Create an orchestrator
    orchestrator = PipelineOrchestrator(factory, context)

    # Create output directory
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train a model
    print("Training model...")
    model, metrics = orchestrator.train_model(
        data_path=None,  # Use sample data
        output_dir=str(output_dir),
        model_name="example_model",
    )
    print(f"Model trained with metrics: {metrics}")

    # Make predictions
    print("\nMaking predictions...")
    data = pd.DataFrame(
        {
            "equipment_tag": ["new_tag1", "new_tag2"],
            "manufacturer": ["new_mfg1", "new_mfg2"],
            "model": ["new_model1", "new_model2"],
        }
    )
    predictions = orchestrator.predict(
        model=model,
        data=data,
        output_path=str(output_dir / "predictions.csv"),
    )
    print(f"Predictions: {predictions}")

    # Evaluate the model
    print("\nEvaluating model...")
    evaluation = orchestrator.evaluate(
        model=model,
        data=pd.DataFrame(
            {
                "equipment_tag": ["eval_tag1", "eval_tag2", "eval_tag3"],
                "manufacturer": ["eval_mfg1", "eval_mfg2", "eval_mfg3"],
                "model": ["eval_model1", "eval_model2", "eval_model3"],
                "category_name": ["cat1", "cat2", "cat3"],
                "uniformat_code": ["code1", "code2", "code3"],
                "mcaa_system_category": ["sys1", "sys2", "sys3"],
                "Equipment_Type": ["type1", "type2", "type3"],
                "System_Subtype": ["subtype1", "subtype2", "subtype3"],
            }
        ),
        output_path=str(output_dir / "evaluation.json"),
    )
    print(f"Evaluation results: {evaluation}")

    # Get execution summary
    print("\nExecution summary:")
    summary = orchestrator.get_execution_summary()
    print(f"Status: {summary['status']}")
    print(f"Component execution times: {summary['component_execution_times']}")
    print(f"Total execution time: {summary['total_execution_time']}")


if __name__ == "__main__":
    main()
