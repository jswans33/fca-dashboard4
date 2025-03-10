"""
Pipeline Factory Example

This example demonstrates how to use the Pipeline Factory to create and configure
pipeline components with proper dependencies.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from nexusml.src.utils.di.container import DIContainer
from nexusml.src.pipeline.factory import PipelineFactory
from nexusml.src.pipeline.interfaces import (
    DataLoader,
    DataPreprocessor,
    FeatureEngineer,
    ModelBuilder,
    ModelEvaluator,
    ModelSerializer,
    ModelTrainer,
    Predictor,
)
from nexusml.src.pipeline.registry import ComponentRegistry


# Mock implementations for demonstration purposes
class CSVDataLoader(DataLoader):
    """Example CSV data loader."""

    def __init__(self, file_path=None):
        self.file_path = file_path

    def load_data(self, data_path=None, **kwargs):
        """Load data from a CSV file."""
        path = data_path or self.file_path
        print(f"Loading data from {path}")
        # In a real implementation, this would use pd.read_csv
        return pd.DataFrame(
            {
                "description": ["Heat Exchanger", "Pump", "Fan", "Boiler"],
                "service_life": [20.0, 15.0, 10.0, 25.0],
                "target": ["23.01", "23.21", "23.34", "23.52"],
            }
        )

    def get_config(self):
        """Get the configuration for the data loader."""
        return {"file_path": self.file_path}


class StandardPreprocessor(DataPreprocessor):
    """Example standard preprocessor."""

    def preprocess(self, data, **kwargs):
        """Preprocess the input data."""
        print("Preprocessing data")
        # In a real implementation, this would clean and prepare the data
        return data

    def verify_required_columns(self, data):
        """Verify that all required columns exist."""
        required_columns = ["description", "service_life"]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Required column '{column}' not found")
        return data


class TextFeatureEngineer(FeatureEngineer):
    """Example text feature engineer."""

    def engineer_features(self, data, **kwargs):
        """Engineer features from the input data."""
        print("Engineering features")
        # In a real implementation, this would transform text into features
        return data

    def fit(self, data, **kwargs):
        """Fit the feature engineer to the input data."""
        print("Fitting feature engineer")
        return self

    def transform(self, data, **kwargs):
        """Transform the input data using the fitted feature engineer."""
        print("Transforming data")
        return data


class RandomForestModelBuilder(ModelBuilder):
    """Example random forest model builder."""

    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators

    def build_model(self, **kwargs):
        """Build a machine learning model."""
        print(f"Building model with {self.n_estimators} estimators")
        # In a real implementation, this would create a scikit-learn pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        return Pipeline(
            [("classifier", RandomForestClassifier(n_estimators=self.n_estimators))]
        )

    def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
        """Optimize hyperparameters for the model."""
        print("Optimizing hyperparameters")
        return model


class StandardModelTrainer(ModelTrainer):
    """Example standard model trainer."""

    def train(self, model, x_train, y_train, **kwargs):
        """Train a model on the provided data."""
        print("Training model")
        # In a real implementation, this would fit the model
        return model

    def cross_validate(self, model, x, y, **kwargs):
        """Perform cross-validation on the model."""
        print("Cross-validating model")
        return {"accuracy": [0.9]}


class StandardModelEvaluator(ModelEvaluator):
    """Example standard model evaluator."""

    def evaluate(self, model, x_test, y_test, **kwargs):
        """Evaluate a trained model on test data."""
        print("Evaluating model")
        return {"accuracy": 0.9}

    def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
        """Analyze model predictions in detail."""
        print("Analyzing predictions")
        return {"confusion_matrix": [[1, 0], [0, 1]]}


class PickleModelSerializer(ModelSerializer):
    """Example pickle model serializer."""

    def save_model(self, model, path, **kwargs):
        """Save a trained model to disk."""
        print(f"Saving model to {path}")
        # In a real implementation, this would use pickle or joblib

    def load_model(self, path, **kwargs):
        """Load a trained model from disk."""
        print(f"Loading model from {path}")
        # In a real implementation, this would use pickle or joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        return Pipeline([("classifier", RandomForestClassifier())])


class StandardPredictor(Predictor):
    """Example standard predictor."""

    def predict(self, model, data, **kwargs):
        """Make predictions using a trained model."""
        print("Making predictions")
        # In a real implementation, this would use model.predict
        return pd.DataFrame({"prediction": ["23.01", "23.21", "23.34", "23.52"]})

    def predict_proba(self, model, data, **kwargs):
        """Make probability predictions using a trained model."""
        print("Making probability predictions")
        # In a real implementation, this would use model.predict_proba
        return {
            "target": pd.DataFrame(
                {
                    "23.01": [0.9, 0.1, 0.1, 0.1],
                    "23.21": [0.1, 0.9, 0.1, 0.1],
                    "23.34": [0.1, 0.1, 0.9, 0.1],
                    "23.52": [0.1, 0.1, 0.1, 0.9],
                }
            )
        }


def main():
    """Main function to demonstrate the pipeline factory."""
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
    registry.set_default_implementation(FeatureEngineer, "text")
    registry.set_default_implementation(ModelBuilder, "random_forest")
    registry.set_default_implementation(ModelTrainer, "standard")
    registry.set_default_implementation(ModelEvaluator, "standard")
    registry.set_default_implementation(ModelSerializer, "pickle")
    registry.set_default_implementation(Predictor, "standard")

    # Create a factory
    factory = PipelineFactory(registry, container)

    # Create pipeline components
    data_loader = factory.create_data_loader(file_path="data.csv")
    preprocessor = factory.create_data_preprocessor()
    feature_engineer = factory.create_feature_engineer()
    model_builder = factory.create_model_builder(n_estimators=200)
    model_trainer = factory.create_model_trainer()
    model_evaluator = factory.create_model_evaluator()
    model_serializer = factory.create_model_serializer()
    predictor = factory.create_predictor()

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
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    print("\n=== Building and Training Model ===")
    model = model_builder.build_model()
    trained_model = model_trainer.train(model, X_train, y_train)

    print("\n=== Evaluating Model ===")
    evaluation = model_evaluator.evaluate(trained_model, X_test, y_test)
    print(f"Model evaluation: {evaluation}")

    print("\n=== Saving Model ===")
    model_serializer.save_model(trained_model, "model.pkl")

    print("\n=== Making Predictions ===")
    new_data = pd.DataFrame(
        {"description": ["New Heat Exchanger"], "service_life": [22.0]}
    )
    predictions = predictor.predict(trained_model, new_data)
    print(f"Predictions: {predictions}")

    print("\n=== Component Dependencies ===")

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
    print(f"Component data_loader: {type(component.data_loader).__name__}")
    print(f"Component preprocessor: {type(component.preprocessor).__name__}")


if __name__ == "__main__":
    main()
