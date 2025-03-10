#!/usr/bin/env python
"""
Pipeline Orchestrator Example

This example demonstrates how to use the PipelineOrchestrator to train a model,
make predictions, and evaluate the model's performance.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.pipeline.registry import ComponentRegistry
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


# Data loader implementation that handles both CSV and Excel files
class StandardDataLoader(DataLoader):
    """Data loader implementation that handles CSV and Excel files."""

    def __init__(self, file_path=None):
        self.file_path = file_path

    def load_data(self, data_path=None, **kwargs):
        """Load data from a file (CSV or Excel)."""
        path = data_path or self.file_path
        logger = logging.getLogger("pipeline_orchestrator_example")
        logger.info(f"Loading data from {path}")

        # In a real implementation, this would handle file not found errors properly
        try:
            # Handle the case where path might be None
            if path is None:
                raise ValueError("No data path provided")

            # Determine file type based on extension
            if path.lower().endswith(".csv"):
                return pd.read_csv(path)
            elif path.lower().endswith((".xls", ".xlsx")):
                return pd.read_excel(path)
            else:
                raise ValueError(f"Unsupported file format: {path}")

        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading data: {str(e)}")

    def get_config(self):
        """Get the configuration for the data loader."""
        return {"file_path": self.file_path}


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "pipeline_orchestrator_example.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("pipeline_orchestrator_example")


def create_orchestrator():
    """Create a PipelineOrchestrator instance."""
    # Create a component registry
    registry = ComponentRegistry()

    # Register default implementations
    # In a real application, you would register your actual implementations
    # For this example, we'll use the StandardDataLoader defined at module level
    
    # Create an instance of the module-level StandardDataLoader
    data_loader = StandardDataLoader()

    # Simple DataPreprocessor implementation
    class StandardPreprocessor(DataPreprocessor):
        """Standard data preprocessor implementation."""

        def preprocess(self, data, **kwargs):
            """Preprocess the input data."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Preprocessing data")

            # In a real implementation, this would clean and prepare the data
            # For this example, we'll just return the data as is
            return data

        def verify_required_columns(self, data):
            """Verify that all required columns exist in the DataFrame."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Verifying required columns")

            # Define required columns
            required_columns = ["description", "service_life"]

            # Check if required columns exist
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                # For this example, we'll add missing columns with default values
                logger.warning(f"Missing required columns: {missing_columns}")
                for col in missing_columns:
                    if col == "description":
                        data[col] = "Unknown"
                    elif col == "service_life":
                        data[col] = 15.0  # Default service life

            return data

    # Simple FeatureEngineer implementation
    class SimpleFeatureEngineer(FeatureEngineer):
        """Simple feature engineer implementation."""

        def engineer_features(self, data, **kwargs):
            """Engineer features from the input data."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Engineering features")

            # In a real implementation, this would transform raw data into features
            # For this example, we'll add required columns with default values

            # Add combined_text column
            if "description" in data.columns:
                data["combined_text"] = data["description"]
            else:
                data["combined_text"] = "Unknown description"

            # Add service_life column if it doesn't exist
            if "service_life" not in data.columns:
                data["service_life"] = 15.0  # Default service life

            # Add required target columns for the orchestrator
            required_target_columns = [
                "category_name",
                "uniformat_code",
                "mcaa_system_category",
                "Equipment_Type",
                "System_Subtype",
            ]

            for col in required_target_columns:
                if col not in data.columns:
                    data[col] = "Unknown"  # Default value for target columns

            return data

        def fit(self, data, **kwargs):
            """Fit the feature engineer to the input data."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Fitting feature engineer")

            # In a real implementation, this would fit transformers
            # For this example, we'll just return self
            return self

        def transform(self, data, **kwargs):
            """Transform the input data using the fitted feature engineer."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Transforming data with feature engineer")

            # In a real implementation, this would apply transformations
            # For this example, we'll just call engineer_features
            return self.engineer_features(data, **kwargs)

    # Simple ModelBuilder implementation
    class SimpleModelBuilder(ModelBuilder):
        """Simple model builder implementation."""

        def __init__(self, n_estimators=100):
            self.n_estimators = n_estimators

        def build_model(self, **kwargs):
            """Build a machine learning model."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info(f"Building model with {self.n_estimators} estimators")

            # In a real implementation, this would create a scikit-learn pipeline
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline

            return Pipeline(
                [("classifier", RandomForestClassifier(n_estimators=self.n_estimators))]
            )

        def optimize_hyperparameters(self, model, x_train, y_train, **kwargs):
            """Optimize hyperparameters for the model."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Optimizing hyperparameters")

            # In a real implementation, this would perform hyperparameter optimization
            # For this example, we'll just return the model as is
            return model

    # Simple ModelTrainer implementation
    class SimpleModelTrainer(ModelTrainer):
        """Simple model trainer implementation."""

        def train(self, model, x_train, y_train, **kwargs):
            """Train a model on the provided data."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Training model")

            # Actually fit the model to avoid NotFittedError
            if hasattr(model, "fit"):
                # Use only numerical features (service_life) for training
                # to avoid ValueError with text data
                if "service_life" in x_train.columns:
                    numerical_features = x_train[["service_life"]]
                    model.fit(numerical_features, y_train)
                else:
                    # If no numerical features, create a dummy feature
                    import numpy as np

                    dummy_features = np.ones((len(x_train), 1))
                    model.fit(dummy_features, y_train)

            return model

        def cross_validate(self, model, x, y, **kwargs):
            """Perform cross-validation on the model."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Cross-validating model")

            # In a real implementation, this would perform cross-validation
            # For this example, we'll just return dummy results
            return {"accuracy": [0.9], "f1": [0.85]}

    # Simple ModelEvaluator implementation
    class SimpleModelEvaluator(ModelEvaluator):
        """Simple model evaluator implementation."""

        def evaluate(self, model, x_test, y_test, **kwargs):
            """Evaluate a trained model on test data."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Evaluating model")

            # In a real implementation, this would evaluate the model
            # For this example, we'll just return dummy metrics
            return {"accuracy": 0.92, "f1": 0.88}

        def analyze_predictions(self, model, x_test, y_test, y_pred, **kwargs):
            """Analyze model predictions in detail."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Analyzing predictions")

            # In a real implementation, this would analyze predictions
            # For this example, we'll just return dummy analysis
            return {"confusion_matrix": [[10, 1], [2, 8]]}

    # Simple ModelSerializer implementation
    class SimpleModelSerializer(ModelSerializer):
        """Simple model serializer implementation."""

        def save_model(self, model, path, **kwargs):
            """Save a trained model to disk."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info(f"Saving model to {path}")

            # In a real implementation, this would save the model
            # For this example, we'll just log the action
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        def load_model(self, path, **kwargs):
            """Load a trained model from disk."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info(f"Loading model from {path}")

            # In a real implementation, this would load the model
            # For this example, we'll just return a dummy model
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.pipeline import Pipeline

            return Pipeline([("classifier", RandomForestClassifier())])

    # Simple Predictor implementation
    class SimplePredictor(Predictor):
        """Simple predictor implementation."""

        def predict(self, model, data, **kwargs):
            """Make predictions using a trained model."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Making predictions")

            # In a real implementation, this would use model.predict
            # For this example, we'll just return dummy predictions
            predictions = pd.DataFrame(
                {
                    "category_name": ["HVAC"] * len(data),
                    "uniformat_code": ["D3010"] * len(data),
                    "mcaa_system_category": ["Mechanical"] * len(data),
                    "Equipment_Type": ["Air Handling"] * len(data),
                    "System_Subtype": ["Cooling"] * len(data),
                }
            )

            return predictions

        def predict_proba(self, model, data, **kwargs):
            """Make probability predictions using a trained model."""
            logger = logging.getLogger("pipeline_orchestrator_example")
            logger.info("Making probability predictions")

            # In a real implementation, this would use model.predict_proba
            # For this example, we'll just return dummy probabilities
            return {
                "category_name": pd.DataFrame(
                    {"HVAC": [0.9] * len(data), "Plumbing": [0.1] * len(data)}
                )
            }

    # Register the components
    registry.register(DataLoader, "standard", lambda: data_loader)
    registry.register(DataPreprocessor, "standard", StandardPreprocessor)
    registry.register(FeatureEngineer, "simple", SimpleFeatureEngineer)
    registry.register(ModelBuilder, "simple", SimpleModelBuilder)
    registry.register(ModelTrainer, "simple", SimpleModelTrainer)
    registry.register(ModelEvaluator, "simple", SimpleModelEvaluator)
    registry.register(ModelSerializer, "simple", SimpleModelSerializer)
    registry.register(Predictor, "simple", SimplePredictor)

    # Set default implementations
    registry.set_default_implementation(DataLoader, "standard")
    registry.set_default_implementation(DataPreprocessor, "standard")
    registry.set_default_implementation(FeatureEngineer, "simple")
    registry.set_default_implementation(ModelBuilder, "simple")
    registry.set_default_implementation(ModelTrainer, "simple")
    registry.set_default_implementation(ModelEvaluator, "simple")
    registry.set_default_implementation(ModelSerializer, "simple")
    registry.set_default_implementation(Predictor, "simple")

    # Create a dependency injection container
    container = DIContainer()
    
    # Create instances of all components
    data_preprocessor = StandardPreprocessor()
    feature_engineer = SimpleFeatureEngineer()
    model_builder = SimpleModelBuilder()
    model_trainer = SimpleModelTrainer()
    model_evaluator = SimpleModelEvaluator()
    model_serializer = SimpleModelSerializer()
    predictor = SimplePredictor()
    
    # Register components with the container
    container.register_instance(DataLoader, data_loader)
    container.register_instance(DataPreprocessor, data_preprocessor)
    container.register_instance(FeatureEngineer, feature_engineer)
    container.register_instance(ModelBuilder, model_builder)
    container.register_instance(ModelTrainer, model_trainer)
    container.register_instance(ModelEvaluator, model_evaluator)
    container.register_instance(ModelSerializer, model_serializer)
    container.register_instance(Predictor, predictor)

    # Create a pipeline factory
    factory = PipelineFactory(registry, container)

    # Create a pipeline context
    context = PipelineContext()

    # Create a pipeline orchestrator
    orchestrator = PipelineOrchestrator(factory, context)

    # Expose the container for the verification script
    orchestrator.container = container

    return orchestrator


def train_model_example(orchestrator, logger):
    """Example of training a model using the orchestrator."""
    logger.info("Training model example")

    # Define paths
    data_path = (
        "examples/sample_data.xlsx"  # Use the sample data file in examples directory
    )
    feature_config_path = "nexusml/config/feature_config.yml"
    output_dir = "nexusml/output/models"

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
        logger.info(f"  Status: {summary['status']}")
        logger.info("  Component execution times:")
        for component, time in summary["component_execution_times"].items():
            logger.info(f"    {component}: {time:.2f} seconds")
        logger.info(
            f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds"
        )

        return model

    except Exception as e:
        logger.error(f"Error training model: {e}")
        return None


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
            output_path="nexusml/output/predictions.csv",
        )

        logger.info("Predictions completed successfully")
        logger.info(f"Predictions saved to: {orchestrator.context.get('output_path')}")
        logger.info("Sample predictions:")
        for i, row in predictions.head(3).iterrows():
            logger.info(f"  Item {i+1}:")
            logger.info(f"    Equipment Tag: {data.iloc[i]['equipment_tag']}")
            logger.info(f"    Description: {data.iloc[i]['description']}")
            logger.info(f"    Predicted Category: {row.get('category_name', 'N/A')}")
            logger.info(
                f"    Predicted System Type: {row.get('mcaa_system_category', 'N/A')}"
            )

        # Get execution summary
        summary = orchestrator.get_execution_summary()
        logger.info("Execution summary:")
        logger.info(f"  Status: {summary['status']}")
        logger.info("  Component execution times:")
        for component, time in summary["component_execution_times"].items():
            logger.info(f"    {component}: {time:.2f} seconds")
        logger.info(
            f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds"
        )

        return predictions

    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None


def evaluate_example(orchestrator, model, logger):
    """Example of evaluating a model using the orchestrator."""
    logger.info("Evaluation example")

    # Define paths
    data_path = (
        "examples/sample_data.xlsx"  # Use the sample data file in examples directory
    )
    output_path = "nexusml/output/evaluation_results.json"

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

        # Get execution summary
        summary = orchestrator.get_execution_summary()
        logger.info("Execution summary:")
        logger.info(f"  Status: {summary['status']}")
        logger.info("  Component execution times:")
        for component, time in summary["component_execution_times"].items():
            logger.info(f"    {component}: {time:.2f} seconds")
        logger.info(
            f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds"
        )

        return results

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None


def save_load_model_example(orchestrator, model, logger):
    """Example of saving and loading a model using the orchestrator."""
    logger.info("Save/Load model example")

    # Define paths
    model_path = "nexusml/output/models/saved_model.pkl"

    # Save the model
    try:
        saved_path = orchestrator.save_model(model, model_path)
        logger.info(f"Model saved to: {saved_path}")

        # Load the model
        loaded_model = orchestrator.load_model(saved_path)
        logger.info("Model loaded successfully")

        return loaded_model

    except Exception as e:
        logger.error(f"Error in save/load model example: {e}")
        return None


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


def main():
    """Main function to run the example."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Pipeline Orchestrator Example")

    # Create orchestrator
    orchestrator = create_orchestrator()

    # Run examples
    model = train_model_example(orchestrator, logger)

    if model:
        predict_example(orchestrator, model, logger)
        evaluate_example(orchestrator, model, logger)
        save_load_model_example(orchestrator, model, logger)

    # Error handling example
    error_handling_example(orchestrator, logger)

    logger.info("Pipeline Orchestrator Example completed")


if __name__ == "__main__":
    main()
