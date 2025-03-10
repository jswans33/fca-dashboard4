#!/usr/bin/env python
"""
NexusML Refactoring Verification Script

This script verifies that all components of the refactored NexusML library
work correctly together. It tests component resolution, pipeline creation,
and end-to-end functionality.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.di.container import DIContainer
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.di.registration import register_core_components
from nexusml.core.di.pipeline_registration import register_pipeline_components
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.config.manager import ConfigurationManager


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "verification.log", encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        encoding='utf-8',
    )
    return logging.getLogger("verification_script")


def verify_component_resolution(logger):
    """Verify that all components can be resolved from the DI container."""
    logger.info("Verifying component resolution...")
    
    # Get the default container provider
    provider = ContainerProvider()
    
    # Register core components
    register_core_components(provider)
    
    # Register pipeline components
    register_pipeline_components(provider)
    
    # Get the container from the provider
    container = provider.container
    
    # Try to resolve key components
    components_to_verify = [
        ("ConfigurationManager", "Configuration manager"),
        ("ConfigurableDataLoadingStage", "Data loading stage"),
        ("ConfigDrivenValidationStage", "Validation stage"),
        ("SimpleFeatureEngineeringStage", "Feature engineering stage"),
        ("RandomSplittingStage", "Data splitting stage"),
        ("ConfigDrivenModelBuildingStage", "Model building stage"),
        ("StandardModelTrainingStage", "Model training stage"),
        ("ClassificationEvaluationStage", "Model evaluation stage"),
        ("ModelCardSavingStage", "Model saving stage"),
        ("StandardPredictionStage", "Prediction stage")
    ]
    
    success = True
    for component_class, component_desc in components_to_verify:
        try:
            # Import the component class dynamically
            module_parts = component_class.split('.')
            if len(module_parts) == 1:
                # Try to find the module path for the class
                if component_class == "ConfigurationManager":
                    module_path = "nexusml.config.manager"
                elif component_class == "ConfigurableDataLoadingStage":
                    module_path = "nexusml.core.pipeline.stages.data_loading"
                elif component_class == "ConfigDrivenValidationStage":
                    module_path = "nexusml.core.pipeline.stages.validation"
                elif component_class == "SimpleFeatureEngineeringStage":
                    module_path = "nexusml.core.pipeline.stages.feature_engineering"
                elif component_class == "RandomSplittingStage":
                    module_path = "nexusml.core.pipeline.stages.data_splitting"
                elif component_class == "ConfigDrivenModelBuildingStage":
                    module_path = "nexusml.core.pipeline.stages.model_building"
                elif component_class == "StandardModelTrainingStage":
                    module_path = "nexusml.core.pipeline.stages.model_training"
                elif component_class == "ClassificationEvaluationStage":
                    module_path = "nexusml.core.pipeline.stages.model_evaluation"
                elif component_class == "ModelCardSavingStage":
                    module_path = "nexusml.core.pipeline.stages.model_saving"
                elif component_class == "StandardPredictionStage":
                    module_path = "nexusml.core.pipeline.stages.prediction"
                else:
                    logger.warning(f"Unknown module path for {component_class}")
                    continue
                
                # Try to import the class
                try:
                    module = __import__(module_path, fromlist=[component_class])
                    component_type = getattr(module, component_class)
                    # Try to resolve the component
                    component = container.resolve(component_type)
                    logger.info(f"✅ Successfully resolved {component_desc} ({component_class})")
                except (ImportError, AttributeError) as e:
                    logger.error(f"❌ Failed to import {component_class}: {e}")
                    success = False
                    continue
            else:
                # If the component class includes the module path
                module_path = '.'.join(module_parts[:-1])
                class_name = module_parts[-1]
                try:
                    module = __import__(module_path, fromlist=[class_name])
                    component_type = getattr(module, class_name)
                    # Try to resolve the component
                    component = container.resolve(component_type)
                    logger.info(f"✅ Successfully resolved {component_desc} ({class_name})")
                except (ImportError, AttributeError) as e:
                    logger.error(f"❌ Failed to import {class_name} from {module_path}: {e}")
                    success = False
                    continue
        except Exception as e:
            logger.error(f"❌ Failed to resolve {component_desc}: {e}")
            success = False
    
    return success


def verify_pipeline_factory(logger):
    """Verify that the pipeline factory can create different pipeline types."""
    logger.info("Verifying pipeline factory...")
    
    # Create a component registry
    registry = ComponentRegistry()
    
    # Get the default container provider
    provider = ContainerProvider()
    
    # Register core components
    register_core_components(provider)
    
    # Register pipeline components
    register_pipeline_components(provider)
    
    # Get the container from the provider
    container = provider.container
    
    # Create a pipeline factory
    factory = PipelineFactory(registry, container)
    
    # Try to create different pipeline types
    pipeline_types = ["training", "prediction", "evaluation"]
    
    success = True
    for pipeline_type in pipeline_types:
        try:
            pipeline = factory.create_pipeline(pipeline_type)
            logger.info(f"✅ Successfully created {pipeline_type} pipeline")
        except Exception as e:
            logger.error(f"❌ Failed to create {pipeline_type} pipeline: {e}")
            success = False
    
    return success


def verify_end_to_end(logger):
    """Verify end-to-end functionality with a simple example."""
    logger.info("Verifying end-to-end functionality...")
    
    # Import the example script functions
    try:
        from examples.pipeline_orchestrator_example import create_orchestrator, train_model_example, StandardDataLoader
        
        # Create orchestrator - this will create and register all components
        orchestrator = create_orchestrator()
        
        # Access the container from the orchestrator
        if not hasattr(orchestrator, 'container'):
            logger.error("❌ End-to-end verification failed: orchestrator does not expose container")
            return False
            
        # Register the ModelBuilder with the factory's container
        from nexusml.core.model_building.base import ModelBuilder
        from nexusml.core.model_building.builders.random_forest import RandomForestBuilder
        
        # Get the factory from the orchestrator
        factory = orchestrator.factory
        
        # Register ModelBuilder with the factory's container
        logger.info("Registering ModelBuilder in the factory's container")
        factory.container.register_instance(ModelBuilder, RandomForestBuilder())
        
        # Register ModelTrainer with the factory's container
        from nexusml.core.model_training.base import ModelTrainer
        from nexusml.core.model_training.trainers.standard import StandardModelTrainer
        logger.info("Registering ModelTrainer in the factory's container")
        factory.container.register_instance(ModelTrainer, StandardModelTrainer())
        
        # Register ModelEvaluator with the factory's container
        from nexusml.core.model_building.base import ModelEvaluator
        from nexusml.core.pipeline.components.model_evaluator import EnhancedModelEvaluator
        logger.info("Registering ModelEvaluator in the factory's container")
        factory.container.register_instance(ModelEvaluator, EnhancedModelEvaluator())
        
        # Run a simple training example
        model = train_model_example(orchestrator, logger)
        
        if model:
            logger.info("✅ End-to-end verification successful")
            return True
        else:
            logger.error("❌ End-to-end verification failed: model training failed")
            return False
    except Exception as e:
        logger.error(f"❌ End-to-end verification failed: {e}")
        return False


def verify_prediction_pipeline(logger):
    """Verify the prediction pipeline functionality."""
    logger.info("Verifying prediction pipeline...")
    
    try:
        # Try to import the module using a direct path approach
        import sys
        import importlib.util
        
        # Get the absolute path to the prediction_pipeline_example.py file
        file_path = Path(__file__).resolve().parent.parent.parent / "examples" / "prediction_pipeline_example.py"
        
        if not file_path.exists():
            logger.error(f"❌ Prediction pipeline example file not found at: {file_path}")
            return False
            
        # Import the module using importlib
        spec = importlib.util.spec_from_file_location("prediction_pipeline_example", file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["prediction_pipeline_example"] = module
        spec.loader.exec_module(module)
        
        # Get the orchestrator_prediction_example function
        orchestrator_prediction_example = module.orchestrator_prediction_example
        
        # Create orchestrator - this will create and register all components
        from examples.pipeline_orchestrator_example import create_orchestrator
        orchestrator = create_orchestrator()
        
        # Run the prediction example with the orchestrator
        orchestrator_prediction_example(logger, orchestrator)
        
        # Check if the output file was created
        output_file = Path("examples/output/orchestrator_prediction_results.csv")
        if output_file.exists():
            logger.info("✅ Prediction pipeline verification successful")
            return True
        else:
            logger.error("❌ Prediction pipeline verification failed: output file not created")
            return False
    except Exception as e:
        logger.error(f"❌ Prediction pipeline verification failed: {e}")
        return False


def verify_feature_engineering(logger):
    """Verify the feature engineering components."""
    logger.info("Verifying feature engineering components...")
    
    try:
        import pandas as pd
        from nexusml.core.feature_engineering.config_driven import ConfigDrivenFeatureEngineer
        
        # Create a sample DataFrame
        data = pd.DataFrame({
            "description": ["Air Handling Unit", "Centrifugal Chiller", "Centrifugal Pump"],
            "service_life": [20, 25, 15],
        })
        
        # Create a feature engineering configuration
        config = {
            "text_columns": ["description"],
            "numeric_columns": ["service_life"],
            "transformations": [
                {
                    "type": "text_normalizer",
                    "columns": ["description"],
                    "lowercase": True,
                    "remove_punctuation": True,
                    "output_column_suffix": "_normalized"
                },
                {
                    "type": "numeric_scaler",
                    "columns": ["service_life"],
                    "method": "standard",
                    "output_column_suffix": "_scaled"
                }
            ]
        }
        
        # Create a feature engineer
        feature_engineer = ConfigDrivenFeatureEngineer(config=config)
        
        # Fit and transform the data
        transformed_data = feature_engineer.fit_transform(data)
        
        # Verify the transformation
        # Check if any columns with 'normalized' or 'scaled' in their names exist
        normalized_columns = [col for col in transformed_data.columns if 'normalized' in col]
        scaled_columns = [col for col in transformed_data.columns if 'scaled' in col]
        
        if normalized_columns and scaled_columns:
            logger.info(f"✅ Feature engineering verification successful. Found normalized columns: {normalized_columns} and scaled columns: {scaled_columns}")
            return True
        else:
            logger.error(f"❌ Feature engineering verification failed: expected columns not found. Available columns: {transformed_data.columns.tolist()}")
            return False
    except Exception as e:
        logger.error(f"❌ Feature engineering verification failed: {e}")
        return False


def verify_model_building(logger):
    """Verify the model building components."""
    logger.info("Verifying model building components...")
    
    try:
        import pandas as pd
        import numpy as np
        from nexusml.core.model_building.builders.random_forest import RandomForestBuilder
        
        # Create a sample DataFrame
        X = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "combined_text": ["text " + str(i) for i in range(100)],
            "service_life": np.random.randint(10, 30, 100),
        })
        # Create a single-output target as a DataFrame to match the model's expectations
        y = pd.DataFrame({
            "target": np.random.randint(0, 2, 100),
        })
        
        # Create a model builder with parameters
        model_builder = RandomForestBuilder(
            n_estimators=10,
            max_depth=5,
            random_state=42
        )
        
        # Build a model
        model = model_builder.build_model()
        
        # Verify the model
        if hasattr(model, "fit") and hasattr(model, "predict"):
            # Try to fit the model
            model.fit(X, y)
            
            # Try to make predictions
            predictions = model.predict(X)
            
            if len(predictions) == len(X):
                logger.info("✅ Model building verification successful")
                return True
            else:
                logger.error("❌ Model building verification failed: prediction length mismatch")
                return False
        else:
            logger.error("❌ Model building verification failed: model missing required methods")
            return False
    except Exception as e:
        logger.error(f"❌ Model building verification failed: {e}")
        return False


def verify_model_card_system(logger):
    """Verify the model card system."""
    logger.info("Verifying model card system...")
    
    try:
        import tempfile
        from nexusml.core.model_card.model_card import ModelCard
        from nexusml.core.model_card.generator import ModelCardGenerator
        from nexusml.core.model_card.viewer import print_model_card_summary
        
        # Create a temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Create a model card
            model_card = ModelCard(
                model_id="test_model",
                model_type="random_forest",
                description="A test model for verification",
                author="NexusML Team"
            )
            
            # Add some data to the model card
            model_card.add_training_data_info(
                source="test_data.csv",
                size=1000,
                features=["feature1", "feature2", "combined_text", "service_life"],
                target="target"
            )
            
            model_card.add_metrics({
                "accuracy": 0.95,
                "precision": 0.94,
                "recall": 0.93,
                "f1": 0.935
            })
            
            model_card.add_parameters({
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            })
            
            model_card.add_limitation("This model is for testing purposes only.")
            model_card.set_intended_use("This model is intended for testing the model card system.")
            
            # Save the model card
            model_card_path = temp_dir_path / "test_model.card.json"
            model_card.save(model_card_path)
            
            # Check if the file was created
            if not model_card_path.exists():
                logger.error("❌ Model card system verification failed: model card file not created")
                return False
            
            # Load the model card back
            loaded_model_card = ModelCard.load(model_card_path)
            
            # Verify the loaded model card
            if (loaded_model_card.data["model_id"] != "test_model" or
                loaded_model_card.data["model_type"] != "random_forest" or
                loaded_model_card.data["metrics"]["accuracy"] != 0.95):
                logger.error("❌ Model card system verification failed: loaded model card data mismatch")
                return False
            
            # Test the model card generator
            generator = ModelCardGenerator()
            
            # Create a simple model for testing
            from sklearn.ensemble import RandomForestClassifier
            import pandas as pd
            import numpy as np
            
            X = pd.DataFrame({
                "feature1": np.random.rand(10),
                "feature2": np.random.rand(10)
            })
            y = pd.Series(np.random.randint(0, 2, 10), name="target")
            
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            # Generate a model card from the model
            generated_card = generator.generate_from_training(
                model=model,
                model_id="generated_model",
                X_train=X,
                y_train=y,
                metrics={"accuracy": 0.9},
                description="A generated model card"
            )
            
            # Save the generated model card
            generated_card_path = temp_dir_path / "generated_model.card.json"
            generated_card.save(generated_card_path)
            
            # Check if the file was created
            if not generated_card_path.exists():
                logger.error("❌ Model card system verification failed: generated model card file not created")
                return False
            
            # Create a markdown version
            markdown_path = temp_dir_path / "generated_model.md"
            with open(markdown_path, "w") as f:
                f.write(generated_card.to_markdown())
            
            # Check if the markdown file was created
            if not markdown_path.exists():
                logger.error("❌ Model card system verification failed: markdown file not created")
                return False
            
            logger.info("✅ Model card system verification successful")
            return True
    except Exception as e:
        logger.error(f"❌ Model card system verification failed: {e}")
        return False


def main():
    """Main function to run the verification script."""
    logger = setup_logging()
    logger.info("Starting NexusML Refactoring Verification")
    
    # Verify component resolution
    component_resolution_success = verify_component_resolution(logger)
    
    # Verify pipeline factory
    pipeline_factory_success = verify_pipeline_factory(logger)
    
    # Verify feature engineering
    feature_engineering_success = verify_feature_engineering(logger)
    
    # Verify model building
    model_building_success = verify_model_building(logger)
    
    # Verify model card system
    model_card_success = verify_model_card_system(logger)
    
    # Verify end-to-end functionality
    end_to_end_success = verify_end_to_end(logger)
    
    # Verify prediction pipeline
    prediction_pipeline_success = verify_prediction_pipeline(logger)
    
    # Overall verification result
    all_successful = (
        component_resolution_success and
        pipeline_factory_success and
        feature_engineering_success and
        model_building_success and
        model_card_success and
        end_to_end_success and
        prediction_pipeline_success
    )
    
    if all_successful:
        logger.info("✅ All verification tests passed!")
    else:
        logger.error("❌ Some verification tests failed")
        if not component_resolution_success:
            logger.error("  - Component resolution verification failed")
        if not pipeline_factory_success:
            logger.error("  - Pipeline factory verification failed")
        if not feature_engineering_success:
            logger.error("  - Feature engineering verification failed")
        if not model_building_success:
            logger.error("  - Model building verification failed")
        if not model_card_success:
            logger.error("  - Model card system verification failed")
        if not end_to_end_success:
            logger.error("  - End-to-end verification failed")
        if not prediction_pipeline_success:
            logger.error("  - Prediction pipeline verification failed")
    
    logger.info("NexusML Refactoring Verification completed")


if __name__ == "__main__":
    main()