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
        from examples.pipeline_orchestrator_example import create_orchestrator, train_model_example
        
        # Get the default container provider
        provider = ContainerProvider()
        
        # Register core components
        register_core_components(provider)
        
        # Register pipeline components
        register_pipeline_components(provider)
        
        # Create orchestrator
        orchestrator = create_orchestrator()
        
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
        from examples.prediction_pipeline_example import orchestrator_prediction_example
        
        # Get the default container provider
        provider = ContainerProvider()
        
        # Register core components
        register_core_components(provider)
        
        # Register pipeline components
        register_pipeline_components(provider)
        
        # Run the prediction example
        orchestrator_prediction_example(logger)
        
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
        if not end_to_end_success:
            logger.error("  - End-to-end verification failed")
        if not prediction_pipeline_success:
            logger.error("  - Prediction pipeline verification failed")
    
    logger.info("NexusML Refactoring Verification completed")


if __name__ == "__main__":
    main()