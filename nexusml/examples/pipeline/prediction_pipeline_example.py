#!/usr/bin/env python
"""
Prediction Pipeline Example

This example demonstrates how to use the prediction pipeline to make predictions
on new data using a trained model.
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

from nexusml.src.utils.di.container import DIContainer
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.factory import PipelineFactory
from nexusml.src.pipeline.orchestrator import PipelineOrchestrator
from nexusml.src.pipeline.registry import ComponentRegistry


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
            logging.FileHandler(log_dir / "prediction_pipeline_example.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return logging.getLogger("prediction_pipeline_example")


def create_orchestrator():
    """Create a PipelineOrchestrator instance."""
    # Import the orchestrator creation function from the pipeline_orchestrator_example
    from examples.pipeline_orchestrator_example import create_orchestrator as create_base_orchestrator
    
    # Create the orchestrator
    return create_base_orchestrator()


def load_model(orchestrator, model_path="nexusml/output/models/equipment_classifier.pkl"):
    """Load a trained model."""
    logger = logging.getLogger("prediction_pipeline_example")
    logger.info(f"Loading model from {model_path}")
    
    try:
        model = orchestrator.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def load_prediction_data(data_path="examples/data/sample_prediction_data.csv"):
    """Load data for prediction."""
    logger = logging.getLogger("prediction_pipeline_example")
    logger.info(f"Loading prediction data from {data_path}")
    
    try:
        # Check if the file exists
        if not Path(data_path).exists():
            logger.error(f"File not found: {data_path}")
            # Create a sample data file if it doesn't exist
            create_sample_prediction_data(data_path)
            logger.info(f"Created sample prediction data at {data_path}")
        
        # Load the data
        if data_path.lower().endswith(".csv"):
            data = pd.read_csv(data_path)
        elif data_path.lower().endswith((".xls", ".xlsx")):
            data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(data)} records for prediction")
        return data
    except Exception as e:
        logger.error(f"Error loading prediction data: {e}")
        return None


def create_sample_prediction_data(output_path="examples/data/sample_prediction_data.csv"):
    """Create sample prediction data."""
    logger = logging.getLogger("prediction_pipeline_example")
    logger.info(f"Creating sample prediction data at {output_path}")
    
    # Create a sample DataFrame
    data = pd.DataFrame(
        {
            "equipment_tag": ["AHU-01", "CHW-01", "P-01", "VAV-01", "FCU-01"],
            "manufacturer": ["Trane", "Carrier", "Armstrong", "Johnson Controls", "Daikin"],
            "model": ["M-1000", "C-2000", "A-3000", "J-4000", "D-5000"],
            "description": [
                "Air Handling Unit with cooling coil",
                "Centrifugal Chiller for HVAC system",
                "Centrifugal Pump for chilled water",
                "Variable Air Volume terminal unit",
                "Fan Coil Unit for zone temperature control",
            ],
            "service_life": [20, 25, 15, 15, 10],
        }
    )
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the data
    data.to_csv(output_path, index=False)
    
    logger.info(f"Created sample prediction data with {len(data)} records")
    return data


def orchestrator_prediction_example(logger=None, orchestrator=None):
    """Example of using the orchestrator for prediction."""
    if logger is None:
        logger = setup_logging()
    
    logger.info("Starting prediction pipeline example")
    
    # Create orchestrator if not provided
    if orchestrator is None:
        orchestrator = create_orchestrator()
    
    # Load model
    model = load_model(orchestrator)
    if model is None:
        logger.error("Failed to load model, cannot proceed with prediction")
        return
    
    # Load prediction data
    data = load_prediction_data()
    if data is None:
        logger.error("Failed to load prediction data, cannot proceed with prediction")
        return
    
    # Make predictions
    try:
        output_path = "examples/output/orchestrator_prediction_results.csv"
        predictions = orchestrator.predict(
            model=model,
            data=data,
            output_path=output_path,
        )
        
        logger.info("Predictions completed successfully")
        logger.info(f"Predictions saved to: {output_path}")
        logger.info("Sample predictions:")
        for i, row in predictions.head(3).iterrows():
            logger.info(f"  Item {i+1}:")
            logger.info(f"    Equipment Tag: {data.iloc[i]['equipment_tag']}")
            logger.info(f"    Description: {data.iloc[i]['description']}")
            logger.info(f"    Predicted Category: {row.get('category_name', 'N/A')}")
            logger.info(f"    Predicted System Type: {row.get('mcaa_system_category', 'N/A')}")
        
        # Get execution summary
        summary = orchestrator.get_execution_summary()
        logger.info("Execution summary:")
        logger.info(f"  Status: {summary['status']}")
        logger.info("  Component execution times:")
        for component, time in summary["component_execution_times"].items():
            logger.info(f"    {component}: {time:.2f} seconds")
        logger.info(f"  Total execution time: {summary.get('total_execution_time', 0):.2f} seconds")
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return None


def main():
    """Main function to run the example."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Prediction Pipeline Example")
    
    # Run the prediction example
    orchestrator_prediction_example(logger)
    
    logger.info("Prediction Pipeline Example completed")


if __name__ == "__main__":
    main()
