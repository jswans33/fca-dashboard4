#!/usr/bin/env python
"""
Model Card Example

This example demonstrates how to create, save, and visualize model cards
using the NexusML model card system.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the project root to the Python path if needed
import sys
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.src.models.cards.model_card import ModelCard
from nexusml.src.models.cards.generator import ModelCardGenerator
from nexusml.src.models.cards.viewer import print_model_card_summary, export_model_card_html

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_card_example")

def create_sample_data():
    """Create sample data for the example."""
    # Create a sample DataFrame
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
        "feature3": np.random.rand(100),
    })
    y = pd.Series(np.random.randint(0, 2, 100), name="target")
    
    # Split into train and test sets
    train_idx = np.random.choice(len(X), size=int(len(X) * 0.8), replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])
    
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """Train a simple model for the example."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    
    return metrics

def example_1_manual_model_card():
    """Example 1: Creating a model card manually."""
    logger.info("Example 1: Creating a model card manually")
    
    # Create output directory
    output_dir = Path("examples/output/model_cards")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a model card
    model_card = ModelCard(
        model_id="example_model_1",
        model_type="random_forest",
        description="A simple example model for binary classification",
        author="NexusML Team"
    )
    
    # Add training data information
    model_card.add_training_data_info(
        source="synthetic_data.csv",
        size=80,
        features=["feature1", "feature2", "feature3"],
        target="target"
    )
    
    # Add model parameters
    model_card.add_parameters({
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42
    })
    
    # Add metrics
    model_card.add_metrics({
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1": 0.935
    })
    
    # Add limitations
    model_card.add_limitation("This model is trained on synthetic data and may not perform well on real-world data.")
    model_card.add_limitation("The model has not been tested for fairness across different demographic groups.")
    
    # Set intended use
    model_card.set_intended_use("This model is intended for educational purposes only.")
    
    # Save the model card
    model_card_path = output_dir / "example_model_1.card.json"
    model_card.save(model_card_path)
    logger.info(f"Model card saved to {model_card_path}")
    
    # Save as Markdown
    markdown_path = output_dir / "example_model_1.md"
    with open(markdown_path, "w") as f:
        f.write(model_card.to_markdown())
    logger.info(f"Model card markdown saved to {markdown_path}")
    
    # Export as HTML
    html_path = output_dir / "example_model_1.html"
    html_content = export_model_card_html(model_card_path, html_path)
    logger.info(f"Model card HTML exported to {html_path}")
    
    # Print summary
    logger.info("Model Card Summary:")
    print_model_card_summary(model_card_path)
    
    return model_card_path

def example_2_generated_model_card():
    """Example 2: Generating a model card from a trained model."""
    logger.info("Example 2: Generating a model card from a trained model")
    
    # Create output directory
    output_dir = Path("examples/output/model_cards")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    X_train, y_train, X_test, y_test = create_sample_data()
    
    # Train a model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"Model metrics: {metrics}")
    
    # Create a model card generator
    generator = ModelCardGenerator()
    
    # Generate a model card
    model_card = generator.generate_from_training(
        model=model,
        model_id="example_model_2",
        X_train=X_train,
        y_train=y_train,
        metrics=metrics,
        description="A model generated from sample data",
        author="NexusML Team",
        intended_use="This model is intended for demonstration purposes only."
    )
    
    # Save the model card
    model_card_path = output_dir / "example_model_2.card.json"
    model_card.save(model_card_path)
    logger.info(f"Generated model card saved to {model_card_path}")
    
    # Save as Markdown
    markdown_path = output_dir / "example_model_2.md"
    with open(markdown_path, "w") as f:
        f.write(model_card.to_markdown())
    logger.info(f"Generated model card markdown saved to {markdown_path}")
    
    # Export as HTML
    html_path = output_dir / "example_model_2.html"
    html_content = export_model_card_html(model_card_path, html_path)
    logger.info(f"Generated model card HTML exported to {html_path}")
    
    # Print summary
    logger.info("Generated Model Card Summary:")
    print_model_card_summary(model_card_path)
    
    return model_card_path

def example_3_model_card_with_pipeline():
    """Example 3: Using model cards with the pipeline system."""
    logger.info("Example 3: Using model cards with the pipeline system")
    
    # Create output directory
    output_dir = Path("examples/output/model_cards")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import pipeline components
    try:
        from nexusml.src.pipeline.stages.model_saving import ModelCardSavingStage
        from nexusml.src.pipeline.context import PipelineContext
        
        # Create sample data
        X_train, y_train, X_test, y_test = create_sample_data()
        
        # Train a model
        model = train_model(X_train, y_train)
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Create a pipeline context
        context = PipelineContext()
        context.set("trained_model", model)
        context.set("evaluation_results", {"overall": metrics})
        
        # Create a model saving stage
        model_saving_stage = ModelCardSavingStage()
        
        # Create the output path
        model_path = output_dir / "pipeline_model.pkl"
        
        # Execute the stage
        model_saving_stage.execute(context, output_path=model_path)
        
        logger.info(f"Model saved with model card to {model_path}")
        
        # Check if the model card was created
        model_card_path = model_path.with_suffix(".card.json")
        markdown_path = model_path.with_suffix(".md")
        
        if model_card_path.exists() and markdown_path.exists():
            logger.info(f"Model card created at {model_card_path} and {markdown_path}")
            
            # Print summary
            logger.info("Pipeline Model Card Summary:")
            print_model_card_summary(model_card_path)
            
            return model_card_path
        else:
            logger.error("Failed to create model card with pipeline")
            return None
    except ImportError as e:
        logger.error(f"Failed to import pipeline components: {e}")
        logger.info("Skipping pipeline example")
        return None

def main():
    """Run all examples."""
    logger.info("Starting Model Card Examples")
    
    # Example 1: Creating a model card manually
    example_1_path = example_1_manual_model_card()
    
    # Example 2: Generating a model card from a trained model
    example_2_path = example_2_generated_model_card()
    
    # Example 3: Using model cards with the pipeline system
    example_3_path = example_3_model_card_with_pipeline()
    
    logger.info("Model Card Examples Completed")
    
    # Print paths to all created model cards
    logger.info("Created Model Cards:")
    if example_1_path:
        logger.info(f"  - Example 1: {example_1_path}")
    if example_2_path:
        logger.info(f"  - Example 2: {example_2_path}")
    if example_3_path:
        logger.info(f"  - Example 3: {example_3_path}")

if __name__ == "__main__":
    main()