"""
Notebook Utilities

This module provides utility functions for use in Jupyter notebooks,
making them more modular and maintainable.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from nexusml.core.pipeline.components.data_loader import StandardDataLoader

# Set up logging
logger = logging.getLogger(__name__)


def setup_notebook_environment():
    """
    Set up the notebook environment with common configurations.

    This includes matplotlib settings, seaborn styling, etc.
    """
    # Set up matplotlib
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("notebook")

    # Return a dictionary of useful paths
    return {
        "project_root": get_project_root(),
        "data_dir": os.path.join(get_project_root(), "data"),
        "examples_dir": os.path.join(get_project_root(), "examples"),
        "outputs_dir": os.path.join(get_project_root(), "outputs"),
    }


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.

    Returns:
        Absolute path to the project root directory.
    """
    # Assuming this module is in nexusml/utils/
    module_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the project root
    return os.path.dirname(os.path.dirname(module_dir))


def discover_and_load_data(
    file_name: Optional[str] = None,
    search_paths: Optional[List[str]] = None,
    file_extensions: Optional[List[str]] = None,
    show_available: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Discover available data files and load the specified one.

    Args:
        file_name: Name of the file to load. If None, uses the first available file.
        search_paths: List of paths to search for data files. If None, uses default paths.
        file_extensions: List of file extensions to include. If None, uses defaults.
        show_available: Whether to print the list of available files.

    Returns:
        Tuple of (loaded DataFrame, file path)
    """
    # Create a data loader
    data_loader = StandardDataLoader()

    # Discover available data files
    available_files = data_loader.discover_data_files(
        search_paths=search_paths, file_extensions=file_extensions
    )

    if not available_files:
        raise FileNotFoundError("No data files found in the specified search paths")

    # Show available files if requested
    if show_available:
        print(f"Found {len(available_files)} data files:")
        for i, (name, path) in enumerate(available_files.items(), 1):
            print(f"  {i}. {name}: {path}")
        print()

    # Select the file to load
    if file_name is None:
        # Use the first available file
        file_name = list(available_files.keys())[0]

    if file_name not in available_files:
        raise FileNotFoundError(f"File not found: {file_name}")

    file_path = available_files[file_name]
    print(f"Loading data from: {file_name}")

    # Load the data
    data = data_loader.load_data(file_path)
    print(f"Data loaded successfully with shape: {data.shape}")

    return data, file_path


def explore_data(
    data: pd.DataFrame, show_summary: bool = True, show_missing: bool = True
) -> Dict:
    """
    Explore a DataFrame and return useful statistics.

    Args:
        data: DataFrame to explore
        show_summary: Whether to print summary statistics
        show_missing: Whether to print missing value information

    Returns:
        Dictionary of exploration results
    """
    results = {}

    # Data types
    print("Data types:")
    print(data.dtypes)
    print()
    results["dtypes"] = data.dtypes

    # Missing values
    if show_missing:
        missing = data.isnull().sum()
        missing_percent = (missing / len(data)) * 100
        missing_info = pd.DataFrame(
            {"Missing Values": missing, "Percent Missing": missing_percent}
        )
        print("Missing values:")
        print(missing_info[missing_info["Missing Values"] > 0])
        print()
        results["missing"] = missing_info

    # Summary statistics
    if show_summary:
        summary = data.describe()
        print("Summary statistics:")
        print(summary)
        print()
        results["summary"] = summary

    return results


def setup_pipeline_components():
    """
    Set up the standard pipeline components for a NexusML experiment.

    Returns:
        Dictionary containing the pipeline components
    """
    # Import the components we know exist
    from nexusml.core.di.container import DIContainer
    from nexusml.core.pipeline.components.data_loader import StandardDataLoader
    from nexusml.core.pipeline.context import PipelineContext
    from nexusml.core.pipeline.factory import PipelineFactory

    # Import interfaces
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

    # Create a registry and container
    registry = ComponentRegistry()
    container = DIContainer()

    # Register the data loader (we know this exists)
    registry.register(DataLoader, "csv", StandardDataLoader)
    registry.register(DataLoader, "excel", StandardDataLoader)
    registry.set_default_implementation(DataLoader, "excel")

    # Try to import and register other components
    component_imports = {
        "data_preprocessor": {
            "interface": DataPreprocessor,
            "implementation": "StandardDataPreprocessor",
            "module": "nexusml.core.pipeline.components.data_preprocessor",
        },
        "feature_engineer": {
            "interface": FeatureEngineer,
            "implementation": "StandardFeatureEngineer",
            "module": "nexusml.core.pipeline.components.feature_engineer",
        },
        "model_builder": {
            "interface": ModelBuilder,
            "implementation": "RandomForestModelBuilder",
            "module": "nexusml.core.pipeline.components.model_builder",
        },
        "model_trainer": {
            "interface": ModelTrainer,
            "implementation": "StandardModelTrainer",
            "module": "nexusml.core.pipeline.components.model_trainer",
        },
        "model_evaluator": {
            "interface": ModelEvaluator,
            "implementation": "EnhancedModelEvaluator",
            "module": "nexusml.core.pipeline.components.model_evaluator",
        },
        "model_serializer": {
            "interface": ModelSerializer,
            "implementation": "PickleModelSerializer",
            "module": "nexusml.core.pipeline.components.model_serializer",
        },
        # Predictor module doesn't exist yet, so commenting out
        # "predictor": {
        #     "interface": Predictor,
        #     "implementation": "StandardPredictor",
        #     "module": "nexusml.core.pipeline.components.predictor",
        # },
    }

    # Try to import and register each component
    for component_name, component_info in component_imports.items():
        try:
            # Dynamically import the module and get the implementation class
            module = __import__(
                component_info["module"], fromlist=[component_info["implementation"]]
            )
            implementation = getattr(module, component_info["implementation"])

            # Register the implementation
            registry.register(
                component_info["interface"], component_name, implementation
            )
            registry.set_default_implementation(
                component_info["interface"], component_name
            )
            logger.info(
                f"Registered {component_info['implementation']} for {component_name}"
            )
        except (ImportError, AttributeError) as e:
            logger.warning(f"Could not import {component_info['implementation']}: {e}")

    # Create a factory and orchestrator
    factory = PipelineFactory(registry, container)
    context = PipelineContext()
    orchestrator = PipelineOrchestrator(factory, context)

    return {
        "registry": registry,
        "container": container,
        "factory": factory,
        "context": context,
        "orchestrator": orchestrator,
    }


def visualize_metrics(metrics: Dict, figsize: Tuple[int, int] = (10, 6)):
    """
    Visualize model metrics.

    Args:
        metrics: Dictionary of metrics
        figsize: Figure size as (width, height)
    """
    # Create a bar chart of the metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    plt.figure(figsize=figsize)
    sns.barplot(x="Metric", y="Value", data=metrics_df)
    plt.title("Model Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(cm, figsize: Tuple[int, int] = (10, 8)):
    """
    Visualize a confusion matrix.

    Args:
        cm: Confusion matrix
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
