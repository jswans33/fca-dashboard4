"""
Model Card Generator Module

This module provides the ModelCardGenerator class for automatically
generating model cards during the model training process.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from nexusml.config.manager import ConfigurationManager
from nexusml.src.models.cards.model_card import ModelCard


class ModelCardGenerator:
    """
    Class for automatically generating model cards during model training.

    This class extracts relevant information from the training process
    and creates a standardized model card.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize a new model card generator.

        Args:
            config_manager: Configuration manager for loading configuration
        """
        self.config_manager = config_manager or ConfigurationManager()

    def generate_from_training(
        self,
        model: Any,
        model_id: str,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[Union[pd.Series, pd.DataFrame]] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[Union[pd.Series, pd.DataFrame]] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        data_source: Optional[str] = None,
        description: Optional[str] = None,
        author: Optional[str] = None,
        intended_use: Optional[str] = None,
    ) -> ModelCard:
        """
        Generate a model card from training data and results.

        Args:
            model: The trained model
            model_id: Unique identifier for the model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            metrics: Performance metrics
            parameters: Model hyperparameters
            data_source: Source of the training data
            description: Brief description of the model's purpose
            author: Author or team responsible for the model
            intended_use: Description of intended use cases

        Returns:
            Generated model card
        """
        # Create a new model card
        model_card = ModelCard(
            model_id=model_id,
            model_type=self._get_model_type(model),
            description=description,
            author=author,
            config_manager=self.config_manager,
        )

        # Add training data information
        if X_train is not None:
            model_card.add_training_data_info(
                source=data_source,
                size=len(X_train),
                features=(
                    X_train.columns.tolist()
                    if isinstance(X_train, pd.DataFrame)
                    else None
                ),
                target=y_train.name if isinstance(y_train, pd.Series) else None,
            )

        # Add model parameters
        if parameters:
            model_card.add_parameters(parameters)
        elif hasattr(model, "get_params"):
            try:
                model_card.add_parameters(model.get_params())
            except Exception:
                # If get_params fails, try to extract parameters from the model object
                params = {}
                for attr_name in dir(model):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr_value = getattr(model, attr_name)
                        if isinstance(
                            attr_value, (int, float, str, bool, list, dict, tuple)
                        ):
                            params[attr_name] = attr_value
                    except Exception:
                        pass
                model_card.add_parameters(params)

        # Add metrics
        if metrics:
            model_card.add_metrics(metrics)

        # Add intended use
        if intended_use:
            model_card.set_intended_use(intended_use)
        else:
            model_card.set_intended_use(
                "This model is designed for classifying equipment based on descriptions and other features."
            )

        # Add common limitations
        model_card.add_limitation(
            "This model may not perform well on data that is significantly different from the training data."
        )
        model_card.add_limitation(
            "The model's performance may degrade over time as data distributions change."
        )

        return model_card

    def _get_model_type(self, model: Any) -> str:
        """
        Determine the type of model.

        Args:
            model: The model object

        Returns:
            String representation of the model type
        """
        # Try to get the model type from the class name
        model_type = type(model).__name__

        # Check for common model types
        if hasattr(model, "estimators_") and hasattr(model, "estimator"):
            return "ensemble"
        elif "RandomForest" in model_type:
            return "random_forest"
        elif "GradientBoosting" in model_type:
            return "gradient_boosting"
        elif "LogisticRegression" in model_type:
            return "logistic_regression"
        elif "SVC" in model_type or "SVM" in model_type:
            return "support_vector_machine"
        elif "DecisionTree" in model_type:
            return "decision_tree"
        elif "KNeighbors" in model_type:
            return "k_nearest_neighbors"
        elif "NeuralNetwork" in model_type or "MLPClassifier" in model_type:
            return "neural_network"
        elif "Pipeline" in model_type:
            # For scikit-learn pipelines, try to get the final estimator type
            if hasattr(model, "steps") and model.steps:
                final_step = model.steps[-1]
                if len(final_step) > 1:
                    return self._get_model_type(final_step[1])

        # Default to the class name if no specific type is identified
        return model_type.lower()
