"""
Model Saving Stage Module

This module provides implementations of the ModelSavingStage interface for
saving trained models and associated metadata.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from sklearn.pipeline import Pipeline

from nexusml.config.manager import ConfigurationManager
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import BaseModelSavingStage
from nexusml.core.model_card.model_card import ModelCard
from nexusml.core.model_card.generator import ModelCardGenerator


class PickleModelSavingStage(BaseModelSavingStage):
    """
    Implementation of ModelSavingStage for saving models using pickle.
    """

    def __init__(
        self,
        name: str = "PickleModelSaving",
        description: str = "Saves models using pickle",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the pickle model saving stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading saving configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("trained_model") or context.has("model")

    def save_model(
        self, model: Pipeline, path: Union[str, Path], metadata: Dict[str, Any], **kwargs
    ) -> None:
        """
        Save a model using pickle.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            metadata: Model metadata to save.
            **kwargs: Additional arguments for saving.

        Raises:
            IOError: If the model cannot be saved.
        """
        try:
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save the model using pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)

            # Save metadata to a separate file
            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            raise IOError(f"Failed to save model: {e}")


class ModelCardSavingStage(BaseModelSavingStage):
    """
    Implementation of ModelSavingStage for saving models with model cards.
    """

    def __init__(
        self,
        name: str = "ModelCardSaving",
        description: str = "Saves models with model cards",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the model card saving stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading saving configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("trained_model") or context.has("model")

    def save_model(
        self, model: Pipeline, path: Union[str, Path], metadata: Dict[str, Any], **kwargs
    ) -> None:
        """
        Save a model with a model card.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            metadata: Model metadata to save.
            **kwargs: Additional arguments for saving.

        Raises:
            IOError: If the model cannot be saved.
        """
        try:
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Create parent directories if they don't exist
            path.parent.mkdir(parents=True, exist_ok=True)

            # Save the model using pickle
            with open(path, "wb") as f:
                pickle.dump(model, f)

            # Create a model card using the generator
            generator = ModelCardGenerator(config_manager=self.config_manager)
            model_card = generator.generate_from_training(
                model=model,
                model_id=kwargs.get("model_id", path.stem),
                X_train=kwargs.get("X_train"),
                y_train=kwargs.get("y_train"),
                metrics=metadata.get("evaluation_results", {}).get("overall", {}),
                parameters=kwargs.get("parameters"),
                description=kwargs.get("model_description",
                    "A machine learning model for equipment classification."),
                author=kwargs.get("model_authors", "NexusML Team"),
                intended_use=kwargs.get("intended_use",
                    "This model is designed for classifying equipment based on descriptions and other features.")
            )
            
            # Save the model card as JSON
            model_card_json_path = path.with_suffix(".card.json")
            model_card.save(model_card_json_path)
            
            # Save the model card as Markdown
            model_card_md_path = path.with_suffix(".md")
            with open(model_card_md_path, "w") as f:
                f.write(model_card.to_markdown())

            # Save metadata to a separate file
            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            raise IOError(f"Failed to save model: {e}")

    def _create_model_card(
        self, model: Pipeline, metadata: Dict[str, Any], **kwargs
    ) -> str:
        """
        Create a model card for the model.

        Args:
            model: Trained model pipeline.
            metadata: Model metadata.
            **kwargs: Additional arguments for creating the model card.

        Returns:
            Model card as a string.
        """
        # Get model card information from kwargs or config
        model_name = kwargs.get("model_name", self.config.get("model_name", "NexusML Model"))
        model_version = kwargs.get(
            "model_version", self.config.get("model_version", "1.0.0")
        )
        model_description = kwargs.get(
            "model_description",
            self.config.get(
                "model_description", "A machine learning model for equipment classification."
            ),
        )
        model_type = kwargs.get(
            "model_type",
            self.config.get("model_type", "Classification"),
        )
        model_authors = kwargs.get(
            "model_authors", self.config.get("model_authors", ["NexusML Team"])
        )
        model_license = kwargs.get(
            "model_license", self.config.get("model_license", "Proprietary")
        )

        # Get evaluation metrics from metadata
        evaluation_metrics = metadata.get("evaluation_results", {})
        overall_metrics = evaluation_metrics.get("overall", {})

        # Create the model card
        model_card = f"""# {model_name}

## Model Details

- **Version:** {model_version}
- **Type:** {model_type}
- **Description:** {model_description}
- **Authors:** {', '.join(model_authors)}
- **License:** {model_license}
- **Created:** {metadata.get('created_at', 'Unknown')}

## Intended Use

This model is designed for classifying equipment based on descriptions and other features.

## Training Data

The model was trained on a dataset of equipment descriptions and associated metadata.

## Evaluation Results

### Overall Metrics

"""

        # Add overall metrics
        if overall_metrics:
            for metric, value in overall_metrics.items():
                model_card += f"- **{metric}:** {value}\n"
        else:
            model_card += "No overall metrics available.\n"

        # Add per-column metrics
        model_card += "\n### Per-Column Metrics\n\n"
        for col, metrics in evaluation_metrics.items():
            if col == "overall" or col == "predictions" or col == "error_analysis":
                continue
            model_card += f"#### {col}\n\n"
            for metric, value in metrics.items():
                if metric == "classification_report" or metric == "confusion_matrix" or metric == "class_metrics":
                    continue
                model_card += f"- **{metric}:** {value}\n"
            model_card += "\n"

        # Add model parameters
        model_card += "\n## Model Parameters\n\n"
        try:
            # Try to get model parameters
            if hasattr(model, "get_params"):
                params = model.get_params()
                for param, value in params.items():
                    model_card += f"- **{param}:** {value}\n"
            else:
                model_card += "No model parameters available.\n"
        except Exception:
            model_card += "Error retrieving model parameters.\n"

        # Add limitations and ethical considerations
        model_card += """
## Limitations

This model may not perform well on data that is significantly different from the training data.

## Ethical Considerations

This model should be used responsibly and in accordance with applicable laws and regulations.
"""

        return model_card


class ConfigDrivenModelSavingStage(BaseModelSavingStage):
    """
    Implementation of ModelSavingStage that uses configuration for model saving.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenModelSaving",
        description: str = "Saves models based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the configuration-driven model saving stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading saving configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self._savers = {
            "pickle": PickleModelSavingStage(
                config=config, config_manager=config_manager
            ),
            "model_card": ModelCardSavingStage(
                config=config, config_manager=config_manager
            ),
        }

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("trained_model") or context.has("model")

    def save_model(
        self, model: Pipeline, path: Union[str, Path], metadata: Dict[str, Any], **kwargs
    ) -> None:
        """
        Save a model based on configuration.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
            metadata: Model metadata to save.
            **kwargs: Additional arguments for saving.

        Raises:
            IOError: If the model cannot be saved.
        """
        # Get the saving type from kwargs or config
        saving_type = kwargs.get(
            "saving_type", self.config.get("saving_type", "model_card")
        )

        # Get the appropriate saver
        if saving_type not in self._savers:
            raise ValueError(f"Unsupported saving type: {saving_type}")

        saver = self._savers[saving_type]

        # Save the model
        saver.save_model(model, path, metadata, **kwargs)