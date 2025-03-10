"""
Model Serializer Component

This module provides a standard implementation of the ModelSerializer interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseModelSerializer

# Set up logging
logger = logging.getLogger(__name__)


class PickleModelSerializer(BaseModelSerializer):
    """
    Implementation of the ModelSerializer interface using pickle.

    This class serializes and deserializes models using the pickle module,
    with configuration provided by the ConfigurationProvider.
    """

    def __init__(
        self,
        name: str = "PickleModelSerializer",
        description: str = "Model serializer using pickle",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the PickleModelSerializer.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = config_provider or ConfigurationProvider()

        # Create a default serialization configuration
        self.config = {
            "serialization": {
                "default_directory": "nexusml/output/models",
                "protocol": pickle.HIGHEST_PROTOCOL,
                "compress": True,
                "file_extension": ".pkl",
            }
        }

        # Try to update from configuration provider if available
        try:
            # Check if there's a classification section in the config
            if hasattr(self._config_provider.config, "classification"):
                classifier_config = (
                    self._config_provider.config.classification.model_dump()
                )
                if "serialization" in classifier_config:
                    self.config.update(classifier_config["serialization"])
                    logger.info(
                        "Updated serialization configuration from classification section"
                    )
            logger.debug(f"Using serialization configuration: {self.config}")
        except Exception as e:
            logger.warning(f"Could not load serialization configuration: {e}")
            logger.info("Using default serialization configuration")

        logger.info(f"Initialized {name}")

    def save_model(self, model: Pipeline, path: Union[str, Path], **kwargs) -> None:
        """
        Save a trained model to disk.

        Args:
            model: Trained model pipeline to save.
            path: Path where the model should be saved.
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

            # Get serialization settings
            serialization_settings = self.config.get("serialization", {})
            protocol = kwargs.get(
                "protocol",
                serialization_settings.get("protocol", pickle.HIGHEST_PROTOCOL),
            )
            compress = kwargs.get(
                "compress", serialization_settings.get("compress", True)
            )

            # Add file extension if not present
            file_extension = serialization_settings.get("file_extension", ".pkl")
            if not str(path).endswith(file_extension):
                path = Path(str(path) + file_extension)

            # Log serialization parameters
            logger.debug(f"Saving model to {path}")
            logger.debug(f"Using protocol={protocol}, compress={compress}")

            # Save the model using pickle
            with open(path, "wb") as f:
                pickle.dump(model, f, protocol=protocol)

            logger.info(f"Model saved successfully to {path}")

            # Save metadata if requested
            if kwargs.get("save_metadata", False):
                metadata = kwargs.get("metadata", {})
                metadata_path = path.with_suffix(".meta.json")

                import json

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"Model metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise IOError(f"Error saving model: {str(e)}") from e

    def load_model(self, path: Union[str, Path], **kwargs) -> Pipeline:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model.
            **kwargs: Additional arguments for loading.

        Returns:
            Loaded model pipeline.

        Raises:
            IOError: If the model cannot be loaded.
            ValueError: If the loaded file is not a valid model.
        """
        try:
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Add file extension if not present and file doesn't exist
            if not path.exists():
                file_extension = self.config.get("serialization", {}).get(
                    "file_extension", ".pkl"
                )
                if not str(path).endswith(file_extension):
                    path = Path(str(path) + file_extension)

            # Check if the file exists
            if not path.exists():
                raise FileNotFoundError(f"Model file not found at {path}")

            logger.debug(f"Loading model from {path}")

            # Load the model using pickle
            with open(path, "rb") as f:
                model = pickle.load(f)

            # Verify that the loaded object is a Pipeline
            if not isinstance(model, Pipeline):
                raise ValueError(f"Loaded object is not a Pipeline: {type(model)}")

            logger.info(f"Model loaded successfully from {path}")

            # Load metadata if it exists
            metadata_path = path.with_suffix(".meta.json")
            if metadata_path.exists() and kwargs.get("load_metadata", False):
                import json

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                logger.info(f"Model metadata loaded from {metadata_path}")

                # If a metadata_callback is provided, call it with the metadata
                metadata_callback = kwargs.get("metadata_callback")
                if metadata_callback and callable(metadata_callback):
                    metadata_callback("metadata", metadata)
                # Otherwise, store metadata in kwargs for backward compatibility
                elif "metadata" not in kwargs:
                    kwargs["metadata"] = metadata

            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise IOError(f"Error loading model: {str(e)}") from e

    def list_saved_models(
        self, directory: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        List all saved models in the specified directory.

        Args:
            directory: Directory to search for models. If None, uses the default directory.

        Returns:
            Dictionary mapping model names to their metadata.

        Raises:
            IOError: If the directory cannot be accessed.
        """
        try:
            # Use default directory if none provided
            default_dir = self.config.get("serialization", {}).get(
                "default_directory", "nexusml/output/models"
            )
            directory_path = directory if directory is not None else default_dir

            # Convert to Path object if it's a string
            if isinstance(directory_path, str):
                directory_path = Path(directory_path)

            # Create directory if it doesn't exist
            if directory_path is not None:
                directory_path.mkdir(parents=True, exist_ok=True)
            else:
                # Fallback to a default directory if somehow we still have None
                directory_path = Path("nexusml/output/models")
                directory_path.mkdir(parents=True, exist_ok=True)

            # Get file extension
            file_extension = self.config.get("serialization", {}).get(
                "file_extension", ".pkl"
            )

            # Find all model files
            model_files = list(directory_path.glob(f"*{file_extension}"))

            # Create result dictionary
            result = {}

            for model_file in model_files:
                model_name = model_file.stem

                # Get file stats
                stats = model_file.stat()

                # Check for metadata file
                metadata_path = model_file.with_suffix(".meta.json")
                metadata = None

                if metadata_path.exists():
                    import json

                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                result[model_name] = {
                    "path": str(model_file),
                    "size_bytes": stats.st_size,
                    "modified_time": stats.st_mtime,
                    "metadata": metadata,
                }

            logger.info(f"Found {len(result)} saved models in {directory_path}")
            return result

        except Exception as e:
            logger.error(f"Error listing saved models: {str(e)}")
            raise IOError(f"Error listing saved models: {str(e)}") from e

    def delete_model(self, path: Union[str, Path]) -> bool:
        """
        Delete a saved model.

        Args:
            path: Path to the model to delete.

        Returns:
            True if the model was deleted, False otherwise.

        Raises:
            IOError: If the model cannot be deleted.
        """
        try:
            # Convert path to Path object if it's a string
            if isinstance(path, str):
                path = Path(path)

            # Add file extension if not present and file doesn't exist
            if not path.exists():
                file_extension = self.config.get("serialization", {}).get(
                    "file_extension", ".pkl"
                )
                if not str(path).endswith(file_extension):
                    path = Path(str(path) + file_extension)

            # Check if the file exists
            if not path.exists():
                logger.warning(f"Model file not found at {path}")
                return False

            # Delete the model file
            path.unlink()

            # Delete metadata file if it exists
            metadata_path = path.with_suffix(".meta.json")
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted model metadata at {metadata_path}")

            logger.info(f"Deleted model at {path}")
            return True

        except Exception as e:
            logger.error(f"Error deleting model: {str(e)}")
            raise IOError(f"Error deleting model: {str(e)}") from e
