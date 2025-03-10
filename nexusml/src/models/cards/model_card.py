"""
Model Card Module

This module provides the ModelCard class for creating, validating, and
managing model cards that document machine learning models.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nexusml.config.manager import ConfigurationManager


class ModelCard:
    """
    Class representing a model card for documenting machine learning models.
    
    Model cards provide standardized documentation for models, including
    information about their purpose, performance, limitations, and other
    metadata.
    """
    
    def __init__(
        self,
        model_id: str,
        model_type: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        author: Optional[str] = None,
        config_manager: Optional[ConfigurationManager] = None
    ):
        """
        Initialize a new model card.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., random_forest, gradient_boosting)
            version: Version of the model
            description: Brief description of the model's purpose
            author: Author or team responsible for the model
            config_manager: Configuration manager for loading schema
        """
        self.config_manager = config_manager or ConfigurationManager()
        
        # Initialize required fields
        self.data = {
            "model_id": model_id,
            "model_type": model_type,
            "creation_date": datetime.now().isoformat(),
            "version": version,
            "metrics": {}
        }
        
        # Add optional fields if provided
        if description:
            self.data["description"] = description
        if author:
            self.data["author"] = author
            
        # Initialize other sections
        self.data["training_data"] = {}
        self.data["parameters"] = {}
        self.data["limitations"] = []
    
    def add_training_data_info(
        self,
        source: Optional[str] = None,
        size: Optional[int] = None,
        features: Optional[List[str]] = None,
        target: Optional[str] = None
    ) -> None:
        """
        Add information about the training data.
        
        Args:
            source: Source of the training data
            size: Number of samples in the training data
            features: List of features used for training
            target: Target variable
        """
        training_data = {}
        if source:
            training_data["source"] = source
        if size:
            training_data["size"] = size
        if features:
            training_data["features"] = features
        if target:
            training_data["target"] = target
            
        self.data["training_data"] = training_data
    
    def add_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Add model hyperparameters.
        
        Args:
            parameters: Dictionary of model hyperparameters
        """
        self.data["parameters"] = parameters
    
    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Add performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.data["metrics"] = metrics
    
    def add_limitation(self, limitation: str) -> None:
        """
        Add a known limitation of the model.
        
        Args:
            limitation: Description of a limitation
        """
        if "limitations" not in self.data:
            self.data["limitations"] = []
        self.data["limitations"].append(limitation)
    
    def set_intended_use(self, intended_use: str) -> None:
        """
        Set the intended use cases for the model.
        
        Args:
            intended_use: Description of intended use cases
        """
        self.data["intended_use"] = intended_use
    
    def validate(self) -> bool:
        """
        Validate the model card against the schema.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Load the schema
            schema_path = "config/schemas/model_card_schema.json"
            schema = self.config_manager.load_json_schema(schema_path)
            
            # Validate against the schema
            from jsonschema import validate
            validate(instance=self.data, schema=schema)
            return True
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model card to a dictionary.
        
        Returns:
            Dictionary representation of the model card
        """
        return self.data.copy()
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the model card to a JSON string.
        
        Args:
            indent: Number of spaces for indentation
            
        Returns:
            JSON string representation of the model card
        """
        return json.dumps(self.data, indent=indent)
    
    def to_markdown(self) -> str:
        """
        Convert the model card to a Markdown string.
        
        Returns:
            Markdown string representation of the model card
        """
        model_card = f"""# {self.data['model_id']}

## Model Details

- **Version:** {self.data['version']}
- **Type:** {self.data['model_type']}
- **Created:** {self.data['creation_date']}
"""

        if "description" in self.data:
            model_card += f"- **Description:** {self.data['description']}\n"
        
        if "author" in self.data:
            model_card += f"- **Author:** {self.data['author']}\n"
        
        # Add training data information
        if self.data["training_data"]:
            model_card += "\n## Training Data\n\n"
            
            if "source" in self.data["training_data"]:
                model_card += f"- **Source:** {self.data['training_data']['source']}\n"
            
            if "size" in self.data["training_data"]:
                model_card += f"- **Size:** {self.data['training_data']['size']} samples\n"
            
            if "features" in self.data["training_data"]:
                model_card += f"- **Features:** {', '.join(self.data['training_data']['features'])}\n"
            
            if "target" in self.data["training_data"]:
                model_card += f"- **Target:** {self.data['training_data']['target']}\n"
        
        # Add metrics
        if self.data["metrics"]:
            model_card += "\n## Performance Metrics\n\n"
            
            for metric, value in self.data["metrics"].items():
                model_card += f"- **{metric}:** {value}\n"
        
        # Add parameters
        if self.data["parameters"]:
            model_card += "\n## Model Parameters\n\n"
            
            for param, value in self.data["parameters"].items():
                model_card += f"- **{param}:** {value}\n"
        
        # Add limitations
        if self.data["limitations"]:
            model_card += "\n## Limitations\n\n"
            
            for limitation in self.data["limitations"]:
                model_card += f"- {limitation}\n"
        
        # Add intended use
        if "intended_use" in self.data:
            model_card += f"\n## Intended Use\n\n{self.data['intended_use']}\n"
        
        return model_card
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model card to a file.
        
        Args:
            path: Path to save the model card
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path], config_manager: Optional[ConfigurationManager] = None) -> 'ModelCard':
        """
        Load a model card from a file.
        
        Args:
            path: Path to the model card file
            config_manager: Configuration manager for loading schema
            
        Returns:
            Loaded model card
        """
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Create a new instance
        model_card = cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            version=data["version"],
            description=data.get("description"),
            author=data.get("author"),
            config_manager=config_manager
        )
        
        # Update with loaded data
        model_card.data = data
        
        return model_card
    
    def __str__(self) -> str:
        """
        Get a string representation of the model card.
        
        Returns:
            String representation of the model card
        """
        return f"ModelCard(id={self.data['model_id']}, type={self.data['model_type']}, version={self.data['version']})"