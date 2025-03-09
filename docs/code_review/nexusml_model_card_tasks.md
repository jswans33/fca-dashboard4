Based on my review of the existing code, I can see that there's already a model card system in place, but it's relatively simple. The current implementation:

1. Has a `ModelCardConfig` class in `nexusml/config/model_card.py` that provides access to model card configuration.
2. Has a JSON schema for model cards in `nexusml/config/schemas/model_card_schema.json`.
3. Has a `ModelCardSavingStage` class in `nexusml/core/pipeline/stages/model_saving.py` that saves models with model cards.

The current implementation generates a Markdown file with model details, but it doesn't have a structured JSON representation of the model card that could be used for programmatic access or visualization.

Based on the detailed tasks document, the pending model card system tasks include:
1. Define model card schema
2. Implement model card generation during training
3. Add model card validation
4. Create model card visualization tools

Let me outline a simple implementation plan that builds on the existing code but adds the missing functionality.

Based on my review of the existing code, I can see that there's already a partial model card system in place. Let me outline a simple implementation plan that builds on the existing code to complete the model card system:

## Simple Model Card System Implementation Plan

### 1. Create a Structured Model Card Class

Currently, the system generates Markdown model cards, but there's no structured representation. Let's create a proper `ModelCard` class:

```python
# nexusml/core/model_card/model_card.py
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
        """Initialize a new model card."""
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
    
    # Methods for adding data to the model card
    def add_training_data_info(self, source=None, size=None, features=None, target=None):
        """Add information about the training data."""
        # Implementation details...
        
    def add_parameters(self, parameters):
        """Add model hyperparameters."""
        self.data["parameters"] = parameters
    
    def add_metrics(self, metrics):
        """Add performance metrics."""
        self.data["metrics"] = metrics
    
    # Methods for validation and serialization
    def validate(self):
        """Validate the model card against the schema."""
        # Implementation using existing schema...
        
    def to_dict(self):
        """Convert the model card to a dictionary."""
        return self.data.copy()
    
    def to_json(self, indent=2):
        """Convert the model card to a JSON string."""
        return json.dumps(self.data, indent=indent)
    
    def to_markdown(self):
        """Convert the model card to a Markdown string."""
        # Implementation similar to existing _create_model_card method...
        
    def save(self, path):
        """Save the model card to a file."""
        # Implementation details...
        
    @classmethod
    def load(cls, path, config_manager=None):
        """Load a model card from a file."""
        # Implementation details...
```

### 2. Create a Model Card Generator

```python
# nexusml/core/model_card/generator.py
"""
Model Card Generator Module

This module provides the ModelCardGenerator class for automatically
generating model cards during the model training process.
"""

from typing import Any, Dict, Optional, Union

import pandas as pd

from nexusml.config.manager import ConfigurationManager
from nexusml.core.model_card.model_card import ModelCard

class ModelCardGenerator:
    """
    Class for automatically generating model cards during model training.
    """
    
    def __init__(self, config_manager=None):
        """Initialize a new model card generator."""
        self.config_manager = config_manager or ConfigurationManager()
    
    def generate_from_training(
        self,
        model,
        model_id,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        metrics=None,
        parameters=None,
        data_source=None,
        description=None,
        author=None
    ):
        """Generate a model card from training data and results."""
        # Create a new model card
        model_card = ModelCard(
            model_id=model_id,
            model_type=self._get_model_type(model),
            description=description,
            author=author,
            config_manager=self.config_manager
        )
        
        # Add training data information
        model_card.add_training_data_info(
            source=data_source,
            size=len(X_train) if X_train is not None else None,
            features=X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None,
            target=y_train.name if isinstance(y_train, pd.Series) else None
        )
        
        # Add model parameters
        if parameters:
            model_card.add_parameters(parameters)
        elif hasattr(model, 'get_params'):
            model_card.add_parameters(model.get_params())
        
        # Add metrics
        if metrics:
            model_card.add_metrics(metrics)
        
        return model_card
    
    def _get_model_type(self, model):
        """Determine the type of model."""
        # Implementation details...
```

### 3. Update the Model Saving Stage

Modify the existing `ModelCardSavingStage` to use the new `ModelCard` class:

```python
# In nexusml/core/pipeline/stages/model_saving.py

from nexusml.core.model_card.model_card import ModelCard
from nexusml.core.model_card.generator import ModelCardGenerator

class ModelCardSavingStage(BaseModelSavingStage):
    # ...existing code...
    
    def save_model(self, model, path, metadata, **kwargs):
        """Save a model with a model card."""
        try:
            # ...existing code for saving the model...
            
            # Create a model card using the generator
            generator = ModelCardGenerator(config_manager=self.config_manager)
            model_card = generator.generate_from_training(
                model=model,
                model_id=kwargs.get("model_id", Path(path).stem),
                X_train=kwargs.get("X_train"),
                y_train=kwargs.get("y_train"),
                metrics=metadata.get("evaluation_results", {}).get("overall", {}),
                parameters=kwargs.get("parameters"),
                description=kwargs.get("model_description"),
                author=kwargs.get("model_authors", ["NexusML Team"])
            )
            
            # Save the model card as JSON
            model_card_json_path = Path(path).with_suffix(".card.json")
            with open(model_card_json_path, "w") as f:
                f.write(model_card.to_json())
            
            # Save the model card as Markdown
            model_card_md_path = Path(path).with_suffix(".md")
            with open(model_card_md_path, "w") as f:
                f.write(model_card.to_markdown())
            
            # ...existing code for saving metadata...
            
        except Exception as e:
            raise IOError(f"Failed to save model: {e}")
```

### 4. Create a Simple Model Card Viewer

```python
# nexusml/core/model_card/viewer.py
"""
Model Card Viewer Module

This module provides utilities for viewing and visualizing model cards.
"""

import json
from pathlib import Path
from typing import Optional, Union

from nexusml.core.model_card.model_card import ModelCard

def print_model_card_summary(model_card_path: Union[str, Path]) -> None:
    """
    Print a summary of a model card to the console.
    
    Args:
        model_card_path: Path to the model card JSON file
    """
    model_card = ModelCard.load(model_card_path)
    
    print(f"Model Card: {model_card.data['model_id']} (v{model_card.data['version']})")
    print(f"Type: {model_card.data['model_type']}")
    print(f"Created: {model_card.data['creation_date']}")
    
    if "description" in model_card.data:
        print(f"\nDescription: {model_card.data['description']}")
    
    if "metrics" in model_card.data and model_card.data["metrics"]:
        print("\nPerformance Metrics:")
        for metric, value in model_card.data["metrics"].items():
            print(f"  {metric}: {value}")
    
    if "limitations" in model_card.data and model_card.data["limitations"]:
        print("\nLimitations:")
        for limitation in model_card.data["limitations"]:
            print(f"  - {limitation}")

def export_model_card_html(model_card_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
    """
    Export a model card as an HTML file.
    
    Args:
        model_card_path: Path to the model card JSON file
        output_path: Path to save the HTML file (optional)
        
    Returns:
        Path to the saved HTML file
    """
    model_card = ModelCard.load(model_card_path)
    
    # Generate a simple HTML representation
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Card: {model_card.data['model_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .section {{ margin-bottom: 20px; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Model Card: {model_card.data['model_id']}</h1>
    <div class="section">
        <p><strong>Version:</strong> {model_card.data['version']}</p>
        <p><strong>Type:</strong> {model_card.data['model_type']}</p>
        <p><strong>Created:</strong> {model_card.data['creation_date']}</p>
    </div>
    """
    
    if "description" in model_card.data:
        html += f"""
    <div class="section">
        <h2>Description</h2>
        <p>{model_card.data['description']}</p>
    </div>
        """
    
    if "metrics" in model_card.data and model_card.data["metrics"]:
        html += """
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metrics">
        """
        
        for metric, value in model_card.data["metrics"].items():
            html += f"""
            <div class="metric">
                <p><strong>{metric}</strong></p>
                <p>{value}</p>
            </div>
            """
        
        html += """
        </div>
    </div>
        """
    
    html += """
</body>
</html>
    """
    
    # Save the HTML file if output_path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(html)
        
        return str(output_path)
    
    return html
```

### 5. Create a Simple Command-Line Tool

```python
# nexusml/scripts/model_card_tool.py
"""
Model Card Tool

A command-line tool for working with model cards.
"""

import argparse
import sys
from pathlib import Path

from nexusml.core.model_card.model_card import ModelCard
from nexusml.core.model_card.viewer import print_model_card_summary, export_model_card_html

def main():
    """Main entry point for the model card tool."""
    parser = argparse.ArgumentParser(description="Model Card Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View a model card")
    view_parser.add_argument("path", help="Path to the model card JSON file")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export a model card to HTML")
    export_parser.add_argument("path", help="Path to the model card JSON file")
    export_parser.add_argument("--output", "-o", help="Path to save the HTML file")
    
    args = parser.parse_args()
    
    if args.command == "view":
        print_model_card_summary(args.path)
    elif args.command == "export":
        output_path = args.output or Path(args.path).with_suffix(".html")
        html_path = export_model_card_html(args.path, output_path)
        print(f"Exported model card to {html_path}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Implementation Steps

1. Create the directory structure:
   ```
   nexusml/core/model_card/
   ├── __init__.py
   ├── model_card.py
   ├── generator.py
   └── viewer.py
   ```

2. Implement the `ModelCard` class in `model_card.py`

3. Implement the `ModelCardGenerator` class in `generator.py`

4. Update the `ModelCardSavingStage` class to use the new model card system

5. Implement the model card viewer in `viewer.py`

6. Create a command-line tool in `scripts/model_card_tool.py`

This implementation builds on the existing code but adds structured model cards with JSON serialization, a generator for creating model cards during training, and simple visualization tools. It's a relatively simple approach that could be implemented in a short time frame while still providing value.