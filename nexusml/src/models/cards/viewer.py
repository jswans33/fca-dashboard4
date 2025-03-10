"""
Model Card Viewer Module

This module provides utilities for viewing and visualizing model cards.
"""

import json
from pathlib import Path
from typing import Optional, Union

from nexusml.src.models.cards.model_card import ModelCard


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


def export_model_card_html(
    model_card_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Export a model card as an HTML file.

    Args:
        model_card_path: Path to the model card JSON file
        output_path: Path to save the HTML file (optional)

    Returns:
        Path to the saved HTML file or HTML content if output_path is None
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

    if "training_data" in model_card.data and model_card.data["training_data"]:
        html += """
    <div class="section">
        <h2>Training Data</h2>
        <ul>
        """

        for key, value in model_card.data["training_data"].items():
            if key == "features" and isinstance(value, list):
                html += f"""
            <li><strong>{key}:</strong> {', '.join(value)}</li>
                """
            else:
                html += f"""
            <li><strong>{key}:</strong> {value}</li>
                """

        html += """
        </ul>
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

    if "parameters" in model_card.data and model_card.data["parameters"]:
        html += """
    <div class="section">
        <h2>Model Parameters</h2>
        <ul>
        """

        for param, value in model_card.data["parameters"].items():
            html += f"""
            <li><strong>{param}:</strong> {value}</li>
            """

        html += """
        </ul>
    </div>
        """

    if "limitations" in model_card.data and model_card.data["limitations"]:
        html += """
    <div class="section">
        <h2>Limitations</h2>
        <ul>
        """

        for limitation in model_card.data["limitations"]:
            html += f"""
            <li>{limitation}</li>
            """

        html += """
        </ul>
    </div>
        """

    if "intended_use" in model_card.data:
        html += f"""
    <div class="section">
        <h2>Intended Use</h2>
        <p>{model_card.data['intended_use']}</p>
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
