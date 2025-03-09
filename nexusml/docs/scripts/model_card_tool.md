# Utility Script: model_card_tool.py

## Overview

The `model_card_tool.py` script is a command-line utility for working with model cards in the NexusML system. Model cards are structured documents that provide essential information about machine learning models, including their intended use, performance metrics, limitations, and ethical considerations. This tool allows users to view model card summaries in the terminal and export model cards to HTML format for better visualization and sharing.

Key features include:

1. **Model Card Viewing**: Display a summary of a model card in the terminal
2. **HTML Export**: Export a model card to HTML format for better visualization
3. **Command-Line Interface**: Easy-to-use CLI with subcommands
4. **Path Handling**: Flexible input and output path handling

## Usage

```bash
python -m nexusml.scripts.model_card_tool [command] [options]
```

### Commands

#### View Command

View a model card summary in the terminal:

```bash
python -m nexusml.scripts.model_card_tool view PATH
```

**Arguments:**

- `PATH`: Path to the model card JSON file

**Example:**
```bash
python -m nexusml.scripts.model_card_tool view outputs/models/equipment_classifier_model_card.json
```

#### Export Command

Export a model card to HTML format:

```bash
python -m nexusml.scripts.model_card_tool export PATH [--output OUTPUT]
```

**Arguments:**

- `PATH`: Path to the model card JSON file
- `--output`, `-o`: Path to save the HTML file (optional, defaults to the same path with .html extension)

**Example:**
```bash
# Export with default output path
python -m nexusml.scripts.model_card_tool export outputs/models/equipment_classifier_model_card.json

# Export with custom output path
python -m nexusml.scripts.model_card_tool export outputs/models/equipment_classifier_model_card.json --output documentation/model_card.html
```

## Functions

### `main()`

Main entry point for the model card tool.

**Example:**
```python
from nexusml.scripts.model_card_tool import main

# Run the model card tool
main()
```

**Notes:**

- This function sets up the argument parser with subcommands for viewing and exporting model cards
- It handles the command-line arguments and calls the appropriate functions based on the command
- If no valid command is provided, it prints the help message and exits with a non-zero status code

## Model Card Format

The tool works with model cards in JSON format. A model card typically includes the following information:

1. **Model Details**: Name, version, type, and other basic information
2. **Intended Use**: The intended use cases and users of the model
3. **Factors**: Factors that may affect model performance
4. **Metrics**: Performance metrics and evaluation results
5. **Evaluation Data**: Information about the data used for evaluation
6. **Training Data**: Information about the data used for training
7. **Quantitative Analyses**: Quantitative analyses of model performance
8. **Ethical Considerations**: Ethical considerations related to the model
9. **Caveats and Recommendations**: Caveats and recommendations for using the model

Example model card JSON structure:
```json
{
  "model_details": {
    "name": "Equipment Classifier",
    "version": "1.0.0",
    "type": "Classification",
    "description": "A model for classifying equipment based on descriptions"
  },
  "intended_use": {
    "primary_uses": ["Equipment classification for facility management"],
    "primary_users": ["Facility managers", "Asset managers"],
    "out_of_scope_uses": ["Critical safety applications"]
  },
  "factors": {
    "relevant_factors": ["Equipment type", "Description quality"],
    "evaluation_factors": ["Equipment category distribution"]
  },
  "metrics": {
    "performance_measures": [
      {"name": "Accuracy", "value": 0.92},
      {"name": "F1 Score", "value": 0.91},
      {"name": "Precision", "value": 0.90},
      {"name": "Recall", "value": 0.89}
    ],
    "decision_thresholds": {"confidence": 0.5}
  },
  "evaluation_data": {
    "dataset": "Equipment Test Dataset",
    "motivation": "Representative sample of equipment descriptions",
    "preprocessing": "Standard text preprocessing"
  },
  "training_data": {
    "dataset": "Equipment Training Dataset",
    "motivation": "Comprehensive coverage of equipment types",
    "preprocessing": "Standard text preprocessing"
  },
  "quantitative_analyses": {
    "unitary_results": "Performance is consistent across equipment types",
    "intersectional_results": "No significant performance differences between categories"
  },
  "ethical_considerations": {
    "ethical_risks": "Misclassification may lead to incorrect maintenance schedules",
    "mitigation_strategies": "Regular model updates and human review"
  },
  "caveats_and_recommendations": {
    "caveats": "Model performance depends on description quality",
    "recommendations": "Provide detailed equipment descriptions for best results"
  }
}
```

## HTML Output

When exporting a model card to HTML, the tool generates a well-formatted HTML document that presents the model card information in a more readable and visually appealing format. The HTML output includes:

1. **Header**: Model name and version
2. **Sections**: Each section of the model card is presented in its own section
3. **Tables**: Performance metrics and other tabular data are presented in tables
4. **Styling**: Basic CSS styling for better readability

Example HTML output structure:
```html
<!DOCTYPE html>
<html>
<head>
    <title>Model Card: Equipment Classifier</title>
    <style>
        /* CSS styling */
    </style>
</head>
<body>
    <h1>Model Card: Equipment Classifier v1.0.0</h1>
    
    <h2>Model Details</h2>
    <p>Type: Classification</p>
    <p>Description: A model for classifying equipment based on descriptions</p>
    
    <h2>Intended Use</h2>
    <h3>Primary Uses</h3>
    <ul>
        <li>Equipment classification for facility management</li>
    </ul>
    <!-- More sections -->
    
    <h2>Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Accuracy</td>
            <td>0.92</td>
        </tr>
        <!-- More metrics -->
    </table>
    
    <!-- More sections -->
</body>
</html>
```

## Dependencies

- **argparse**: Standard library module for command-line argument parsing
- **sys**: Standard library module for system-specific parameters and functions
- **pathlib**: Standard library module for path manipulation
- **nexusml.core.model_card.model_card**: Module for working with model cards
- **nexusml.core.model_card.viewer**: Module for viewing and exporting model cards

## Notes and Warnings

- The script requires the NexusML package to be installed or available in the Python path
- The script expects model cards to be in JSON format
- When exporting to HTML, if no output path is specified, the script will use the same path as the input file but with a .html extension
- The script will overwrite existing files without warning
- The script does not currently support creating or editing model cards, only viewing and exporting them
- The HTML export functionality requires the model_card.viewer module, which may have additional dependencies
- The script is designed to be run from the command line, but can also be imported and used programmatically
