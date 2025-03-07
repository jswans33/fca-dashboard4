"""
Fix Experiment Template Script

This script updates the experiment_template.ipynb notebook to use the enhanced data loader
from the notebook_utils module, fixing the data path issue and making the notebook more modular.
"""

import json
import os
from pathlib import Path


def fix_notebook():
    """Fix the experiment_template.ipynb notebook."""
    notebook_path = Path(__file__).parent / "experiment_template.ipynb"

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return False

    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Find the cell with the data loading code
    data_loading_cell_index = None
    for i, cell in enumerate(notebook["cells"]):
        if cell[
            "cell_type"
        ] == "code" and 'data_path = "../examples/sample_data.xlsx"' in "".join(
            cell["source"]
        ):
            data_loading_cell_index = i
            break

    if data_loading_cell_index is None:
        print("Error: Could not find the data loading cell in the notebook")
        return False

    # Replace the data loading cell with the enhanced version
    new_data_loading_code = [
        "# Import the data loading utility from notebook_utils\n",
        "from nexusml.utils.notebook_utils import discover_and_load_data\n",
        "\n",
        "# Discover and load data\n",
        "data, data_path = discover_and_load_data()\n",
        "\n",
        "# Display the first few rows\n",
        'print(f"Data shape: {data.shape}")\n',
        "data.head()",
    ]

    notebook["cells"][data_loading_cell_index]["source"] = new_data_loading_code

    # Save the updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)

    print(f"✅ Successfully updated {notebook_path}")
    print("The notebook now uses the enhanced data loader from notebook_utils.")
    print("This fixes the data path issue and makes the notebook more modular.")

    return True


if __name__ == "__main__":
    fix_notebook()
