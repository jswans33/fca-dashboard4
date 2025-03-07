"""
Fix Data Path Script for NexusML Notebooks

This script provides a simple fix for the data path issue in the experiment_template.ipynb notebook.
Run this script before running the notebook cells to ensure the data path is correct.
"""

import json
import os
import sys

import nbformat


def fix_notebook_data_path(notebook_path):
    """
    Fix the data path in the specified notebook.

    Args:
        notebook_path: Path to the notebook file
    """
    try:
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Find the cell with the data path definition
        for cell in notebook.cells:
            if (
                cell.cell_type == "code"
                and 'data_path = "../examples/sample_data.xlsx"' in cell.source
            ):
                # Replace the incorrect path with the correct one
                cell.source = cell.source.replace(
                    'data_path = "../examples/sample_data.xlsx"',
                    'data_path = "../../examples/sample_data.xlsx"  # Fixed path',
                )
                print(f"✅ Fixed data path in notebook: {notebook_path}")
                break
        else:
            print(f"⚠️ Could not find data path definition in notebook: {notebook_path}")
            return False

        # Save the modified notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        return True

    except Exception as e:
        print(f"❌ Error fixing notebook: {e}")
        return False


def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the experiment template notebook
    notebook_path = os.path.join(script_dir, "experiment_template.ipynb")

    # Fix the notebook
    if fix_notebook_data_path(notebook_path):
        print("\nNotebook updated successfully!")
        print("You can now run the notebook cells without the FileNotFoundError.")
    else:
        print("\nFailed to update the notebook.")
        print("You can manually update the data path in the notebook:")
        print('Change: data_path = "../examples/sample_data.xlsx"')
        print('To:     data_path = "../../examples/sample_data.xlsx"')


if __name__ == "__main__":
    main()
