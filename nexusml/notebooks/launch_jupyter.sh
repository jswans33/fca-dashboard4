#!/bin/bash

# Script to launch Jupyter notebooks for NexusML
# This script should be run from Git Bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
NEXUSML_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create a dedicated runtime directory for Jupyter to avoid permission issues
mkdir -p "$PROJECT_ROOT/.venv/jupyter_runtime"

# Activate the virtual environment
source "$PROJECT_ROOT/.venv/Scripts/activate"

# Ensure UV is installed
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    pip install uv
fi

# Ensure Jupyter is installed
uv pip install jupyter

# Install required dependencies
uv pip install python-dotenv tqdm

# Install optional dependencies if needed
read -p "Do you want to install the anthropic package for Claude API integration? (y/n) " install_anthropic
if [[ $install_anthropic == "y" || $install_anthropic == "Y" ]]; then
    uv pip install anthropic
    echo "Anthropic package installed successfully."
fi

# Install nexusml in development mode if not already installed
if ! python -c "import nexusml" &> /dev/null; then
    echo "Installing nexusml package in development mode..."
    cd "$NEXUSML_ROOT" && uv pip install -e .
    cd "$SCRIPT_DIR"  # Return to the original directory
fi

# Print environment information
echo "Project root: $PROJECT_ROOT"
echo "NexusML root: $NEXUSML_ROOT"
echo "Jupyter runtime directory: $(pwd)/.venv/jupyter_runtime"
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Launch Jupyter notebook with explicit runtime directory to avoid permission issues
export JUPYTER_RUNTIME_DIR="$(pwd)/.venv/jupyter_runtime"

# Start Jupyter notebook server with browser auto-opening
python -m jupyter notebook "$SCRIPT_DIR"