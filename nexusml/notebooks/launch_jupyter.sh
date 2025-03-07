#!/bin/bash

# Script to launch Jupyter notebooks for NexusML
# This script should be run from Git Bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Create a dedicated runtime directory for Jupyter to avoid permission issues
mkdir -p "$PROJECT_ROOT/.venv/jupyter_runtime"

# Activate the virtual environment
source "$PROJECT_ROOT/.venv/Scripts/activate"

# Ensure Jupyter is installed
pip install jupyter

# Launch Jupyter notebook with explicit runtime directory to avoid permission issues
export JUPYTER_RUNTIME_DIR="$(pwd)/.venv/jupyter_runtime"
python -m jupyter notebook "$SCRIPT_DIR"