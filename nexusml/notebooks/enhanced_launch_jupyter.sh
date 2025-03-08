#!/bin/bash

# Enhanced Script to launch Jupyter notebooks for NexusML
# This script provides improved error handling, clearer output, and better dependency management
# It should be run from Git Bash on Windows or a standard terminal on Linux/macOS

# Set up colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print a formatted header
print_header() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

# Print a success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print an error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print a warning message
print_warning() {
    echo -e "${YELLOW}! $1${NC}"
}

# Print a step message
print_step() {
    echo -e "${BLUE}→ $1${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
NEXUSML_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_header "NexusML Jupyter Notebook Launcher"
echo "This script will set up and launch Jupyter notebooks for NexusML"

# Display environment information
echo -e "\n${BOLD}Environment Information:${NC}"
echo "• Script directory: $SCRIPT_DIR"
echo "• Project root: $PROJECT_ROOT"
echo "• NexusML root: $NEXUSML_ROOT"

# Create a dedicated runtime directory for Jupyter to avoid permission issues
print_step "Creating Jupyter runtime directory..."
mkdir -p "$PROJECT_ROOT/.venv/jupyter_runtime"
JUPYTER_RUNTIME_DIR="$PROJECT_ROOT/.venv/jupyter_runtime"
export JUPYTER_RUNTIME_DIR

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    print_warning "No active virtual environment detected"
    
    # Check if .venv exists
    if [[ -d "$PROJECT_ROOT/.venv" ]]; then
        print_step "Activating existing virtual environment..."
        
        # Determine the activate script based on platform
        if [[ -f "$PROJECT_ROOT/.venv/Scripts/activate" ]]; then
            source "$PROJECT_ROOT/.venv/Scripts/activate"
        elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            source "$PROJECT_ROOT/.venv/bin/activate"
        else
            print_error "Could not find activation script for virtual environment"
            print_step "Creating a new virtual environment..."
            python -m venv "$PROJECT_ROOT/.venv"
            
            if [[ -f "$PROJECT_ROOT/.venv/Scripts/activate" ]]; then
                source "$PROJECT_ROOT/.venv/Scripts/activate"
            elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
                source "$PROJECT_ROOT/.venv/bin/activate"
            else
                print_error "Failed to create and activate virtual environment"
                exit 1
            fi
        fi
    else
        print_step "Creating a new virtual environment..."
        python -m venv "$PROJECT_ROOT/.venv"
        
        if [[ -f "$PROJECT_ROOT/.venv/Scripts/activate" ]]; then
            source "$PROJECT_ROOT/.venv/Scripts/activate"
        elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            source "$PROJECT_ROOT/.venv/bin/activate"
        else
            print_error "Failed to create and activate virtual environment"
            exit 1
        fi
    fi
    
    if [[ -n "$VIRTUAL_ENV" ]]; then
        print_success "Virtual environment activated: $VIRTUAL_ENV"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
else
    print_success "Using active virtual environment: $VIRTUAL_ENV"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1)
print_success "Using Python: $PYTHON_VERSION"

# Install package managers
print_header "Setting Up Package Managers"

# Check for pip and upgrade if needed
if command_exists pip; then
    print_step "Upgrading pip..."
    python -m pip install --upgrade pip
    print_success "pip upgraded successfully"
else
    print_error "pip not found. This is unusual and may indicate a problem with your Python installation."
    exit 1
fi

# Install UV if not already installed
if ! command_exists uv; then
    print_step "Installing UV package manager..."
    pip install uv
    
    if command_exists uv; then
        print_success "UV installed successfully"
    else
        print_warning "Failed to install UV. Falling back to pip for package installation."
    fi
else
    print_success "UV already installed"
fi

# Function to install packages with fallback
install_packages() {
    local packages=("$@")
    
    if command_exists uv; then
        print_step "Installing packages using UV: ${packages[*]}"
        uv pip install "${packages[@]}"
    else
        print_step "Installing packages using pip: ${packages[*]}"
        pip install "${packages[@]}"
    fi
}

# Install required dependencies
print_header "Installing Required Dependencies"
install_packages jupyter notebook ipykernel python-dotenv tqdm

# Install optional dependencies if needed
print_header "Optional Dependencies"
echo "Some features require additional packages."

read -p "Install matplotlib and seaborn for visualizations? (y/n) " install_viz
if [[ $install_viz == "y" || $install_viz == "Y" ]]; then
    install_packages matplotlib seaborn
    print_success "Visualization packages installed"
else
    print_warning "Visualization packages not installed. Some notebook features may not work."
fi

read -p "Install anthropic package for Claude API integration? (y/n) " install_anthropic
if [[ $install_anthropic == "y" || $install_anthropic == "Y" ]]; then
    install_packages anthropic
    print_success "Anthropic package installed"
else
    print_warning "Anthropic package not installed. OmniClass description generation will not be available."
fi

# Install nexusml in development mode if not already installed
print_header "NexusML Package Setup"
if ! python -c "import nexusml" &> /dev/null; then
    print_step "Installing nexusml package in development mode..."
    cd "$NEXUSML_ROOT" && pip install -e .
    cd "$SCRIPT_DIR"  # Return to the original directory
    
    if python -c "import nexusml" &> /dev/null; then
        print_success "nexusml package installed successfully"
    else
        print_error "Failed to install nexusml package"
        exit 1
    fi
else
    print_success "nexusml package already installed"
    NEXUSML_PATH=$(python -c "import nexusml; print(nexusml.__file__)")
    echo "• Installed at: $NEXUSML_PATH"
fi

# Check for notebook initialization files
print_step "Checking for notebook initialization files..."
if [[ -f "$SCRIPT_DIR/init_notebook.py" ]]; then
    print_success "Found init_notebook.py"
else
    print_warning "init_notebook.py not found. Creating it..."
    cat > "$SCRIPT_DIR/init_notebook.py" << 'EOF'
"""
Notebook Initialization Script for NexusML

This script initializes the environment for Jupyter notebooks,
ensuring that the nexusml package can be imported correctly.

Usage:
    # At the beginning of your notebook, add:
    %run init_notebook.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional


def init_notebook_environment() -> Dict[str, str]:
    """
    Initialize the notebook environment by adding the necessary paths
    to the Python path to ensure nexusml can be imported.
    
    Returns:
        Dictionary of useful paths for notebooks
    """
    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    
    # Go up to the project root (parent of nexusml)
    project_root = script_dir.parent.parent
    
    # Add project root to Python path if not already there
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to Python path")
    
    # Add nexusml parent directory to Python path if not already there
    nexusml_parent = project_root
    if str(nexusml_parent) not in sys.path:
        sys.path.insert(0, str(nexusml_parent))
        print(f"Added {nexusml_parent} to Python path")
    
    # Print confirmation
    print(f"Notebook environment initialized. You can now import nexusml.")
    print(f"Project root: {project_root}")
    
    # Create and return useful paths
    paths: Dict[str, str] = {
        "project_root": str(project_root),
        "nexusml_root": str(script_dir.parent),
        "notebooks_dir": str(script_dir),
        "data_dir": str(project_root / "data"),
        "examples_dir": str(project_root / "examples"),
        "outputs_dir": str(project_root / "outputs"),
    }
    return paths


# When run directly (via %run), initialize the environment
paths = init_notebook_environment()

# Try to import nexusml to verify it works
try:
    import nexusml
    print(f"Successfully imported nexusml from {nexusml.__file__}")
except ImportError as e:
    print(f"Warning: Failed to import nexusml: {e}")
    print("You may need to install the nexusml package or check your Python path.")
EOF
    print_success "Created init_notebook.py"
fi

if [[ -f "$SCRIPT_DIR/setup_notebook.py" ]]; then
    print_success "Found setup_notebook.py"
else
    print_warning "setup_notebook.py not found. Creating it..."
    cat > "$SCRIPT_DIR/setup_notebook.py" << 'EOF'
"""
Notebook Setup Script for NexusML

This script sets up the environment for Jupyter notebooks,
ensuring that the nexusml package can be imported correctly
and providing utility functions for working with paths.

Usage:
    # At the beginning of your notebook, add:
    %run setup_notebook.py
    
    # Then you can use the setup_notebook_environment function:
    paths = setup_notebook_environment()
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Import the init_notebook_environment function from init_notebook.py
from init_notebook import init_notebook_environment

def setup_notebook_environment() -> Dict[str, str]:
    """
    Set up the notebook environment by adding the necessary paths
    to the Python path to ensure nexusml can be imported.
    
    Returns:
        Dictionary of useful paths for notebooks
    """
    # Call the init_notebook_environment function
    paths = init_notebook_environment()
    
    # Add additional setup if needed
    print("Notebook environment setup complete.")
    
    return dict(paths)  # Convert to dict to ensure correct return type

# When run directly (via %run), set up the environment
if __name__ == "__main__":
    paths = setup_notebook_environment()
    
    # Print the paths
    print("\nAvailable paths:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
EOF
    print_success "Created setup_notebook.py"
fi

# Print environment information
print_header "Environment Information"
echo "• Python executable: $(which python)"
echo "• Python version: $(python --version)"
echo "• Jupyter runtime directory: $JUPYTER_RUNTIME_DIR"
echo "• Working directory: $(pwd)"

# Launch Jupyter notebook
print_header "Starting Jupyter Notebook Server"
echo "Once the server starts, copy and paste the URL with token into your browser."
echo "The URL will look like: http://localhost:8888/tree?token=<token>"
echo ""
echo "Available notebooks:"
echo "• enhanced_modular_template.ipynb - Enhanced template with improved error handling and features"
echo "• modular_template.ipynb - Original modular template"
echo ""
echo "Press Ctrl+C twice to stop the Jupyter server when you're done."
echo ""

# Start Jupyter notebook server
python -m jupyter notebook "$SCRIPT_DIR" --no-browser