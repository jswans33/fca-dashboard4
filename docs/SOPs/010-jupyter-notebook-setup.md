# SOP 010: Setting Up and Using Jupyter Notebooks with NexusML

## 1. Purpose

This SOP provides a comprehensive guide for setting up and using Jupyter notebooks with the NexusML project, ensuring proper package imports and environment configuration.

## 2. Prerequisites

- Python 3.8+
- Git Bash (for Windows users)
- NexusML project cloned to your local machine

### 2.1 Required Dependencies

The following dependencies are required for using Jupyter notebooks with NexusML:

- jupyter
- python-dotenv
- tqdm

### 2.2 Optional Dependencies

The following dependencies are optional but may be required for specific functionality:

- anthropic - Required for OmniClass description generation with Claude API

## 3. Installing Dependencies

### 3.1 Using the Launch Script (Recommended)

The `launch_jupyter.sh` script will automatically install the required dependencies using UV (a faster alternative to pip) and prompt you to install optional dependencies:

```bash
cd nexusml/notebooks
bash launch_jupyter.sh
```

### 3.2 Manual Installation

If you prefer to install the dependencies manually:

#### 3.2.1 Using UV (Recommended)

UV is a modern, fast Python package installer and resolver that's an alternative to pip.

```bash
# Activate the virtual environment
source .venv/Scripts/activate  # Windows with Git Bash
# OR
source .venv/bin/activate      # Linux/macOS

# Install UV if not already installed
pip install uv

# Install required dependencies
uv pip install jupyter python-dotenv tqdm

# Install optional dependencies if needed
uv pip install anthropic  # For OmniClass description generation
```

#### 3.2.2 Using pip

```bash
# Activate the virtual environment
source .venv/Scripts/activate  # Windows with Git Bash
# OR
source .venv/bin/activate      # Linux/macOS

# Install required dependencies
pip install jupyter python-dotenv tqdm

# Install optional dependencies if needed
pip install anthropic  # For OmniClass description generation
```

## 4. Starting Jupyter Notebook Server

### 4.1 Using the Makefile (Recommended)

The easiest way to start the Jupyter notebook server is using the Makefile target:

```bash
# From the project root directory
make nexusml-notebook
```

This will:
1. Navigate to the notebooks directory
2. Run the launch script that sets up the environment
3. Start the Jupyter notebook server
4. Automatically open the Jupyter interface in your default web browser

### 4.2 Using the Launch Script Directly

Alternatively, you can run the launch script directly:

```bash
# Navigate to the notebooks directory
cd nexusml/notebooks

# Run the launch script
bash launch_jupyter.sh
```

This script will:
1. Activate the virtual environment
2. Install UV if needed
3. Install Jupyter and required dependencies using UV
4. Install the nexusml package in development mode if not already installed
5. Start the Jupyter notebook server in the notebooks directory
6. Automatically open the Jupyter interface in your default web browser

### 4.3 Manual Setup

If you prefer to set up manually:

```bash
# Activate the virtual environment
source .venv/Scripts/activate  # Windows with Git Bash
# OR
source .venv/bin/activate      # Linux/macOS

# Install Jupyter
pip install jupyter

# Install nexusml in development mode
cd nexusml
pip install -e .
cd ..

# Start Jupyter notebook
jupyter notebook
```

## 5. Initializing the Notebook Environment

To ensure that the nexusml package can be imported correctly in your notebooks, use one of the following methods:

### 5.1 Using the Initialization Script (Recommended)

At the beginning of your notebook, add the following cell and run it:

```python
%run init_notebook.py
```

This will:
1. Add the necessary directories to the Python path
2. Verify that nexusml can be imported
3. Return useful paths for your notebook

### 5.2 Using the Setup Script (Alternative)

Alternatively, you can use the setup_notebook.py script which provides the `setup_notebook_environment()` function:

```python
%run setup_notebook.py

# Then you can use the setup_notebook_environment function:
paths = setup_notebook_environment()
print("Project paths:")
for name, path in paths.items():
    print(f"  {name}: {path}")
```

This will:
1. Run the initialization script
2. Set up additional environment variables if needed
3. Return a dictionary of useful paths for your notebook

### 5.3 Manual Initialization

Alternatively, you can manually add the following code to your notebook:

```python
import os
import sys
from pathlib import Path

# Get the directory of this notebook
notebook_dir = Path(os.getcwd())

# Go up to the project root (parent of nexusml)
project_root = notebook_dir.parent.parent

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

# Try to import nexusml to verify it works
try:
    import nexusml
    print(f"Successfully imported nexusml from {nexusml.__file__}")
except ImportError as e:
    print(f"Failed to import nexusml: {e}")
```

## 6. Using the Path Utilities

NexusML includes path utilities to help with file and directory management:

```python
from nexusml.utils.path_utils import (
    get_project_root,
    get_nexusml_root,
    find_data_files,
    resolve_path
)

# Get project root directory
project_root = get_project_root()
print(f"Project root: {project_root}")

# Find data files
data_files = find_data_files()
print("Available data files:")
for name, path in data_files.items():
    print(f"  {name}: {path}")

# Resolve a path relative to the project root
config_path = resolve_path("config/settings.yml", project_root)
print(f"Config path: {config_path}")
```

## 7. Working with Multiple Virtual Environments

If you have multiple virtual environments, ensure you're using the correct one:

1. Check which virtual environment is active:
   ```python
   import sys
   print(sys.executable)  # Should point to your project's virtual environment
   ```

2. If you need to switch environments, restart the kernel and select the correct environment:
   - In Jupyter Notebook: Kernel → Change kernel → Select your environment
   - In JupyterLab: Kernel → Change Kernel → Select your environment

## 8. Troubleshooting

### 8.1 Module Not Found Errors

If you encounter "ModuleNotFoundError: No module named 'nexusml'":

1. Run the initialization script:
   ```python
   %run init_notebook.py
   ```

2. Verify the Python path:
   ```python
   import sys
   print(sys.path)  # Check if the project root is in the path
   ```

3. Install nexusml in development mode:
   ```bash
   cd nexusml
   pip install -e .
   ```

### 8.2 Missing Dependencies

If you encounter errors about missing dependencies like "No module named 'dotenv'" or "No module named 'anthropic'":

1. Install the required dependencies:
   ```bash
   pip install python-dotenv tqdm
   ```

2. Install optional dependencies if needed:
   ```bash
   pip install anthropic
   ```

### 8.3 Kernel Dying

If the kernel keeps dying:

1. Check memory usage - you might be running out of memory
2. Try restarting the kernel with a clean state
3. Verify that all dependencies are installed correctly

### 8.4 Path Issues

If you're having issues with file paths:

1. Use the path utilities provided by nexusml:
   ```python
   from nexusml.utils.path_utils import resolve_path, get_project_root
   ```

2. Print the current working directory:
   ```python
   import os
   print(os.getcwd())
   ```

## 9. Best Practices

1. **Use relative imports**: When importing from nexusml, use relative imports to avoid path issues
2. **Keep notebooks modular**: Break down complex tasks into smaller, reusable functions
3. **Document your code**: Add comments and markdown cells to explain your workflow
4. **Save intermediate results**: For long-running processes, save intermediate results to disk
5. **Use version control**: Commit your notebooks to version control, but clear outputs before committing

## 10. Example Notebook Structure

### 10.1 Using init_notebook.py

```python
# Initialize the environment
%run init_notebook.py

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import from nexusml
from nexusml.core.model import EquipmentClassifier
from nexusml.utils.path_utils import find_data_files

# Load data
data_files = find_data_files()
data_path = data_files.get("sample_data.xlsx")
data = pd.read_excel(data_path)

# Process data
# ...

# Train model
# ...

# Evaluate results
# ...
```

### 10.2 Using setup_notebook.py

```python
# Set up the notebook environment
%run setup_notebook.py

# Get paths from the setup function
paths = setup_notebook_environment()
print("Project paths:")
for name, path in paths.items():
    print(f"  {name}: {path}")

# Import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import from nexusml
from nexusml.core.model import EquipmentClassifier
from nexusml.utils.path_utils import find_data_files

# Load data
data_files = find_data_files()
data_path = data_files.get("sample_data.xlsx")
data = pd.read_excel(data_path)

# Process data
# ...

# Train model
# ...

# Evaluate results
# ...
```

## 11. References

- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [SOP 009: Test-Driving the Equipment Classification Model in Jupyter Notebook](./009-model-testing-jupyter.md)