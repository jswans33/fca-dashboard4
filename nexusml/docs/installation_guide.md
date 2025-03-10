# NexusML Installation Guide

This guide provides comprehensive instructions for installing and configuring NexusML, a Python machine learning package for equipment classification.

## System Requirements

### Python Version
- Python 3.8 or higher is required
- Python 3.9 or 3.10 is recommended for optimal performance

### Operating System Compatibility
- Linux (Ubuntu 20.04+, CentOS 7+, etc.)
- macOS (10.15+)
- Windows 10/11

### Hardware Recommendations
- CPU: 2+ cores recommended for training
- RAM: Minimum 4GB, 8GB+ recommended for larger datasets
- Disk Space: 500MB for the package and dependencies, plus additional space for models and data

## Dependencies

### Core Dependencies
NexusML requires the following Python packages:

```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
imbalanced-learn>=0.8.0
pyyaml>=6.0
setuptools>=57.0.0
wheel>=0.36.0
python-dotenv>=0.19.0
tqdm>=4.62.0
```

### Optional Dependencies

#### AI Features
For AI-enhanced features:
```
anthropic>=0.5.0
```

#### Development Dependencies
For development and testing:
```
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
isort>=5.10.0
flake8>=4.0.0
mypy>=0.9.0
```

## Installation Methods

### Standard Installation

The simplest way to install NexusML is using pip:

```bash
pip install nexusml
```

This will install the core package with all required dependencies.

### Installation with AI Features

To install NexusML with AI-enhanced features:

```bash
pip install "nexusml[ai]"
```

### Development Installation

For development or contributing to NexusML:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/nexusml.git
   cd nexusml
   ```

2. Install in development mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Virtual Environment Installation (Recommended)

It's recommended to install NexusML in a virtual environment to avoid dependency conflicts:

#### Using venv (Python's built-in virtual environment)

```bash
# Create a virtual environment
python -m venv nexusml-env

# Activate the virtual environment
# On Windows:
nexusml-env\Scripts\activate
# On macOS/Linux:
source nexusml-env/bin/activate

# Install NexusML
pip install nexusml
```

#### Using conda

```bash
# Create a conda environment
conda create -n nexusml-env python=3.9

# Activate the environment
conda activate nexusml-env

# Install NexusML
pip install nexusml
```

## Installation Verification

After installation, verify that NexusML is correctly installed:

```bash
# Start Python
python

# Import and check version
>>> import nexusml
>>> print(nexusml.__version__)
0.1.0  # Your version may differ
```

You can also run a simple example to verify functionality:

```bash
# Run the simple example
python -m nexusml.examples.simple_example
```

If the example runs without errors and produces output, the installation is working correctly.

## Configuration After Installation

### Configuration Files

NexusML uses YAML configuration files to control its behavior. The system looks for configuration files in the following locations (in order of precedence):

1. Custom path specified in `ConfigProvider.initialize()`
2. Environment variable `NEXUSML_CONFIG_PATH`
3. `nexusml_config.yml` in the current working directory
4. `config/nexusml_config.yml` in the package directory

### Creating a Basic Configuration File

Create a file named `nexusml_config.yml` in your project directory:

```yaml
feature_engineering:
  text_columns:
    - description
    - name
  numerical_columns:
    - service_life
    - cost
  categorical_columns:
    - category
    - type
  transformers:
    text:
      type: tfidf
      max_features: 1000
    numerical:
      type: standard_scaler
    categorical:
      type: one_hot_encoder

model_building:
  model_type: random_forest
  hyperparameters:
    n_estimators: 100
    max_depth: 10
  optimization:
    method: grid_search
    cv: 5
    scoring: f1_macro

data_loading:
  encoding: utf-8
  delimiter: ","
  required_columns:
    - description
    - service_life
    - category

paths:
  output_dir: outputs
  models_dir: outputs/models
  data_dir: data
```

### Environment Variable Configuration

You can override configuration settings using environment variables:

```bash
# Override feature engineering settings
export NEXUSML_FEATURE_ENGINEERING_TEXT_COLUMNS=description,name,manufacturer
export NEXUSML_MODEL_BUILDING_HYPERPARAMETERS_N_ESTIMATORS=200
export NEXUSML_PATHS_OUTPUT_DIR=/custom/output/path
```

Environment variables use the format:
- Prefix: `NEXUSML_`
- Section and keys: Uppercase with underscores
- Nested keys: Separated by underscores
- Lists: Comma-separated values

## Troubleshooting

### Common Installation Issues

#### Package Not Found

If you encounter a "Package not found" error:

```
ERROR: Could not find a version that satisfies the requirement nexusml
ERROR: No matching distribution found for nexusml
```

Solutions:
- Ensure you're using Python 3.8 or higher
- Update pip: `pip install --upgrade pip`
- Check your internet connection
- If installing from a private repository, ensure you have proper authentication

#### Dependency Conflicts

If you encounter dependency conflicts:

```
ERROR: Cannot install nexusml due to conflicting dependencies
```

Solutions:
- Use a virtual environment for a clean installation
- Try installing with the `--ignore-installed` flag: `pip install --ignore-installed nexusml`
- Check for conflicting packages in your environment

#### Import Errors After Installation

If you can install but encounter import errors:

```python
>>> import nexusml
ImportError: No module named 'nexusml'
```

Solutions:
- Ensure you're using the same Python environment where you installed the package
- Check if the package is installed: `pip list | grep nexusml`
- Try reinstalling: `pip uninstall nexusml && pip install nexusml`

### Platform-Specific Issues

#### Windows

- If you encounter path-related errors, ensure you're using forward slashes (`/`) or double backslashes (`\\`) in configuration paths
- For permission issues, try running the command prompt as administrator

#### macOS

- If you encounter SSL certificate errors, you may need to install certificates: `pip install --upgrade certifi`
- For permission issues: `pip install --user nexusml`

#### Linux

- For permission issues: `pip install --user nexusml` or use a virtual environment
- If dependencies fail to build, you may need to install development tools: `sudo apt-get install python3-dev build-essential` (Ubuntu/Debian) or `sudo yum install python3-devel gcc` (CentOS/RHEL)

## Upgrading

### Standard Upgrade

To upgrade NexusML to the latest version:

```bash
pip install --upgrade nexusml
```

### Upgrade with AI Features

To upgrade NexusML with AI features:

```bash
pip install --upgrade "nexusml[ai]"
```

### Development Version Upgrade

If you installed from the repository:

```bash
cd nexusml
git pull
pip install -e ".[dev]"
```

### Version Compatibility

When upgrading, be aware of potential breaking changes between versions:

- Check the CHANGELOG.md file in the repository for breaking changes
- Test your existing code with the new version before deploying to production
- Consider pinning the version in your requirements file if needed: `nexusml==0.1.0`

## Next Steps

After installation, you can:

1. Explore the [examples directory](../examples/) for usage examples
2. Read the [Usage Guide](usage_guide.md) for comprehensive usage documentation
3. Check the [API Reference](api_reference.md) for detailed API information
4. Review the [Architecture Overview](architecture/overview.md) for system design information