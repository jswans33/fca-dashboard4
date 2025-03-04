# SOP-001: ETL Pipeline Environment Setup

## Purpose

This procedure documents the exact steps required to set up the development environment for the ETL Pipeline v4 project, ensuring consistency across all development environments.

## Scope

This SOP covers:

- Virtual environment setup and activation
- Dependencies installation
- Linter configuration
- Project structure creation
- Configuration files setup
- VS Code workspace configuration

## Prerequisites

- Git installed
- Python 3.9+ installed
- Access to the project repository
- VS Code installed (for workspace setup)

## Procedure

### 1. Virtual Environment Setup

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:

```bash
    source ./.venv/scripts/activate
```

### 2. Dependencies Setup

1. Create a `requirements.txt` file with the following content:

   ```text
   # Core dependencies
   sqlalchemy>=2.0.0,<3.0.0
   alembic>=1.9.0,<2.0.0
   pandas>=1.5.0,<2.0.0
   openpyxl>=3.1.0,<4.0.0  # For Excel support in pandas
   pyyaml>=6.0,<7.0
   psycopg2-binary>=2.9.5,<3.0.0  # PostgreSQL driver
   
   # Development dependencies
   pytest>=7.0.0,<8.0.0
   black>=23.0.0,<24.0.0
   isort>=5.12.0,<6.0.0
   flake8>=6.0.0,<7.0.0
   mypy>=1.0.0,<2.0.0
   ruff>=0.0.262,<1.0.0  # Fast Python linter
   
   # Type checking
   types-PyYAML>=6.0.0,<7.0.0
   types-requests>=2.29.0,<3.0.0
   types-setuptools>=65.0.0,<66.0.0
   types-toml>=0.10.0,<0.11.0
   
   # Logging
   loguru>=0.6.0,<0.7.0
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:

   ```bash
   pip install -e .
   ```

   This step is crucial for allowing the package to be imported using absolute imports (e.g., `from fca_dashboard.config.settings import get_settings`). Without this step, you may encounter `ModuleNotFoundError` when trying to import modules from the package.

   Note: The Makefile's `install` target includes this step, so you can also run `make install` to install all dependencies and the package in development mode.

### 3. Linter Configuration

1. Create `pyproject.toml` with build system, project metadata, and linting configurations:

   ```toml
   [build-system]
   requires = ["setuptools>=42", "wheel"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "fca-dashboard"
   version = "0.1.0"
   description = "ETL Pipeline for FCA Dashboard"
   readme = "README.md"
   requires-python = ">=3.9"
   license = {text = "Proprietary"}
   authors = [
       {name = "ETL Team"}
   ]

   [tool.black]
   line-length = 120
   target-version = ['py39']
   include = '\.pyi?$'
   
   [tool.isort]
   profile = "black"
   line_length = 120
   multi_line_output = 3
   
   [tool.flake8]
   max-line-length = 120
   extend-ignore = "E203"
   exclude = [".git", "__pycache__", "build", "dist", "alembic"]
   
   [tool.mypy]
   python_version = "3.9"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = true
   disallow_incomplete_defs = true
   
   [tool.pytest.ini_options]
   testpaths = ["fca_dashboard/tests"]
   python_files = "test_*.py"
   python_classes = "Test*"
   python_functions = "test_*"
   addopts = "--cov=fca_dashboard --cov-report=html --cov-report=term"

   [tool.coverage.run]
   source = ["fca_dashboard"]
   omit = [
       "*/tests/*",
       "*/alembic/*",
       "*/__pycache__/*",
       "*/migrations/*",
       "*/venv/*",
       "*/.venv/*",
   ]

   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "raise NotImplementedError",
       "if __name__ == .__main__.:",
       "pass",
       "raise ImportError",
   ]
   
   [tool.ruff]
   # Enable Pyflakes ('F'), pycodestyle ('E'), and isort ('I') codes by default.
   select = ["E", "F", "I", "N", "B", "C4", "SIM", "ERA"]
   ignore = []
   
   # Allow autofix for all enabled rules (when `--fix`) is provided.
   fixable = ["ALL"]
   unfixable = []
   
   # Exclude a variety of commonly ignored directories.
   exclude = [
       ".git",
       ".mypy_cache",
       ".ruff_cache",
       ".venv",
       "__pycache__",
       "build",
       "dist",
   ]
   
   # Same as Black.
   line-length = 120
   
   # Allow unused variables when underscore-prefixed.
   dummy-variable-rgx = "^(_+|(_+[a-zA-Z0-9_]*[a-zA-Z0-9]+?))$"
   
   # Assume Python 3.9
   target-version = "py39"
   
   [tool.ruff.mccabe]
   # Unlike Flake8, default to a complexity level of 10.
   max-complexity = 10
   
   [tool.ruff.isort]
   known-first-party = ["fca_dashboard"]
   ```

2. Create a `pytest.ini` file for additional pytest configuration:

   ```ini
   [pytest]
   testpaths = fca_dashboard/tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*

   [mypy]
   python_version = 3.8
   warn_return_any = True
   warn_unused_configs = True
   disallow_untyped_defs = True
   disallow_incomplete_defs = True

   # Ignore missing imports for third-party libraries
   [mypy.plugins.pytest]
   # This tells mypy that pytest fixtures can return Any
   pytest_fixture_function = True

   # Ignore errors in test files
   [mypy-fca_dashboard.tests.*]
   disallow_untyped_defs = False
   disallow_incomplete_defs = False
   ```

2. Create a minimal `setup.py` file to make the package installable:

   ```python
   """
   Setup script for the FCA Dashboard package.
   """

   from setuptools import setup, find_packages

   setup(
       name="fca_dashboard",
       packages=find_packages(),
   )
   ```

2. Test linters:

   ```bash
   black --check .
   isort --check .
   flake8 .
   mypy .
   ```

3.1. nstalling Make on Windows using PowerShell
Here's the step-by-step PowerShell script to install Make using Chocolatey:

First, check if Chocolatey is installed, if not install it:
Then install Make:
Verify the installation:
Alternative: Using Git Bash
If you have Git for Windows installed, you can also use Git Bash which comes with Make pre-installed:

Open Git Bash
Navigate to your project directory
Run Make commands directly in Git Bash
Note: If you're using VS Code, you can set Git Bash as your integrated terminal:

Similar code found with 2 license types - View matches
   
3.2 Create a `Makefile` for convenience commands:

   ```makefile
   .PHONY: lint test format install run clean init-db coverage test-unit test-integration

   lint:
    black --check .
    isort --check .
    flake8 .
    mypy .

   format:
    black .
    isort .

   test:
    pytest fca_dashboard/tests/

   coverage:
    pytest --cov=fca_dashboard --cov-report=html --cov-report=term fca_dashboard/tests/
    @echo "HTML coverage report generated in htmlcov/"

   test-unit:
    pytest fca_dashboard/tests/unit/

   test-integration:
    pytest fca_dashboard/tests/integration/

   install:
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .
   ```

   Note: The `install` target includes installing the package in development mode (`pip install -e .`), which is necessary for absolute imports to work correctly. The additional test targets provide more granular control over test execution:

   - `test`: Runs all tests
   - `test-unit`: Runs only unit tests
   - `test-integration`: Runs only integration tests
   - `coverage`: Runs tests with coverage reporting, generating both terminal and HTML reports

### 3.4. Understanding and Using Makefiles

1. What is a Makefile?
   - A Makefile is a configuration file used by the `make` utility to automate tasks and build processes
   - It contains a set of directives (targets) that define commands to be executed
   - Makefiles help standardize common development tasks across the team

2. Prerequisites for using Makefiles:
   - Windows: Install Make via Chocolatey (`choco install make`) or use Git Bash
   - macOS: Make is typically pre-installed
   - Linux: Install via package manager (e.g., `apt install make`)

3. Makefile structure explanation:
   - `.PHONY`: Declares targets that don't represent files (prevents conflicts with files of the same name)
   - Each target (e.g., `lint:`) defines a task that can be executed with `make <target>`
   - Commands under targets must be indented with tabs, not spaces

4. Additional useful targets to consider:

   ```makefile
   # Add these to your Makefile as needed
   
   # Run the ETL pipeline with default settings
   run:
    python fca_dashboard/main.py --config fca_dashboard/config/settings.yaml
   
   # Clean up generated files
   clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type f -name "*.pyd" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    find . -type d -name "*.egg" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".coverage" -exec rm -rf {} +
    find . -type d -name "htmlcov" -exec rm -rf {} +
    find . -type d -name ".mypy_cache" -exec rm -rf {} +
   
   # Create initial database schema
   init-db:
    python -c "from fca_dashboard.core.models import Base; from sqlalchemy import create_engine; engine = create_engine('sqlite:///etl.db'); Base.metadata.create_all(engine)"
   ```

5. Using the Makefile:
   - Run linters: `make lint`
   - Format code: `make format`
   - Run tests: `make test`
   - Install dependencies: `make install`
   - Execute multiple targets: `make clean install test`

### Installing Make on Windows

1. Using PowerShell (Administrator):
```powershell
# Install Chocolatey if not already installed
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install Make using Chocolatey
choco install make -y
```

2. Alternative: Using Git Bash
- Git Bash comes with Make pre-installed
- Open Git Bash
- Verify installation: `make --version`

3. VS Code Integration
```json
// In .vscode/settings.json
{
    "terminal.integrated.defaultProfile.windows": "Git Bash"
}
```

This will allow you to use Make commands directly in the VS Code integrated terminal.

### 4. Project Structure Setup

1. Create the directory structure:

   ```bash
   mkdir -p fca_dashboard/{alembic,config,core/{interfaces,models},extractors/{strategies},mappers/{strategies},loaders,pipelines,utils,tests/{unit,integration}}
   touch fca_dashboard/__init__.py
   touch fca_dashboard/main.py
   touch fca_dashboard/core/__init__.py
   touch fca_dashboard/extractors/__init__.py
   touch fca_dashboard/mappers/__init__.py
   touch fca_dashboard/loaders/__init__.py
   touch fca_dashboard/pipelines/__init__.py
   touch fca_dashboard/utils/__init__.py
   touch fca_dashboard/tests/__init__.py
   touch fca_dashboard/tests/unit/__init__.py
   touch fca_dashboard/tests/integration/__init__.py
   ```

   Note: We're using `fca_dashboard` as the package name to match the repository name (`fca-dashboard4`), rather than `fca_dashboard` as mentioned in the guide.

### 5. Configuration Files Setup

1. Create `fca_dashboard/config/settings.yaml`:

   ```yaml
   # Database settings
   databases:
     sqlite:
       url: "sqlite:///fca_dashboard.db"
     postgresql:
       url: "postgresql://user:password@localhost/fca_dashboard"
       
   # Pipeline settings
   pipeline_settings:
     batch_size: 5000
     log_level: "INFO"
     
   # Table mappings
   tables:
     equipment:
       mapping_type: "direct"
       column_mappings:
         tag: "Tag"
         name: "Name"
         description: "Description"
   ```

2. Create `fca_dashboard/config/settings.py` for loading settings.

### 7. Configure Logging Early

1. Set up logging configuration right after environment setup:

   ```bash
   # After installing dependencies
   mkdir -p logs
   touch logs/fca_dashboard.log
   ```

2. Create a type stub file for Loguru to help with type checking:

   ```python
   # fca_dashboard/utils/loguru_stubs.pyi
   from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union, overload

   class Logger:
       def remove(self, handler_id: Optional[int] = None) -> None: ...
       
       def add(
           self,
           sink: Union[TextIO, str, Callable, Dict[str, Any]],
           *,
           level: Optional[Union[str, int]] = None,
           format: Optional[str] = None,
           filter: Optional[Union[str, Callable, Dict[str, Any]]] = None,
           colorize: Optional[bool] = None,
           serialize: Optional[bool] = None,
           backtrace: Optional[bool] = None,
           diagnose: Optional[bool] = None,
           enqueue: Optional[bool] = None,
           catch: Optional[bool] = None,
           rotation: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
           retention: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
           compression: Optional[Union[str, int, Callable, Dict[str, Any]]] = None,
           delay: Optional[bool] = None,
           mode: Optional[str] = None,
           buffering: Optional[int] = None,
           encoding: Optional[str] = None,
           **kwargs: Any
       ) -> int: ...
       
       def bind(self, **kwargs: Any) -> "Logger": ...
       
       def opt(
           self,
           *,
           exception: Optional[Union[bool, Tuple[Any, ...], Dict[str, Any]]] = None,
           record: Optional[bool] = None,
           lazy: Optional[bool] = None,
           colors: Optional[bool] = None,
           raw: Optional[bool] = None,
           capture: Optional[bool] = None,
           depth: Optional[int] = None,
           ansi: Optional[bool] = None,
       ) -> "Logger": ...
       
       def trace(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def debug(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def info(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def success(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def warning(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def error(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def critical(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def exception(self, __message: Any, *args: Any, **kwargs: Any) -> None: ...
       def log(self, level: Union[int, str], __message: Any, *args: Any, **kwargs: Any) -> None: ...
       
       def level(self, name: str, no: int = 0, color: Optional[str] = None, icon: Optional[str] = None) -> "Logger": ...
       def disable(self, name: str) -> None: ...
       def enable(self, name: str) -> None: ...
       
       def configure(
           self,
           *,
           handlers: List[Dict[str, Any]] = [],
           levels: List[Dict[str, Any]] = [],
           extra: Dict[str, Any] = {},
           patcher: Optional[Callable] = None,
           activation: List[Tuple[str, bool]] = [],
       ) -> None: ...
       
       def patch(self, patcher: Callable) -> "Logger": ...
       
       def complete(self) -> None: ...
       
       @property
       def catch(self) -> Callable: ...

   logger: Logger
   ```

3. Create a logging utility using Loguru:

   ```python
   # fca_dashboard/utils/logging_config.py
   """
   Logging configuration module for the FCA Dashboard application.

   This module provides functionality to configure logging for the application
   using Loguru, which offers improved formatting, better exception handling,
   and simplified configuration compared to the standard logging module.
   """

   import sys
   from pathlib import Path
   from typing import Optional, Any

   from loguru import logger  # type: ignore


   def configure_logging(
       level: str = "INFO",
       log_file: Optional[str] = None,
       rotation: str = "10 MB",
       retention: str = "1 month",
       format_string: Optional[str] = None
   ) -> None:
       """
       Configure application logging with console and optional file output using Loguru.
       
       Args:
           level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
           log_file: Path to log file. If None, only console logging is configured.
           rotation: When to rotate the log file (e.g., "10 MB", "1 day")
           retention: How long to keep log files (e.g., "1 month", "1 year")
           format_string: Custom format string for log messages
       """
       # Remove default handlers
       logger.remove()
       
       # Default format string if none provided
       if format_string is None:
           format_string = (
               "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
           )
       
       # Add console handler
       logger.add(
           sys.stderr,
           level=level.upper(),
           format=format_string,
           colorize=True
       )
       
       # Add file handler if log_file is provided
       if log_file:
           # Ensure log directory exists
           log_path = Path(log_file)
           log_dir = log_path.parent
           if not log_dir.exists():
               log_dir.mkdir(parents=True, exist_ok=True)
           
           # Add rotating file handler
           logger.add(
               log_file,
               level=level.upper(),
               format=format_string,
               rotation=rotation,
               retention=retention,
               compression="zip"
           )
       
       logger.info(f"Logging configured with level: {level}")


   def get_logger(name: str = "fca_dashboard") -> Any:
       """
       Get a logger instance with the specified name.
       
       Args:
           name: Logger name, typically the module name
           
       Returns:
           Loguru logger instance
       """
       return logger.bind(name=name)
   ```

3. Install type stubs for libraries:

   ```bash
   # Install type stubs for PyYAML (already in requirements.txt)
   pip install types-PyYAML
   ```

   Note: Type stubs are important for static type checking with mypy. They provide type information for libraries that don't have built-in type annotations. If you encounter mypy errors like "Cannot find implementation or library stub for module named X", you may need to install type stubs for that library.

4. Test logging configuration:

   ```bash
   python -c "from fca_dashboard.utils.logging_config import configure_logging; configure_logging('DEBUG', 'logs/fca_dashboard.log')"
   ```

### 8. Unit Testing Setup

1. Create test files for key modules:

   ```bash
   # Create test files for settings and logging modules
   mkdir -p fca_dashboard/tests/unit
   touch fca_dashboard/tests/unit/__init__.py
   touch fca_dashboard/tests/unit/test_settings.py
   touch fca_dashboard/tests/unit/test_logging_config.py
   ```

2. Create a `conftest.py` file in the tests directory to ensure pytest can find modules:

   ```python
   """
   Pytest configuration file.

   This file contains shared fixtures and configuration for pytest.
   """

   import os
   import sys
   from pathlib import Path

   # Add the project root directory to the Python path
   # This ensures that the tests can import modules from the project
   project_root = Path(__file__).parent.parent.parent
   sys.path.insert(0, str(project_root))
   ```

3. Implement unit tests for the Settings module in `fca_dashboard/tests/unit/test_settings.py`:

   ```python
   """
   Unit tests for the Settings module.

   This module contains tests for the Settings class and related functionality
   in the fca_dashboard.config.settings module.
   """

   import os
   import tempfile
   from pathlib import Path

   import pytest
   import yaml

   from fca_dashboard.config.settings import Settings, get_settings


   @pytest.fixture
   def temp_settings_file() -> str:
       """Create a temporary settings file for testing."""
       config_content = """
       database:
         host: localhost
         port: 5432
         user: test_user
         password: secret
       app:
         name: test_app
         debug: true
       """
       with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as temp_file:
           temp_file.write(config_content.encode("utf-8"))
           temp_path = temp_file.name
       
       yield temp_path
       
       # Cleanup
       os.unlink(temp_path)


   def test_settings_load_valid_file(temp_settings_file: str) -> None:
       """Test loading settings from a valid file."""
       settings = Settings(config_path=temp_settings_file)
       assert settings.get("database.host") == "localhost"
       assert settings.get("database.port") == 5432
       assert settings.get("app.name") == "test_app"
       assert settings.get("app.debug") is True


   def test_settings_load_missing_file() -> None:
       """Test that loading a non-existent file raises FileNotFoundError."""
       with pytest.raises(FileNotFoundError):
           Settings(config_path="nonexistent_file.yml")


   def test_settings_get_nonexistent_key(temp_settings_file: str) -> None:
       """Test getting a non-existent key returns the default value."""
       settings = Settings(config_path=temp_settings_file)
       assert settings.get("nonexistent.key") is None
       assert settings.get("nonexistent.key", default="fallback") == "fallback"


   def test_settings_get_nested_keys(temp_settings_file: str) -> None:
       """Test getting nested keys from the configuration."""
       settings = Settings(config_path=temp_settings_file)
       assert settings.get("database.user") == "test_user"
       assert settings.get("database.password") == "secret"


   def test_get_settings_caching(temp_settings_file: str) -> None:
       """Test that get_settings caches instances for the same config path."""
       settings1 = get_settings(temp_settings_file)
       settings2 = get_settings(temp_settings_file)
       
       # Should be the same instance
       assert settings1 is settings2
       
       # Modify the first instance and check that the second reflects the change
       settings1.config["test_key"] = "test_value"
       assert settings2.config["test_key"] == "test_value"


   def test_get_settings_default() -> None:
       """Test that get_settings returns the default instance when no path is provided."""
       settings = get_settings()
       assert isinstance(settings, Settings)
       
       # Should return the same default instance on subsequent calls
       settings2 = get_settings()
       assert settings is settings2
   ```

4. Run the tests using the Makefile:

   ```bash
   # Run all tests
   make test
   
   # Run only unit tests
   make test-unit
   
   # Run tests with coverage reporting
   make coverage
   ```

5. Verify test coverage:
   - Open the HTML coverage report in a browser: `open htmlcov/index.html`
   - Check the terminal output for coverage statistics
   - Aim for at least 80% code coverage for critical modules

### 9. Example Pipeline Usage

1. Example command for running the ETL pipeline:

   ```bash
   python fca_dashboard/main.py --config fca_dashboard/config/settings.yaml --excel-file data/sample.xlsx --table-name equipment --log-level INFO
   ```

2. Using the Makefile for common operations:

   ```bash
   # Run linters
   make lint
   
   # Run tests
   make test
   
   # Run tests with coverage
   make coverage
   
   # Format code
   make format
   ```

### 6. VS Code Workspace Setup

1. Create a `.vscode` directory:

   ```bash
   mkdir -p .vscode
   ```

2. Create `.vscode/settings.json` for linting, autoformatting and Prettier configuration:

   ```json
   {
     "python.linting.enabled": true,
     "python.linting.flake8Enabled": true,
     "python.linting.mypyEnabled": true,
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "editor.rulers": [120],
     "python.testing.pytestEnabled": true,
     "python.testing.unittestEnabled": false,
     "python.testing.nosetestsEnabled": false,
     "python.testing.pytestArgs": [
       "fca_dashboard/tests"
     ],
     "python.pythonPath": "${workspaceFolder}/.venv/bin/python",
     "python.analysis.extraPaths": [
       "${workspaceFolder}"
     ],
     "ruff.enable": true,
     "ruff.organizeImports": true,
     "ruff.fixAll": true,
     "ruff.path": ["${workspaceFolder}/.venv/bin/ruff"],
     "[python]": {
       "editor.defaultFormatter": "charliermarsh.ruff",
       "editor.formatOnSave": true,
       "editor.codeActionsOnSave": {
         "source.fixAll.ruff": true,
         "source.organizeImports.ruff": true
       }
     },
     // Prettier configuration for non-Python files
     "prettier.enable": true,
     "[javascript]": {
       "editor.defaultFormatter": "esbenp.prettier-vscode",
       "editor.formatOnSave": true
     },
     "[json]": {
       "editor.defaultFormatter": "esbenp.prettier-vscode",
       "editor.formatOnSave": true
     },
     "[yaml]": {
       "editor.defaultFormatter": "esbenp.prettier-vscode",
       "editor.formatOnSave": true
     },
     "[markdown]": {
       "editor.defaultFormatter": "esbenp.prettier-vscode",
       "editor.formatOnSave": true
     },
     "prettier.singleQuote": true,
     "prettier.trailingComma": "es5",
     "prettier.printWidth": 100
   }
   ```

   Key settings explained:
   - `"python.linting.enabled": true` - Enables linting for Python files
   - `"python.linting.flake8Enabled": true` - Enables flake8 linter
   - `"python.linting.mypyEnabled": true` - Enables mypy type checking
   - `"python.formatting.provider": "black"` - Sets Black as the formatter
   - `"editor.formatOnSave": true` - Automatically formats code when saving files
   - `"editor.rulers": [120]` - Shows a vertical line at 120 characters
   - `"python.pythonPath": "${workspaceFolder}/.venv/bin/python"` - Points to the virtual environment Python
   - `"ruff.enable": true` - Enables Ruff linter
   - `"ruff.organizeImports": true` - Enables Ruff to organize imports (replaces isort)
   - `"ruff.fixAll": true` - Enables Ruff to fix all auto-fixable issues
   - `"editor.codeActionsOnSave"` - Configures Ruff to run on save for fixing and organizing imports

3. Install VS Code extensions for Python development:
   - Python extension (ms-python.python)
   - Pylance (ms-python.vscode-pylance)
   - Python Docstring Generator (njpwerner.autodocstring)
   - YAML (redhat.vscode-yaml)
   - Ruff (charliermarsh.ruff) - Fast Python linter

   These can be installed via the Extensions view in VS Code or using the command line:

   ```bash
   code --install-extension ms-python.python
   code --install-extension ms-python.vscode-pylance
   code --install-extension njpwerner.autodocstring
   code --install-extension redhat.vscode-yaml
   code --install-extension charliermarsh.ruff
   ```

   Ruff configuration:
   - Ruff is a fast Python linter that replaces multiple tools (flake8, isort, etc.)
   - It's significantly faster than traditional linters
   - It can automatically fix many issues with `--fix` option

4. Create `.vscode/launch.json`:

   ```json
   {
     "version": "0.2.0",
     "configurations": [
       {
         "name": "Python: Current File",
         "type": "python",
         "request": "launch",
         "program": "${file}",
         "console": "integratedTerminal",
         "justMyCode": false
       },
       {
         "name": "Python: ETL Pipeline",
         "type": "python",
         "request": "launch",
         "program": "${workspaceFolder}/fca_dashboard/main.py",
         "args": [
           "--config", "${workspaceFolder}/fca_dashboard/config/settings.yaml"
         ],
         "console": "integratedTerminal",
         "justMyCode": false
       }
     ]
   }
   ```

5. Create a workspace file `fca-dashboard4.code-workspace`:

   ```json
   {
     "folders": [
       {
         "path": "."
       }
     ],
     "settings": {
       "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
     }
   }
   ```

## Verification

1. Verify virtual environment activation:
   - Command prompt should show (.venv) prefix
   - Run `pip -V` to confirm packages are installed from the virtual environment

2. Verify dependencies installation:
   - Run `pip list` to confirm all required packages are installed

3. Verify linter configuration:
   - Run linter tests as described in step 3.2
   - All tests should pass or show expected warnings/errors

4. Verify project structure:
   - Run `find fca_dashboard -type d | sort` to confirm all directories exist
   - Run `find fca_dashboard -type f -name "__init__.py" | sort` to confirm all __init__.py files exist

5. Verify VS Code workspace:
   - Open the workspace file in VS Code
   - Confirm linting and formatting work as expected:
     - Open a Python file and introduce a PEP 8 violation (e.g., extra spaces)
     - Save the file and verify it's automatically formatted
     - Introduce a type error and verify mypy highlights it
     - Introduce a syntax error and verify flake8 highlights it
   - Confirm debugging configurations are available
   - Verify extensions are installed and activated

## Troubleshooting

1. Virtual environment issues:
   - If `python -m venv .venv` fails, ensure Python 3.9+ is installed
   - If activation fails, check that the scripts directory exists in .venv
   - On Windows, you may need to run PowerShell as administrator or adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

2. Dependencies installation issues:
   - If pip install fails with permission errors, ensure the virtual environment is activated
   - If package conflicts occur, try installing packages one by one to identify conflicts
   - For psycopg2-binary installation issues on Windows, you might need to install Visual C++ Build Tools
   - If pandas wheel building takes too long, consider using a pre-built wheel: `pip install --only-binary=:all: pandas`
   - If you encounter `ModuleNotFoundError` when importing from the package, ensure you've installed the package in development mode with `pip install -e .`

3. Linter configuration issues:
   - If linters aren't recognized, ensure they're installed in the virtual environment
   - For VS Code integration issues, install the Python extension and reload the window
   - If black/isort configurations conflict, ensure the profiles are compatible

4. Makefile issues:
   - If `make` command is not found on Windows, install it using Chocolatey
   - If Chocolatey installation fails with permission errors, run PowerShell as administrator:

     ```powershell
     Start-Process powershell -Verb RunAs -ArgumentList "choco install make -y"
     ```

   - If lock file errors occur during installation, you may need to delete the lock file mentioned in the error message
   - Ensure Makefile commands use tabs for indentation, not spaces
   - For Windows users without make, you can run the individual commands directly

5. VS Code linting and formatting issues:
   - If autoformatting doesn't work on save, check that `"editor.formatOnSave": true` is set
   - If linting doesn't work, ensure the Python extension is installed and activated
   - Try reloading the VS Code window (Ctrl+Shift+P, then "Developer: Reload Window")
   - Verify the Python interpreter is correctly set to the virtual environment
   - Check the VS Code output panel (View > Output) and select "Python" to see linting errors
   - If you see "Import X could not be resolved from source" errors in VS Code/Pylance, but the code runs correctly, this is likely just an IDE issue. Try reloading the window or restarting VS Code. You can also try adding the package to the `python.analysis.extraPaths` setting in `.vscode/settings.json`.

6. Project structure issues:
   - On Windows, use appropriate mkdir commands or create directories through Explorer
   - Ensure proper permissions to create directories and files
   - For nested directory creation issues on Windows, create parent directories first

7. Database connection issues:
   - For SQLite: Ensure the database file path is writable
   - For PostgreSQL: Verify connection parameters and that the PostgreSQL service is running
   - Check firewall settings if connecting to a remote database

8. Pipeline execution issues:
   - Verify input file formats match expected formats
   - Check for sufficient disk space for large data operations
   - For memory errors during processing, try reducing batch_size in settings.yaml

9. Logging issues:
   - If using Loguru, ensure it's properly installed: `pip install loguru`
   - For file permission issues, check that the logs directory exists and is writable
   - If log rotation isn't working, verify the rotation and retention parameters

10. Testing issues:

- If pytest can't find your modules, ensure you have a `conftest.py` file that adds the project root to the Python path
- If you get type errors in test files, check the `pytest.ini` configuration for mypy settings
- For coverage issues, verify the `[tool.coverage.run]` and `[tool.coverage.report]` settings in `pyproject.toml`
- If tests pass locally but fail in CI, check for environment-specific issues like file paths or dependencies
- For slow tests, consider using pytest's `-xvs` flags for more verbose output and to stop on the first failure

## Completion Status

This SOP has been completed and verified on March 3, 2025. All steps have been tested and confirmed to work correctly. The environment setup is now complete and ready for development.

## References

- [ETL Pipeline v4 Implementation Guide](../../docs/guide/guide.md)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [VS Code Python setup](https://code.visualstudio.com/docs/python/python-tutorial)
- [Loguru documentation](https://github.com/Delgan/loguru)

## Revision History

| Version | Date       | Author   | Changes                                                                                     |
| ------- | ---------- | -------- | ------------------------------------------------------------------------------------------- |
| 1.0     | 2025-03-03 | ETL Team | Initial version                                                                             |
| 1.1     | 2025-03-03 | ETL Team | Enhanced with version bounds, Makefile, example usage, expanded troubleshooting             |
| 1.2     | 2025-03-03 | ETL Team | Added detailed VS Code setup, linting and autoformat configuration                          |
| 1.3     | 2025-03-03 | ETL Team | Added Makefile troubleshooting section for Windows users                                    |
| 1.4     | 2025-03-03 | ETL Team | Updated to use Loguru for logging, added type stubs, expanded troubleshooting               |
| 1.5     | 2025-03-03 | ETL Team | Updated Makefile to include development mode installation, added notes about its importance |
| 1.6     | 2025-03-03 | ETL Team | Added unit testing setup, improved test configuration, and updated VS Code settings         |
