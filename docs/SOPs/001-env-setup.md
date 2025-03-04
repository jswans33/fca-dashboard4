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
   - Windows: `.venv\Scripts\activate`
   - Unix/MacOS: `source .venv/bin/activate`

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
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### 3. Linter Configuration

1. Create `pyproject.toml` with linting configurations:

   ```toml
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
   testpaths = ["tests"]
   python_files = "test_*.py"
   
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

2. Test linters:

   ```bash
   black --check .
   isort --check .
   flake8 .
   mypy .
   ```

3. Create a `Makefile` for convenience commands:

   ```makefile
   .PHONY: lint test format install

   lint:
    black --check .
    isort --check .
    flake8 .
    mypy .

   format:
    black .
    isort .

   test:
    pytest tests/

   install:
    python -m pip install --upgrade pip
    pip install -r requirements.txt
   ```

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

2. Ensure the logging utility is implemented early in development:

   ```bash
   # Test logging configuration
   python -c "from fca_dashboard.utils.logging_config import configure_logging; configure_logging('DEBUG', 'logs/fca_dashboard.log')"
   ```

### 8. Example Pipeline Usage

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
   
   # Format code
   make format
   ```

### 6. VS Code Workspace Setup

1. Create a `.vscode` directory:

   ```bash
   mkdir -p .vscode
   ```

2. Create `.vscode/settings.json` for linting and autoformatting configuration:

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
       "tests"
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
     }
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

3. Linter configuration issues:
   - If linters aren't recognized, ensure they're installed in the virtual environment
   - For VS Code integration issues, install the Python extension and reload the window
   - If black/isort configurations conflict, ensure the profiles are compatible

7. VS Code linting and formatting issues:
   - If autoformatting doesn't work on save, check that `"editor.formatOnSave": true` is set
   - If linting doesn't work, ensure the Python extension is installed and activated
   - Try reloading the VS Code window (Ctrl+Shift+P, then "Developer: Reload Window")
   - Verify the Python interpreter is correctly set to the virtual environment
   - Check the VS Code output panel (View > Output) and select "Python" to see linting errors

4. Project structure issues:
   - On Windows, use appropriate mkdir commands or create directories through Explorer
   - Ensure proper permissions to create directories and files
   - For nested directory creation issues on Windows, create parent directories first

5. Database connection issues:
   - For SQLite: Ensure the database file path is writable
   - For PostgreSQL: Verify connection parameters and that the PostgreSQL service is running
   - Check firewall settings if connecting to a remote database

6. Pipeline execution issues:
   - Verify input file formats match expected formats
   - Check for sufficient disk space for large data operations
   - For memory errors during processing, try reducing batch_size in settings.yaml

## References

- [ETL Pipeline v4 Implementation Guide](../../docs/guide/guide.md)
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [VS Code Python setup](https://code.visualstudio.com/docs/python/python-tutorial)

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-03-03 | ETL Team | Initial version |
| 1.1 | 2025-03-03 | ETL Team | Enhanced with version bounds, Makefile, example usage, expanded troubleshooting |
| 1.2 | 2025-03-03 | ETL Team | Added detailed VS Code setup, linting and autoformat configuration |
