# NexusML Dependencies

This document provides a comprehensive list of dependencies required by NexusML, including core dependencies, optional dependencies, and development dependencies.

## Core Dependencies

These dependencies are required for the basic functionality of NexusML:

| Dependency | Version | Description |
|------------|---------|-------------|
| scikit-learn | >=1.0.0 | Machine learning library for model building and evaluation |
| pandas | >=1.3.0 | Data manipulation and analysis library |
| numpy | >=1.20.0 | Numerical computing library |
| matplotlib | >=3.4.0 | Data visualization library |
| seaborn | >=0.11.0 | Statistical data visualization library |
| imbalanced-learn | >=0.8.0 | Library for handling imbalanced datasets |
| pyyaml | >=6.0 | YAML parser and emitter for configuration files |
| python-dotenv | >=0.19.0 | Loading environment variables from .env files |
| tqdm | >=4.62.0 | Progress bar library for long-running operations |
| openpyxl | >=3.1.0 | Library for reading and writing Excel files |
| sqlalchemy | >=2.0.0 | SQL toolkit and Object-Relational Mapping (ORM) library |
| alembic | >=1.9.0 | Database migration tool for SQLAlchemy |
| psycopg2-binary | >=2.9.5 | PostgreSQL adapter for Python |
| python-dateutil | >=2.8.2 | Extensions to the standard datetime module |
| loguru | >=0.6.0 | Library for simplified logging |

## Optional Dependencies

These dependencies are optional and provide additional functionality:

### AI Features

```bash
pip install "nexusml[ai]"
```

| Dependency | Version | Description |
|------------|---------|-------------|
| anthropic | >=0.5.0 | Client for the Anthropic Claude API (for OmniClass description generation) |

## Development Dependencies

These dependencies are required for development and testing:

```bash
pip install "nexusml[dev]"
```

| Dependency | Version | Description |
|------------|---------|-------------|
| pytest | >=7.0.0 | Testing framework |
| pytest-cov | >=3.0.0 | Coverage plugin for pytest |
| black | >=22.0.0 | Code formatter |
| isort | >=5.10.0 | Import sorter |
| flake8 | >=4.0.0 | Linter for style guide enforcement |
| mypy | >=0.9.0 | Static type checker |
| ruff | >=0.0.262 | Fast Python linter |
| coverage | >=7.0.0 | Code coverage measurement tool |

### Type Checking

| Dependency | Version | Description |
|------------|---------|-------------|
| types-PyYAML | >=6.0.0 | Type stubs for PyYAML |
| types-requests | >=2.29.0 | Type stubs for requests |
| types-setuptools | >=65.0.0 | Type stubs for setuptools |
| types-toml | >=0.10.0 | Type stubs for toml |

## System Requirements

- Python 3.8 or higher
- Operating System: Platform independent (Windows, macOS, Linux)

## Installation

### Basic Installation

```bash
pip install nexusml
```

### With AI Features

```bash
pip install "nexusml[ai]"
```

### Development Installation

```bash
git clone https://github.com/your-org/nexusml.git
cd nexusml
pip install -e ".[dev]"
```

## Dependency Management

NexusML uses a combination of pyproject.toml and requirements.txt for dependency management:

- pyproject.toml: Used for package metadata and dependencies for the nexusml package
- requirements.txt: Used for development environment setup and additional dependencies

## Compatibility Notes

- NexusML is tested with Python 3.8, 3.9, 3.10, and 3.11
- For optimal performance, using the latest versions of dependencies is recommended
- When using the AI features, an Anthropic API key is required and should be set as the ANTHROPIC_API_KEY environment variable