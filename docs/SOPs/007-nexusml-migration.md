# SOP 007: Migration of Classifier Module to NexusML

## Overview

This SOP outlines the process for migrating the existing classifier module from
the FCA Dashboard project to a standalone project called "NexusML".

## Objective

Create a modern, standalone machine learning classification engine that can be
maintained and developed independently from the main FCA Dashboard project.

## Prerequisites

- Git installed
- Python 3.8+ installed
- Access to the FCA Dashboard repository

## Migration Tasks

### Phase 1: Project Setup

- [x] 1.1 Create a new directory structure for NexusML
- [x] 1.2 Initialize a new Git repository (using existing repository in monorepo
      style)
- [x] 1.3 Set up Python project structure (pyproject.toml, setup.py)
- [x] 1.4 Create a virtual environment for development

### Phase 2: Code Migration

- [x] 2.1 Copy core classifier files to the new project

  - [x] 2.1.1 Migrate `__init__.py` (created basic structure)
  - [x] 2.1.2 Migrate `data_preprocessing.py`
  - [x] 2.1.3 Migrate `evaluation.py`
  - [x] 2.1.4 Migrate `feature_engineering.py`
  - [x] 2.1.5 Migrate `model_building.py`
  - [x] 2.1.6 Migrate `model.py`
  - [x] 2.1.7 Migrate `training.py` (functionality included in model.py)
  - [x] 2.1.8 Migrate deprecated models
  - [x] 2.1.9 Migrate ingest data files

- [x] 2.2 Migrate related utilities

  - [x] 2.2.1 Migrate `verify_classifier.py`
  - [x] 2.2.2 Create utility module structure

- [x] 2.3 Migrate examples

  - [x] 2.3.1 Migrate `classifier_example.py` (as advanced_example.py)
  - [x] 2.3.2 Migrate `classifier_example_simple.py` (as simple_example.py)
  - [x] 2.3.3 Migrate example outputs (created directory structure)

- [x] 2.4 Migrate tests

  - [x] 2.4.1 Migrate `test_classifier_pipeline.py` (as test_pipeline.py)
  - [x] 2.4.2 Set up test infrastructure

- [x] 2.5 Migrate documentation and diagrams
  - [x] 2.5.1 Migrate classifier diagrams (created directory structure)
  - [x] 2.5.2 Create README.md with project overview

### Phase 3: Code Refactoring

- [x] 3.1 Update import statements in all files
- [x] 3.2 Refactor package structure to follow best practices
- [x] 3.3 Remove any FCA Dashboard specific dependencies (with compatibility
      layer)
- [x] 3.4 Implement proper error handling
- [x] 3.5 Update documentation strings

### Phase 4: Integration

- [ ] 4.1 Create an integration layer for FCA Dashboard
- [ ] 4.2 Update FCA Dashboard to use NexusML as an external dependency
- [ ] 4.3 Test integration points
- [ ] 4.4 Document integration approach

### Phase 5: Testing and Validation

- [ ] 5.1 Run all tests to ensure functionality
- [ ] 5.2 Validate model performance matches original
- [ ] 5.3 Perform integration testing with FCA Dashboard
- [ ] 5.4 Document test results

### Phase 6: Deployment

- [ ] 6.1 Package NexusML for distribution
- [ ] 6.2 Create release documentation
- [ ] 6.3 Update FCA Dashboard to use the packaged version
- [ ] 6.4 Remove old classifier code from FCA Dashboard

## Dependencies

- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn

## Dependency Management

### Virtual Environment Strategy

For this monorepo structure, we have two options for managing virtual
environments:

1. **Single Virtual Environment (Recommended for Development)**

   - Install both fca_dashboard and nexusml in the same virtual environment
   - Simpler to manage and use both packages together
   - Use `make install` followed by `make nexusml-install` to set up

2. **Separate Virtual Environments (Recommended for Production/Extraction)**
   - Create a dedicated virtual environment for nexusml when extracting it
   - Better isolation and cleaner dependency management
   - Use `python -m venv nexusml-venv` and then activate it before installing

For the current monorepo approach, we recommend using a single virtual
environment during development, and documenting how to set up a separate
environment when nexusml is extracted into its own repository.

### Using uv for Dependency Management

For managing dependencies in this complex monorepo structure, we recommend using
`uv`, a modern Python packaging tool:

- **Why uv?**

  - Faster installation of packages compared to pip
  - Better handling of dependency resolution
  - Improved caching for faster builds
  - Support for multiple Python environments
  - Compatible with pyproject.toml configuration

- **Installation:**

  ```bash
  pip install uv
  ```

- **Usage:**

  ```bash
  # Create a virtual environment
  uv venv

  # Install dependencies
  uv pip install -e .

  # Install development dependencies
  uv pip install -e ".[dev]"
  ```

- **Managing Multiple Packages:** Since we're using a monorepo approach with
  both fca_dashboard and nexusml, uv can help manage dependencies more
  efficiently across both packages.

- **Makefile Integration:** The following make targets have been added to
  support NexusML:

  ```
  # Test NexusML
  make nexusml-test
  make nexusml-test-unit
  make nexusml-test-integration

  # Generate coverage report for NexusML
  make nexusml-coverage

  # Install NexusML
  make nexusml-install

  # Create a dedicated virtual environment for NexusML
  make nexusml-venv

  # Install uv package manager
  make install-uv

  # Install NexusML using uv (recommended)
  make nexusml-install-uv

  # Run NexusML examples
  make nexusml-run-simple
  make nexusml-run-advanced
  ```

## File Structure for New Project

### Actual Implementation

We implemented a flat package structure to simplify installation in the
monorepo:

```
nexusml/
├── pyproject.toml
├── setup.py
├── README.md
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model.py
│   └── deprecated/
├── utils/
│   ├── __init__.py
│   └── verification.py
├── ingest/
│   ├── __init__.py
│   └── data/
├── examples/
│   ├── __init__.py
│   ├── simple_example.py
│   ├── advanced_example.py
│   └── outputs/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   └── integration/
│       ├── __init__.py
│       └── test_integration.py
└── docs/
    ├── diagrams/
    └── api/
```

### Package Configuration

The package is configured in pyproject.toml to use a flat layout:

```toml
[tool.setuptools]
packages = ["core", "utils", "ingest", "examples"]
package-dir = {"" = "."}
```

This approach allows for easier integration in the monorepo while still
maintaining a clean structure.

### Import Statements

Due to the flat package structure, imports should use:

```python
from core.model import train_enhanced_model, predict_with_enhanced_model
```

rather than:

```python
from nexusml.core.model import train_enhanced_model, predict_with_enhanced_model
```

This is documented in the README.md file.

## Rollback Plan

If issues arise during migration:

1. Revert any changes to FCA Dashboard
2. Document specific issues encountered
3. Develop targeted solutions for each issue
4. Retry migration with updated approach

## Approval

- [ ] Migration plan reviewed and approved
- [ ] Testing strategy approved
- [ ] Integration approach approved

## Completion Criteria

- [ ] All tests pass in the new project
- [ ] FCA Dashboard successfully integrates with NexusML
- [ ] Documentation is complete and accurate
- [ ] Old classifier code is removed from FCA Dashboard
