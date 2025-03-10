# NexusML Immediate Improvements

This document outlines immediate improvements needed for the NexusML project to
make it a production-ready, maintainable machine learning package.

## Current Issues

### 1. Directory Structure Inconsistencies

- **Multiple Output Directories**

  - `nexusml/output/` - Internal output directory
  - `nexusml/outputs/` - Another internal output directory
  - `outputs/` - Root-level output directory
  - **Problem**: Causes confusion about where outputs should be stored
  - **Affected files**: Multiple scripts reference different output paths

- **Multiple Example Directories**

  - `examples/` - Root-level examples
  - `nexusml/examples/` - Package-level examples
  - `fca_dashboard/examples/` - Legacy examples
  - **Problem**: Difficult to find relevant examples, unclear which are current

- **Inconsistent Project Structure**
  - Project has a well-defined structure in
    `nexusml/projects/mech_equipment_classifier/README.md`
  - Actual implementation doesn't follow this structure
  - **Problem**: New developers can't easily understand the codebase
    organization

### 2. Testing Limitations

- **Verification Script Issues**

  - Current verification script (`nexusml/tests/verification_script.py`) has
    warnings:
    - Multiclass-multioutput not supported in scikit-learn scoring
    - Single label confusion matrix warnings
    - Missing output path configurations
  - **Problem**: Tests pass but with numerous warnings

- **Test Coverage Unknown**
  - No coverage reports generated
  - Unclear which components have adequate test coverage
  - **Problem**: Can't identify untested or under-tested code

### 3. Configuration Management

- **Missing Output Configurations**

  - Warning: `No output_dir specified in config, skipping model saving stage`
  - Warning: `No output_path specified in config, results will not be saved`
  - **Problem**: Default configurations are incomplete

- **Inconsistent Configuration Loading**
  - Multiple configuration loading mechanisms
  - **Problem**: Difficult to understand how configuration works

### 4. Placeholder Implementations

- **Incomplete Core Modules**

  - Many core modules have placeholder or stub implementations
  - Files contain "TODO" comments or "future implementation" notes
  - **Problem**: Critical functionality is missing or incomplete

- **Lack of Simple Working Implementations**
  - Some modules have complex interfaces but minimal functionality
  - Missing basic implementations that work end-to-end
  - **Problem**: Cannot use the framework for real projects without significant
    custom code

### 5. Documentation Gaps

- **Outdated Documentation**

  - Documentation plan exists but implementation is incomplete
  - API reference is extensive but may not match current implementation
  - **Problem**: Developers can't rely on documentation

- **Missing Docker Documentation**
  - No documentation for containerization
  - **Problem**: Deployment in containers is not supported

## Improvement Plan

### 1. Implement Core Functionality

- **Identify Placeholder Modules**

  - [ ] Review all core files for placeholder/stub implementations
  - [ ] Identify modules marked with "TODO" or "future implementation" comments
  - [ ] Create inventory of modules needing implementation
  - **Directories to focus on**:
    - `nexusml/core/pipeline/`
    - `nexusml/core/feature_engineering/`
    - `nexusml/core/model_building/`
    - `nexusml/core/model_training/`

- **Implement Simple Working Versions**

  - [ ] Replace placeholder implementations with simple but functional code
  - [ ] Ensure each module has basic functionality that works end-to-end
  - [ ] Add comprehensive tests for each implementation
  - [ ] Document implementation details and limitations
  - **Implementation priorities**:
    - Data loading components
    - Feature engineering transformers
    - Model building components
    - Pipeline stages

- **Validate Implementations**
  - [ ] Run verification script with new implementations
  - [ ] Test with sample datasets
  - [ ] Benchmark performance
  - [ ] Document any remaining limitations

### 2. Standardize Directory Structure

- **Consolidate Output Directories**

  - [ ] Standardize on `nexusml/output/` for all outputs
  - [ ] Update all scripts to use this path
  - [ ] Add migration script to move existing outputs
  - **Files to update**:
    - `nexusml/train_model_pipeline.py`
    - `nexusml/train_model_pipeline_v2.py`
    - `nexusml/predict.py`
    - `nexusml/predict_v2.py`
    - `nexusml/core/pipeline/pipelines/*.py`

- **Consolidate Example Directories**

  - [ ] Move all examples to `nexusml/examples/`
  - [ ] Categorize examples by functionality
  - [ ] Add README.md to each example directory
  - [ ] Deprecate or remove outdated examples
  - **Files to update**:
    - Move `examples/*` to `nexusml/examples/`
    - Move relevant examples from `fca_dashboard/examples/` to
      `nexusml/examples/`

- **Align with Project Structure Template**
  - [ ] Reorganize codebase to match structure in
        `nexusml/projects/mech_equipment_classifier/README.md`
  - [ ] Create missing directories
  - [ ] Move files to appropriate locations
  - **Key directories to create/organize**:
    - `nexusml/data/` - All data files
    - `nexusml/models/` - Saved models
    - `nexusml/notebooks/` - Jupyter notebooks
    - `nexusml/scripts/` - Utility scripts

### 3. Improve Testing

- **Fix Verification Script Warnings**

  - [ ] Implement custom scoring function for multiclass-multioutput
  - [ ] Add explicit labels parameter to confusion matrix
  - [ ] Add default output paths to configurations
  - **Files to update**:
    - `nexusml/tests/verification_script.py`
    - `nexusml/core/model_building/base.py`
    - `nexusml/core/pipeline/components/model_evaluator.py`

- **Add Test Coverage Reporting**

  - [ ] Configure pytest-cov for coverage reporting
  - [ ] Set coverage targets (aim for 80%+)
  - [ ] Add coverage reports to CI pipeline
  - **Files to create/update**:
    - `pytest.ini` or `.coveragerc`
    - CI configuration files

- **Expand Test Suite**
  - [ ] Add unit tests for all core components
  - [ ] Add integration tests for pipelines
  - [ ] Add end-to-end tests with real-world data
  - **Directories to focus on**:
    - `nexusml/core/pipeline/`
    - `nexusml/core/feature_engineering/`
    - `nexusml/core/model_building/`

### 3. Enhance Configuration System

- **Standardize Configuration**

  - [ ] Create default configurations with all required fields
  - [ ] Add validation for configuration files
  - [ ] Document all configuration options
  - **Files to update**:
    - `nexusml/config/nexusml_config.yml`
    - `nexusml/config/manager.py`
    - `nexusml/config/validation.py`

- **Implement Environment Variable Override**
  - [ ] Ensure all configuration options can be overridden
  - [ ] Document environment variable naming convention
  - [ ] Add validation for environment variables
  - **Files to update**:
    - `nexusml/config/manager.py`

### 4. Improve Documentation

- **Update API Documentation**

  - [ ] Verify and update all docstrings
  - [ ] Generate updated API reference
  - [ ] Add examples to docstrings
  - **Files to update**:
    - All Python files in `nexusml/core/`

- **Create User Guides**
  - [ ] Quick start guide
  - [ ] Installation guide
  - [ ] Configuration guide
  - [ ] Examples guide
  - **Files to create/update**:
    - `nexusml/docs/quick_start.md`
    - `nexusml/docs/installation_guide.md` (update)
    - `nexusml/docs/configuration_guide.md`
    - `nexusml/docs/examples_guide.md`

### 6. Add Docker Support

- **Create Docker Configuration**

  - [ ] Create Dockerfile for NexusML
  - [ ] Create docker-compose.yml for development
  - [ ] Add Docker build scripts
  - **Files to create**:
    - `Dockerfile`
    - `docker-compose.yml`
    - `scripts/docker_build.sh`

- **Document Docker Usage**
  - [ ] Add Docker installation instructions
  - [ ] Document Docker development workflow
  - [ ] Document Docker deployment
  - **Files to create**:
    - `nexusml/docs/docker_guide.md`

## Implementation Priorities

### Phase 1: Foundation (Immediate)

1. **Implement Core Functionality**

   - Identify placeholder modules
   - Implement simple working versions
   - Validate implementations
   - Verify the "C:/Repos/fca-dashboard4/.venv/Scripts/python.exe
     c:/Repos/fca-dashboard4/nexusml/tests/verification_script.py" still passes.

2. **Standardize Directory Structure**

   - Consolidate output directories
   - Align with project structure template

3. **Fix Critical Issues**
   - Fix verification script warnings
   - Update default configurations

### Phase 2: Quality

1. **Improve Testing**

   - Add test coverage reporting
   - Expand test suite for core components

2. **Enhance Documentation**
   - Update API documentation
   - Create quick start guide

### Phase 3: Deployment

1. **Add Docker Support**

   - Create Docker configuration
   - Document Docker usage

2. **Finalize Documentation**
   - Complete all user guides
   - Verify all examples

## Project Template for New Projects

For new projects using NexusML, we recommend following the structure defined in
`nexusml/projects/mech_equipment_classifier/README.md`. This provides a
standardized approach to organizing machine learning projects with NexusML.

Key components of the project template:

- Configuration files in `config/`
- Data files in `data/`
- Trained models in `models/`
- Jupyter notebooks in `notebooks/`
- Output files in `outputs/`
- Utility scripts in `scripts/`
- Source code in `src/`
- Tests in `tests/`

New projects should not need to clone the entire NexusML repository. Instead,
they should:

1. Install NexusML as a dependency
2. Follow the project template structure
3. Configure NexusML through configuration files

## Conclusion

By addressing these immediate improvements, NexusML will become a more robust,
maintainable, and user-friendly machine learning package. The implementation of
core functionality with simple working versions, standardized directory
structure, improved testing, enhanced configuration system, better
documentation, and Docker support will make it easier for developers to use and
contribute to the project.
