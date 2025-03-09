# NexusML Refactoring: Detailed Tasks

This document provides a detailed breakdown of tasks for each phase of the NexusML refactoring work plan. It expands on the high-level plan with specific implementation details, file changes, and hierarchical task organization.

## Phase 1: Configuration Centralization (✅ COMPLETED)

### 1. Enhance Config Module

#### 1.1 Create ConfigurationManager Class ✅
- ✅ Implement caching of loaded configurations for performance
- ✅ Add support for environment-specific configurations
- ✅ Create methods for merging configurations from different sources

#### 1.2 Implement Type-Safe Configuration Access ✅
- ✅ Add type-safe access to configuration values
- ✅ Implement validation of configuration values against schemas
- ✅ Add support for default values and fallbacks

#### 1.3 Add Configuration Validation ✅
- ✅ Create JSON Schema definitions for each configuration type
- ✅ Implement validation against schemas during loading
- ✅ Add helpful error messages for invalid configurations

**Files Created/Modified:**
- ✅ `nexusml/config/manager.py` - New file for ConfigurationManager
- ✅ `nexusml/config/__init__.py` - Updated to expose new functionality
- ✅ `nexusml/config/schemas/` - New directory for JSON Schema definitions
- ✅ `nexusml/config/schemas/data_config_schema.json`
- ✅ `nexusml/config/schemas/feature_config_schema.json`
- ✅ `nexusml/config/schemas/model_card_schema.json` - Added for model card validation

### 2. Create Configuration Interfaces

#### 2.1 Define Base Interfaces ✅
- ✅ Create `ConfigSection` base interface
- ✅ Define `DataConfig` interface for data configuration
- ✅ Define `FeatureConfig` interface for feature engineering configuration
- ✅ Define `ModelConfig` interface for model configuration
- ✅ Define `PipelineConfig` interface for pipeline configuration
- ✅ Define `ModelCardConfig` interface for model card information

#### 2.2 Implement Concrete Classes ✅
- ✅ Create `YamlDataConfig` implementation
- ✅ Create `YamlFeatureConfig` implementation
- ✅ Create `YamlModelConfig` implementation
- ✅ Create `YamlPipelineConfig` implementation

#### 2.3 Ensure Backward Compatibility ✅
- ✅ Add adapter methods for legacy code
- ✅ Create compatibility layer for existing configuration formats
- ✅ Add deprecation warnings for old access patterns

**Files Created/Modified:**
- ✅ `nexusml/config/interfaces.py` - New file for configuration interfaces
- ✅ `nexusml/config/implementations/` - New directory for implementations
- ✅ `nexusml/config/implementations/yaml_configs.py`
- ✅ `nexusml/config/implementations/json_configs.py`
- ✅ `nexusml/config/compatibility.py` - For backward compatibility
- ✅ `nexusml/config/sections.py` - For base configuration section class
- ✅ `nexusml/config/model_card.py` - For model card configuration

### 3. Update Path Management

#### 3.1 Extend Path Utilities ✅
- ✅ Create a unified API for path resolution
- ✅ Add support for different path types (absolute, relative, project-relative)
- ✅ Implement path normalization and validation

#### 3.2 Create PathResolver Class ✅
- ✅ Add support for environment-specific path resolution
- ✅ Implement path substitution for variables
- ✅ Add caching for frequently accessed paths

#### 3.3 Add Context-Specific Path Utilities ✅
- ✅ Create helpers for data paths
- ✅ Create helpers for configuration paths
- ✅ Create helpers for output paths
- ✅ Create helpers for reference data paths

**Files Created/Modified:**
- ✅ `nexusml/config/paths.py` - New file for path utilities
- ✅ `nexusml/utils/path_utils.py` - Updated existing utilities
- ✅ `nexusml/config/environment.py` - For environment-specific configuration

### 4. Add Configuration Validation (Additional Task) ✅

#### 4.1 Create Configuration Validator ✅
- ✅ Implement schema-based validation
- ✅ Add additional validation rules for specific configuration types
- ✅ Implement configuration compatibility validation

#### 4.2 Create Test Script ✅
- ✅ Create a test script to verify the functionality of Phase 1
- ✅ Test configuration loading and validation
- ✅ Test path resolution
- ✅ Test model card configuration

**Files Created/Modified:**
- ✅ `nexusml/config/validation.py` - New file for configuration validation
- ✅ `nexusml/tests/test_config_phase1.py` - Test script for Phase 1

## Phase 2: Core Component Refactoring (✅ COMPLETED)

### 0. Type Safety Improvements ✅

#### 0.1 Address mypy Errors ✅
- ✅ Fix type annotations in core components
- ✅ Add proper return type annotations
- ✅ Ensure type safety in all new code
- ✅ Use explicit casting where necessary

#### 0.2 Add Type Stubs ✅
- ✅ Create type stubs for external dependencies
- ✅ Add missing type annotations
- ✅ Improve type inference

**Files Created/Modified:**
- ✅ `nexusml/mypy.ini` - Configuration for mypy
- ✅ `nexusml/types/validation.py` - Type stubs for validation components
- ✅ All new files created in Phase 2 for Data Validation

### 0.3 Leverage Existing Components ✅
- ✅ Identify and document existing components that can be reused
- ✅ Ensure compatibility with the new architecture
- ✅ Refactor as needed to fit into the new design

**Existing Components Leveraged:**
- ✅ `nexusml/core/reference/validation.py` - Reference Validation Functions (adapted in `nexusml/core/validation/adapters.py`)
- `nexusml/core/di/container.py` - Dependency Injection Container
- `nexusml/core/feature_engineering.py` - Feature Engineering Components
- `nexusml/core/model_building.py` - Model Building Components
- `nexusml/core/pipeline/components/model_builder.py` - Model Builder Component

### 1. Data Validation ✅

#### 1.1 Create DataValidator Interface ✅
- ✅ Define clear validation contract based on existing reference validation patterns
- ✅ Add support for different validation levels (error, warning, info)
- ✅ Include methods for getting validation results
- ✅ Ensure compatibility with existing validation functions

#### 1.2 Implement ConfigDrivenValidator ✅
- ✅ Create rule-based validation system using configuration
- ✅ Leverage existing validation functions in `nexusml/core/reference/validation.py`
- ✅ Implement column existence validation
- ✅ Implement data type validation
- ✅ Implement value range validation
- ✅ Implement custom validation rules

#### 1.3 Create Specialized Validators ✅
- ✅ Create `ColumnValidator` for column-level validation
- ✅ Create `RowValidator` for row-level validation
- ✅ Create `DataFrameValidator` for dataset-level validation
- ✅ Ensure all validators follow the same interface

**Files Created/Modified:**
- ✅ `nexusml/core/validation/interfaces.py` - For validation interfaces
- ✅ `nexusml/core/validation/validators.py` - For validator implementations
- ✅ `nexusml/core/validation/rules.py` - For validation rules
- ✅ `nexusml/core/validation/__init__.py` - To expose functionality
- ✅ `nexusml/core/validation/adapters.py` - For adapting existing validation functions
- ✅ `nexusml/types/validation.py` - Type stubs for validation components
- ✅ `nexusml/tests/test_validation.py` - Tests for validation components
- ✅ `nexusml/examples/validation_example.py` - Example usage of validation components
- ✅ `docs/code_review/nexusml_refactoring_phase2_data_validation.md` - Documentation

### 2. Feature Engineering ✅

#### 2.1 Enhance Existing FeatureEngineer Interface ✅
- ✅ Refine the existing `GenericFeatureEngineer` interface in `nexusml/core/feature_engineering.py`
- ✅ Ensure proper implementation of fit, transform, and fit_transform methods
- ✅ Improve configuration-driven behavior
- ✅ Add methods for getting feature information

#### 2.2 Improve ConfigDrivenFeatureEngineer ✅
- ✅ Enhance the existing transformer registry for different transformation types
- ✅ Improve error handling and logging
- ✅ Add support for more transformation types
- ✅ Ensure proper dependency injection

#### 2.3 Refactor Specialized Transformers ✅
- ✅ Refactor existing transformers to follow SOLID principles:
  - ✅ `TextCombiner` for combining text columns
  - ✅ `NumericCleaner` for cleaning numeric columns
  - ✅ `HierarchyBuilder` for building hierarchical columns
  - ✅ `ColumnMapper` for mapping columns
  - ✅ `KeywordClassificationMapper` for keyword-based classification
  - ✅ `ClassificationSystemMapper` for mapping to classification systems
- ✅ Add new transformers as needed:
  - ✅ `MissingValueHandler` for handling missing values
  - ✅ `OutlierDetector` for detecting outliers
  - ✅ `TextNormalizer` for normalizing text
  - ✅ `TextTokenizer` for tokenizing text
  - ✅ `NumericScaler` for scaling numeric columns
  - ✅ `OneHotEncoder` for one-hot encoding categorical columns
  - ✅ `LabelEncoder` for label encoding categorical columns
  - ✅ `HierarchyExpander` for expanding hierarchical columns
  - ✅ `HierarchyFilter` for filtering based on hierarchical columns

**Files Created/Modified:**
- ✅ `nexusml/core/feature_engineering/interfaces.py` - Extract interfaces from existing code
- ✅ `nexusml/core/feature_engineering/base.py` - Base implementations of interfaces
- ✅ `nexusml/core/feature_engineering/registry.py` - Transformer registry
- ✅ `nexusml/core/feature_engineering/config_driven.py` - Configuration-driven feature engineer
- ✅ `nexusml/core/feature_engineering/compatibility.py` - Backward compatibility
- ✅ `nexusml/core/feature_engineering/__init__.py` - Package initialization
- ✅ `nexusml/core/feature_engineering/transformers/` - Directory for refactored transformers
- ✅ `nexusml/core/feature_engineering/transformers/text.py` - Text transformers
- ✅ `nexusml/core/feature_engineering/transformers/numeric.py` - Numeric transformers
- ✅ `nexusml/core/feature_engineering/transformers/categorical.py` - Categorical transformers
- ✅ `nexusml/core/feature_engineering/transformers/hierarchical.py` - Hierarchical transformers
- ✅ `nexusml/core/feature_engineering/transformers/mapping.py` - Mapping functions
- ✅ `nexusml/types/feature_engineering/interfaces.py` - Type stubs for interfaces
- ✅ `nexusml/tests/test_feature_engineering.py` - Tests for feature engineering components
- ✅ `nexusml/examples/feature_engineering_example.py` - Example usage
- ✅ `docs/code_review/nexusml_refactoring_phase2_feature_engineering.md` - Documentation

### 3. Model Building and Training ✅

#### 3.1 Enhance Existing Model Interfaces ✅
- ✅ Refine the existing `BaseModelBuilder` interface in `nexusml/core/pipeline/base.py`
- ✅ Improve the existing `RandomForestModelBuilder` in `nexusml/core/pipeline/components/model_builder.py`
- ✅ Ensure proper implementation of build_model and optimize_hyperparameters methods
- ✅ Add support for more model types

#### 3.2 Refactor and Extend Model Builders ✅
- ✅ Refactor the existing `RandomForestModelBuilder` to follow SOLID principles
- ✅ Create additional model builders:
  - ✅ `GradientBoostingBuilder` for building gradient boosting models
  - ✅ `EnsembleBuilder` for building ensemble models
- ✅ Ensure proper dependency injection and configuration

#### 3.3 Enhance Model Trainers ✅
- ✅ Refactor the existing model training code in `nexusml/core/model_building.py`
- ✅ Create dedicated trainer classes:
  - ✅ `StandardModelTrainer` for training models
  - ✅ `CrossValidationTrainer` for cross-validation
  - ✅ `GridSearchOptimizer` and `RandomizedSearchOptimizer` for hyperparameter optimization
- ✅ Ensure proper integration with the configuration system

**Files Created/Modified:**
- ✅ `nexusml/core/model_building/interfaces.py` - Interface definitions
- ✅ `nexusml/core/model_building/base.py` - Base implementations
- ✅ `nexusml/core/model_building/compatibility.py` - Backward compatibility
- ✅ `nexusml/core/model_building/__init__.py` - Package initialization
- ✅ `nexusml/core/model_building/builders/random_forest.py` - Random Forest builder
- ✅ `nexusml/core/model_building/builders/gradient_boosting.py` - Gradient Boosting builder
- ✅ `nexusml/core/model_building/builders/ensemble.py` - Ensemble builder
- ✅ `nexusml/core/model_training/trainers/standard.py` - Standard trainer
- ✅ `nexusml/core/model_training/trainers/cross_validation.py` - Cross-validation trainer
- ✅ `nexusml/core/model_training/trainers/hyperparameter_optimizer.py` - Hyperparameter optimizers
- ✅ `nexusml/types/model_building/interfaces.py` - Type stubs for interfaces
- ✅ `nexusml/tests/test_model_building.py` - Tests for model building components
- ✅ `nexusml/examples/model_building_example.py` - Example usage
- ✅ `docs/code_review/nexusml_refactoring_phase2_model_building.md` - Documentation

## Phase 3: Pipeline Orchestration

### 1. Pipeline Components (In Progress)

#### 1.1 Define Pipeline Stage Interfaces (Completed)
- ✅ Create `PipelineStage` base interface
- ✅ Define `DataLoadingStage` interface
- ✅ Define `ValidationStage` interface
- ✅ Define `FeatureEngineeringStage` interface
- ✅ Define `ModelBuildingStage` interface
- ✅ Define `ModelTrainingStage` interface
- ✅ Define `ModelEvaluationStage` interface
- ✅ Define `ModelSavingStage` interface
- ✅ Define `PredictionStage` interface
- ✅ Define `DataSplittingStage` interface

#### 1.2 Implement Pipeline Stages (Completed)
- ✅ Create standard implementations for each stage
- ✅ Add configuration-driven behavior for each stage
- ✅ Implement error handling and logging for each stage
- ✅ Create base implementations for common functionality

#### 1.3 Create Utility Classes (Completed)
- ✅ Split complex stages into smaller, focused stages
- ✅ Use composition to combine functionality
- ✅ Create utility classes for common operations

#### 1.4 Testing and Documentation (Completed)
- ✅ Run unit tests for pipeline stages
- ✅ Run example script to verify functionality
- ✅ Fix issues found during testing:
  - Fixed data loading stage to handle duplicate parameters
  - Fixed data splitting stage to handle duplicate parameters
  - Fixed validation stage to use proper implementation of ColumnValidator
  - Fixed model building stage to handle mixed data types (text and numeric)
  - Fixed prediction stage to handle MultiOutputClassifier correctly
- ✅ Create documentation for pipeline stages

**Files Created/Modified:**
- ✅ `nexusml/core/pipeline/stages/interfaces.py` - For pipeline stage interfaces
- ✅ `nexusml/core/pipeline/stages/base.py` - Base implementations for stages
- ✅ `nexusml/core/pipeline/stages/data_loading.py` - Data loading stages
- ✅ `nexusml/core/pipeline/stages/validation.py` - Validation stages
- ✅ `nexusml/core/pipeline/stages/feature_engineering.py` - Feature engineering stages
- ✅ `nexusml/core/pipeline/stages/data_splitting.py` - Data splitting stages
- ✅ `nexusml/core/pipeline/stages/model_building.py` - Model building stages
- ✅ `nexusml/core/pipeline/stages/model_training.py` - Model training stages
- ✅ `nexusml/core/pipeline/stages/model_evaluation.py` - Model evaluation stages
- ✅ `nexusml/core/pipeline/stages/model_saving.py` - Model saving stages
- ✅ `nexusml/core/pipeline/stages/prediction.py` - Prediction stages
- ✅ `nexusml/core/pipeline/stages/__init__.py` - Package initialization
- ✅ `nexusml/core/pipeline/stages/README.md` - Documentation
- ✅ `nexusml/examples/pipeline_stages_example.py` - Example script
- ✅ `nexusml/tests/unit/core/pipeline/test_pipeline_stages.py` - Unit tests

**Next Steps:**
- ✅ Run unit tests to verify functionality
- ✅ Run example script to verify functionality
- ✅ Fix issues found during testing:
  - Fixed data loading stage to handle duplicate parameters
  - Fixed data splitting stage to handle duplicate parameters
  - Fixed validation stage to use proper implementation of ColumnValidator
  - Fixed model building stage to handle mixed data types (text and numeric)
  - Fixed prediction stage to handle MultiOutputClassifier correctly
- ✅ Update documentation based on testing results
- Proceed with implementing the Pipeline Factory and Integration
=======

### 2. Dependency Injection

#### 2.1 Leverage Existing DI Container ✅
- ✅ Utilize the existing DIContainer implementation in `nexusml/core/di/container.py`
- ✅ Utilize the existing ContainerProvider in `nexusml/core/di/provider.py`
- ✅ Utilize the existing decorators in `nexusml/core/di/decorators.py`
- ✅ Utilize the existing registration functions in `nexusml/core/di/registration.py`
- Add any necessary extensions or improvements
- Create documentation for the DI container usage

#### 2.2 Register Components
- Update the existing registration functions to include new components:
  - Register configuration components:
    - ConfigurationProvider
    - YamlDataConfig, YamlFeatureConfig, YamlModelConfig, YamlPipelineConfig
  - Register validation components:
    - ColumnValidator, RowValidator, DataFrameValidator
    - ConfigDrivenValidator
  - Register feature engineering components:
    - TextCombiner, NumericCleaner, HierarchyBuilder
    - MissingValueHandler, OutlierDetector, TextNormalizer
    - ConfigDrivenFeatureEngineer
  - Register model building components:
    - RandomForestBuilder, GradientBoostingBuilder, EnsembleBuilder
    - BaseModelEvaluator, BaseModelSerializer
  - Register model training components:
    - StandardModelTrainer, CrossValidationTrainer
    - GridSearchOptimizer, RandomizedSearchOptimizer
  - Register pipeline components (to be implemented in Phase 3)

#### 2.3 Update Classes for DI
- Update all classes to use constructor injection
- Ensure proper dependency resolution
- Add factory methods where needed

**Files to Create/Modify:**
- `nexusml/core/di/registration.py` - Update to register new components
- Component registration files for each module

### 3. Pipeline Factory

#### 3.1 Create PipelineFactory
- Implement factory methods for creating pipelines
- Add support for different pipeline types
- Use DI container for resolving dependencies

#### 3.2 Implement Pipeline Types
- Create `TrainingPipeline` for model training
- Create `PredictionPipeline` for making predictions
- Create `EvaluationPipeline` for model evaluation

#### 3.3 Add Extension Points
- Implement plugin system for adding new components
- Use strategy pattern for selecting implementations
- Add configuration-driven component selection

**Files to Create/Modify:**
- `nexusml/core/pipeline/factory.py` - For pipeline factory
- `nexusml/core/pipeline/pipelines/` - Directory for pipeline implementations
- `nexusml/core/pipeline/pipelines/training.py`
- `nexusml/core/pipeline/pipelines/prediction.py`
- `nexusml/core/pipeline/pipelines/evaluation.py`
- `nexusml/core/pipeline/registry.py` - For component registry

## Phase 4: Testing and Documentation

### 1. Unit Tests

#### 1.1 Create Component Tests
- Write tests for configuration components
- Write tests for validation components
- Write tests for feature engineering components
- Write tests for model building components
- Write tests for model training components
- Write tests for pipeline components

#### 1.2 Create Mock Objects
- Create mock implementations of interfaces
- Use dependency injection to inject mocks
- Create test fixtures for common test scenarios

#### 1.3 Ensure Test Coverage
- Aim for at least 80% code coverage
- Focus on testing edge cases and error conditions
- Add property-based testing for complex components

**Files to Create/Modify:**
- `nexusml/tests/unit/config/` - For configuration tests
- `nexusml/tests/unit/validation/` - For validation tests
- `nexusml/tests/unit/feature_engineering/` - For feature engineering tests
- `nexusml/tests/unit/model_building/` - For model building tests
- `nexusml/tests/unit/model_training/` - For model training tests
- `nexusml/tests/unit/pipeline/` - For pipeline tests
- `nexusml/tests/fixtures/` - For test fixtures

### 2. Integration Tests

#### 2.1 Create Pipeline Tests
- Write tests for end-to-end training pipeline
- Write tests for end-to-end prediction pipeline
- Write tests for end-to-end evaluation pipeline

#### 2.2 Create Test Configurations
- Create test configurations for different scenarios
- Test with minimal configurations
- Test with comprehensive configurations

#### 2.3 Verify End-to-End Functionality
- Test with real data
- Verify outputs match expected results
- Test error handling and recovery

**Files to Create/Modify:**
- `nexusml/tests/integration/` - For integration tests
- `nexusml/tests/integration/training/` - For training pipeline tests
- `nexusml/tests/integration/prediction/` - For prediction pipeline tests
- `nexusml/tests/integration/evaluation/` - For evaluation pipeline tests
- `nexusml/tests/data/` - For test data
- `nexusml/tests/configs/` - For test configurations

### 3. Documentation

#### 3.1 Update Code Documentation
- Add docstrings to all classes and methods
- Include examples in docstrings
- Add type hints to all functions and methods

#### 3.2 Create Architecture Documentation
- Document overall architecture
- Create component diagrams
- Document design decisions and trade-offs

#### 3.3 Create Usage Examples
- Create example scripts for common tasks
- Add Jupyter notebooks with examples
- Create step-by-step tutorials

**Files to Create/Modify:**
- Update all source files with improved docstrings
- `docs/architecture.md` - For architecture documentation
- `docs/components/` - Directory for component documentation
- `docs/examples/` - Directory for example scripts
- `docs/tutorials/` - Directory for tutorials
- `notebooks/` - Directory for Jupyter notebooks

## Implementation Timeline

### Phase 1: Configuration Centralization (✅ COMPLETED)
- ✅ Enhance Config Module and Create Configuration Interfaces
- ✅ Update Path Management and Integration Testing
- ✅ Add Configuration Validation and Model Card Support

### Phase 2: Core Component Refactoring (✅ COMPLETED - 3 weeks)
- ✅ Week 1: Type Safety Improvements and Data Validation (Completed March 8, 2025)
- ✅ Week 2: Feature Engineering (Completed March 8, 2025)
- ✅ Week 3: Model Building and Training (Completed March 8, 2025)

### Phase 3: Pipeline Orchestration (2 weeks)
- Week 1: Pipeline Components and Dependency Injection (In Progress)
  - ✅ Pipeline Components: Interfaces and base implementations
  - ✅ Pipeline Components: Concrete implementations
  - ⏳ Pipeline Components: Testing and verification
  - ⏳ Dependency Injection: Register components
- Week 2: Pipeline Factory and Integration

### Phase 4: Testing and Documentation (1 week)
- Days 1-3: Unit Tests and Integration Tests
- Days 4-5: Documentation and Examples

## Migration Strategy

To minimize disruption to ongoing development, we'll use the following migration strategy:

1. **Parallel Implementation**: Implement new components alongside existing ones
2. **Adapter Pattern**: Create adapters to allow new components to work with existing code
3. **Feature Flags**: Use feature flags to gradually enable new components
4. **Incremental Adoption**: Allow teams to adopt new components at their own pace
5. **Deprecation Warnings**: Add deprecation warnings to old components
6. **Documentation**: Provide clear migration guides for developers

This approach allows for a smooth transition while ensuring that existing functionality continues to work throughout the refactoring process.