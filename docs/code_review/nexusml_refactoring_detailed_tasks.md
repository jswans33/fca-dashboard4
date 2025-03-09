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

## Phase 3: Pipeline Orchestration (✅ COMPLETED)

### 1. Pipeline Components (✅ COMPLETED)

#### 1.1 Define Pipeline Stage Interfaces (✅ COMPLETED)
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

#### 1.2 Implement Pipeline Stages (✅ COMPLETED)
- ✅ Create standard implementations for each stage
- ✅ Add configuration-driven behavior for each stage
- ✅ Implement error handling and logging for each stage
- ✅ Create base implementations for common functionality

#### 1.3 Create Utility Classes (✅ COMPLETED)
- ✅ Split complex stages into smaller, focused stages
- ✅ Use composition to combine functionality
- ✅ Create utility classes for common operations

#### 1.4 Testing and Documentation (✅ COMPLETED)
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
- ✅ Register pipeline components with the dependency injection container
- ✅ Implement the Pipeline Factory and Integration
=======

### 2. Dependency Injection (✅ COMPLETED)

#### 2.1 Leverage Existing DI Container ✅
- ✅ Utilize the existing DIContainer implementation in `nexusml/core/di/container.py`
- ✅ Utilize the existing ContainerProvider in `nexusml/core/di/provider.py`
- ✅ Utilize the existing decorators in `nexusml/core/di/decorators.py`
- ✅ Utilize the existing registration functions in `nexusml/core/di/registration.py`
- ✅ Add necessary extensions for pipeline component registration
- ✅ Create documentation for the DI container usage

#### 2.2 Register Components ✅
- ✅ Update the existing registration functions to include new components:
  - ✅ Register configuration components:
    - ✅ ConfigurationManager
  - ✅ Register pipeline components:
    - ✅ ConfigurableDataLoadingStage
    - ✅ ConfigDrivenValidationStage
    - ✅ SimpleFeatureEngineeringStage, TextFeatureEngineeringStage, NumericFeatureEngineeringStage
    - ✅ RandomSplittingStage
    - ✅ ConfigDrivenModelBuildingStage, RandomForestModelBuildingStage, GradientBoostingModelBuildingStage
    - ✅ StandardModelTrainingStage
    - ✅ ClassificationEvaluationStage
    - ✅ ModelCardSavingStage
    - ✅ ModelLoadingStage
    - ✅ OutputSavingStage, EvaluationOutputSavingStage
    - ✅ StandardPredictionStage, ProbabilityPredictionStage

**Detailed Steps for Component Registration:**
1. Create a new file `nexusml/core/di/pipeline_registration.py` for registering pipeline components
2. Implement registration functions for each component category:
   ```python
   def register_pipeline_stages(container: DIContainer) -> None:
       # Register data loading stages
       container.register(ConfigurableDataLoadingStage, lambda c: ConfigurableDataLoadingStage())
       
       # Register validation stages
       container.register(ConfigDrivenValidationStage, lambda c: ConfigDrivenValidationStage(
           config_manager=c.resolve(ConfigurationManager)
       ))
       
       # Register feature engineering stages
       container.register(SimpleFeatureEngineeringStage, lambda c: SimpleFeatureEngineeringStage())
       container.register(TextFeatureEngineeringStage, lambda c: TextFeatureEngineeringStage())
       container.register(NumericFeatureEngineeringStage, lambda c: NumericFeatureEngineeringStage())
       
       # Register data splitting stages
       container.register(RandomSplittingStage, lambda c: RandomSplittingStage())
       
       # Register model building stages
       container.register(ConfigDrivenModelBuildingStage, lambda c: ConfigDrivenModelBuildingStage(
           config_manager=c.resolve(ConfigurationManager)
       ))
       container.register(RandomForestModelBuildingStage, lambda c: RandomForestModelBuildingStage(
           config_manager=c.resolve(ConfigurationManager)
       ))
       container.register(GradientBoostingModelBuildingStage, lambda c: GradientBoostingModelBuildingStage(
           config_manager=c.resolve(ConfigurationManager)
       ))
       
       # Register model training stages
       container.register(StandardModelTrainingStage, lambda c: StandardModelTrainingStage())
       
       # Register model evaluation stages
       container.register(ClassificationEvaluationStage, lambda c: ClassificationEvaluationStage())
       
       # Register model saving stages
       container.register(ModelCardSavingStage, lambda c: ModelCardSavingStage(
           config_manager=c.resolve(ConfigurationManager)
       ))
       
       # Register prediction stages
       container.register(StandardPredictionStage, lambda c: StandardPredictionStage())
       container.register(ProbabilityPredictionStage, lambda c: ProbabilityPredictionStage())
   ```
3. Update the main registration function in `nexusml/core/di/registration.py` to call the pipeline registration function
4. Create a test script to verify that all components can be resolved from the container

#### 2.3 Update Classes for DI
- Update all classes to use constructor injection
- Ensure proper dependency resolution
- Add factory methods where needed

**Files to Create/Modify:**
- `nexusml/core/di/registration.py` - Update to register new components
- Component registration files for each module

### 3. Pipeline Factory (✅ COMPLETED)

#### 3.1 Create PipelineFactory ✅
- ✅ Implement factory methods for creating pipelines
- ✅ Add support for different pipeline types (Training, Prediction, Evaluation)
- ✅ Use DI container for resolving dependencies
- ✅ Add configuration-driven pipeline creation

#### 3.2 Implement Pipeline Types ✅
- ✅ Create `BasePipeline` abstract base class
- ✅ Create `TrainingPipeline` for model training
- ✅ Create `PredictionPipeline` for making predictions
- ✅ Create `EvaluationPipeline` for model evaluation
- ✅ Implement proper error handling and logging

#### 3.3 Add Extension Points ✅
- ✅ Create `ComponentRegistry` for managing pipeline components
- ✅ Implement `PluginManager` for discovering and loading plugins
- ✅ Add support for registering components from modules
- ✅ Create extension points for custom components

**Files Created/Modified:**
- ✅ `nexusml/core/pipeline/factory.py` - For pipeline factory
- ✅ `nexusml/core/pipeline/pipelines/` - Directory for pipeline implementations
- ✅ `nexusml/core/pipeline/pipelines/base.py` - Base pipeline class
- ✅ `nexusml/core/pipeline/pipelines/training.py` - Training pipeline
- ✅ `nexusml/core/pipeline/pipelines/prediction.py` - Prediction pipeline
- ✅ `nexusml/core/pipeline/pipelines/evaluation.py` - Evaluation pipeline
- ✅ `nexusml/core/pipeline/registry.py` - For component registry
- ✅ `nexusml/core/pipeline/plugins.py` - For plugin system
- ✅ `nexusml/tests/unit/core/pipeline/test_factory.py` - Tests for factory
- ✅ `nexusml/tests/unit/core/pipeline/test_registry.py` - Tests for registry
- ✅ `nexusml/tests/unit/core/pipeline/test_plugins.py` - Tests for plugins
- ✅ `nexusml/tests/unit/core/pipeline/pipelines/test_base_pipeline.py` - Tests for base pipeline

**Integration Verification:**
- ✅ Unit tests for all components are passing
- ✅ Component resolution test script verifies end-to-end functionality
- ⚠️ Need to update example scripts for full integration verification:
  - The `pipeline_orchestrator_example.py` script needs updating to work with the new registry
  - The `prediction_pipeline_example.py` script has issues with feature engineering

## Phase 4: Testing and Documentation (⏳ IN PROGRESS)

### 1. Unit Tests (✅ COMPLETED)

#### 1.1 Create Component Tests (✅ COMPLETED)
- ✅ Write tests for configuration components:
  - ✅ Test `ConfigurationManager` with different configuration sources
  - ✅ Test configuration validation against schemas
  - ✅ Test environment-specific configuration loading
  - ✅ Test configuration merging from different sources
  - ✅ Test path resolution with different path types
  - ✅ Example test: `test_config_manager_loads_yaml_config`

- ✅ Write tests for validation components:
  - ✅ Test `DataValidator` with different validation rules
  - ✅ Test column-level validation with `ColumnValidator`
  - ✅ Test row-level validation with `RowValidator`
  - ✅ Test dataset-level validation with `DataFrameValidator`
  - ✅ Test validation result reporting and error handling
  - ✅ Example test: `test_column_validator_detects_invalid_values`

- ✅ Write tests for feature engineering components:
  - ✅ Test `FeatureEngineer` with different transformation types
  - ✅ Test text transformers (e.g., `TextCombiner`, `TextNormalizer`)
  - ✅ Test numeric transformers (e.g., `NumericCleaner`, `NumericScaler`)
  - ✅ Test categorical transformers (e.g., `OneHotEncoder`, `LabelEncoder`)
  - ✅ Test hierarchical transformers (e.g., `HierarchyBuilder`, `HierarchyExpander`)
  - ✅ Example test: `test_text_combiner_combines_multiple_columns`

- ✅ Write tests for model building components:
  - ✅ Test `ModelBuilder` with different model types
  - ✅ Test `RandomForestModelBuilder` with different parameters
  - ✅ Test `GradientBoostingBuilder` with different parameters
  - ✅ Test `EnsembleBuilder` with different base models
  - ✅ Test hyperparameter optimization
  - ✅ Example test: `test_random_forest_builder_creates_valid_model`

- ✅ Write tests for model training components:
  - ✅ Test `ModelTrainer` with different training data
  - ✅ Test `StandardModelTrainer` with different models
  - ✅ Test `CrossValidationTrainer` with different fold configurations
  - ✅ Test `GridSearchOptimizer` and `RandomizedSearchOptimizer`
  - ✅ Test training result reporting
  - ✅ Example test: `test_standard_trainer_fits_model_correctly`

- ✅ Write tests for pipeline components:
  - ✅ Test all pipeline stages with different configurations
  - ✅ Test pipeline stage interfaces and implementations
  - ✅ Test pipeline factory with different pipeline types
  - ✅ Test pipeline orchestrator with different workflows
  - ✅ Test pipeline context for state management
  - ✅ Example test: `test_pipeline_factory_creates_training_pipeline`

#### 1.2 Create Mock Objects (✅ COMPLETED)
- ✅ Create mock implementations of interfaces:
  - ✅ Create `MockDataLoader` that returns predefined data
  - ✅ Create `MockFeatureEngineer` that performs simple transformations
  - ✅ Create `MockModelBuilder` that returns a dummy model
  - ✅ Create `MockModelTrainer` that simulates training
  - ✅ Create `MockModelEvaluator` that returns predefined metrics
  - ✅ Example implementation:
    ```python
    class MockDataLoader(DataLoader):
        def __init__(self, predefined_data=None):
            self.predefined_data = predefined_data or pd.DataFrame({
                "feature1": [1, 2, 3],
                "feature2": ["a", "b", "c"],
                "target": [0, 1, 0]
            })
            
        def load_data(self, data_path=None, **kwargs):
            return self.predefined_data
    ```

- ✅ Use dependency injection to inject mocks:
  - ✅ Register mock implementations in the DI container
  - ✅ Configure tests to use mock implementations
  - ✅ Test component interactions with mocks
  - ✅ Example test setup:
    ```python
    def test_pipeline_with_mocks():
        # Create a DI container
        container = DIContainer()
        
        # Register mock implementations
        container.register(DataLoader, lambda c: MockDataLoader())
        container.register(FeatureEngineer, lambda c: MockFeatureEngineer())
        container.register(ModelBuilder, lambda c: MockModelBuilder())
        
        # Create a pipeline factory with the container
        factory = PipelineFactory(ComponentRegistry(), container)
        
        # Create and test the pipeline
        pipeline = factory.create_pipeline("training")
        result = pipeline.run()
        
        # Assert expected results
        assert result.status == "success"
    ```

- ✅ Create test fixtures for common test scenarios:
  - ✅ Create fixtures for sample data with different characteristics
  - ✅ Create fixtures for common configuration scenarios
  - ✅ Create fixtures for different model types
  - ✅ Example pytest fixture:
    ```python
    @pytest.fixture
    def sample_classification_data():
        """Fixture providing sample classification data."""
        return pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "c", "a", "b"],
            "target": [0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def sample_config():
        """Fixture providing a sample configuration."""
        return {
            "data": {
                "input_path": "test_data.csv",
                "target_column": "target"
            },
            "features": {
                "numeric_columns": ["feature1"],
                "categorical_columns": ["feature2"]
            },
            "model": {
                "type": "random_forest",
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            }
        }
    ```

#### 1.3 Ensure Test Coverage (✅ COMPLETED)
- ✅ Aim for at least 80% code coverage:
  - ✅ Use pytest-cov to measure coverage
  - ✅ Add tests for uncovered code paths
  - ✅ Focus on critical components first
  - ✅ Command to run tests with coverage:
    ```bash
    pytest --cov=nexusml --cov-report=term-missing tests/
    ```

- ✅ Focus on testing edge cases and error conditions:
  - ✅ Test with empty datasets
  - ✅ Test with missing values
  - ✅ Test with invalid configurations
  - ✅ Test with incompatible data types
  - ✅ Test error handling and recovery
  - ✅ Example test:
    ```python
    def test_data_validator_with_empty_dataset():
        validator = DataValidator()
        empty_df = pd.DataFrame()
        
        # Should raise a specific error for empty datasets
        with pytest.raises(ValueError, match="Empty dataset"):
            validator.validate(empty_df)
    ```

- ✅ Add property-based testing for complex components:
  - ✅ Use hypothesis for property-based testing
  - ✅ Define properties that should hold for all inputs
  - ✅ Test with a wide range of generated inputs
  - ✅ Example property-based test:
    ```python
    from hypothesis import given
    from hypothesis.strategies import integers, text, lists
    
    @given(
        n_estimators=integers(min_value=1, max_value=200),
        max_depth=integers(min_value=1, max_value=20)
    )
    def test_random_forest_builder_properties(n_estimators, max_depth):
        builder = RandomForestModelBuilder(n_estimators=n_estimators, max_depth=max_depth)
        model = builder.build_model()
        
        # Properties that should hold for all valid inputs
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert model.get_params()["n_estimators"] == n_estimators
        assert model.get_params()["max_depth"] == max_depth
    ```

**Files to Create/Modify:**
- `nexusml/tests/unit/config/test_config_manager.py` - Test configuration loading and validation
- `nexusml/tests/unit/config/test_path_resolver.py` - Test path resolution
- `nexusml/tests/unit/validation/test_data_validator.py` - Test data validation
- `nexusml/tests/unit/validation/test_column_validator.py` - Test column validation
- `nexusml/tests/unit/feature_engineering/test_feature_engineer.py` - Test feature engineering
- `nexusml/tests/unit/feature_engineering/test_transformers.py` - Test individual transformers
- `nexusml/tests/unit/model_building/test_model_builders.py` - Test model builders
- `nexusml/tests/unit/model_training/test_model_trainers.py` - Test model trainers
- `nexusml/tests/unit/pipeline/test_pipeline_stages.py` - Test pipeline stages
- `nexusml/tests/unit/pipeline/test_pipeline_factory.py` - Test pipeline factory
- `nexusml/tests/unit/pipeline/test_pipeline_orchestrator.py` - Test pipeline orchestrator
- `nexusml/tests/fixtures/data_fixtures.py` - Common data fixtures
- `nexusml/tests/fixtures/config_fixtures.py` - Common configuration fixtures
- `nexusml/tests/fixtures/model_fixtures.py` - Common model fixtures
- `nexusml/tests/conftest.py` - Global pytest configuration and fixtures

### 2. Integration Tests (⏳ IN PROGRESS)

#### 2.1 Create Pipeline Tests (✅ COMPLETED)
- ✅ Write tests for end-to-end training pipeline:
  - ✅ Test the complete training workflow from data loading to model saving
  - ✅ Verify that all pipeline stages interact correctly
  - ✅ Test with different model types (RandomForest, GradientBoosting, Ensemble)
  - ✅ Test with different feature engineering configurations
  - ✅ Example test:
    ```python
    def test_end_to_end_training_pipeline():
        # Create orchestrator
        orchestrator = create_test_orchestrator()
        
        # Run training pipeline
        model, metrics = orchestrator.train_model(
            data_path="tests/data/sample_training_data.csv",
            feature_config_path="tests/configs/feature_config.yml",
            test_size=0.3,
            random_state=42,
            output_dir="tests/output/models",
            model_name="test_model"
        )
        
        # Verify model was created
        assert model is not None
        
        # Verify metrics were calculated
        assert "accuracy" in metrics
        assert "f1" in metrics
        
        # Verify model was saved
        assert Path("tests/output/models/test_model.pkl").exists()
    ```

- ✅ Write tests for end-to-end prediction pipeline:
  - ✅ Test the complete prediction workflow from data loading to prediction output
  - ✅ Verify that predictions have the expected format and columns
  - ✅ Test with different input data formats (CSV, Excel)
  - ✅ Test with different model types
  - ✅ Example test:
    ```python
    def test_end_to_end_prediction_pipeline():
        # Create orchestrator
        orchestrator = create_test_orchestrator()
        
        # Load a pre-trained model
        model = orchestrator.load_model("tests/data/test_model.pkl")
        
        # Run prediction pipeline
        predictions = orchestrator.predict(
            model=model,
            data_path="tests/data/sample_prediction_data.csv",
            output_path="tests/output/predictions.csv"
        )
        
        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) > 0
        
        # Verify predictions have expected columns
        expected_columns = ["category_name", "uniformat_code", "mcaa_system_category"]
        for col in expected_columns:
            assert col in predictions.columns
        
        # Verify output file was created
        assert Path("tests/output/predictions.csv").exists()
    ```

- ✅ Write tests for end-to-end evaluation pipeline:
  - ✅ Test the complete evaluation workflow from data loading to metrics calculation
  - ✅ Verify that evaluation metrics are calculated correctly
  - ✅ Test with different evaluation metrics
  - ✅ Test with different model types
  - ✅ Example test:
    ```python
    def test_end_to_end_evaluation_pipeline():
        # Create orchestrator
        orchestrator = create_test_orchestrator()
        
        # Load a pre-trained model
        model = orchestrator.load_model("tests/data/test_model.pkl")
        
        # Run evaluation pipeline
        results = orchestrator.evaluate(
            model=model,
            data_path="tests/data/sample_evaluation_data.csv",
            output_path="tests/output/evaluation_results.json"
        )
        
        # Verify results were generated
        assert results is not None
        assert "metrics" in results
        
        # Verify metrics were calculated
        assert "accuracy" in results["metrics"]
        assert "f1" in results["metrics"]
        assert "precision" in results["metrics"]
        assert "recall" in results["metrics"]
        
        # Verify output file was created
        assert Path("tests/output/evaluation_results.json").exists()
    ```

#### 2.2 Create Test Configurations (✅ COMPLETED)
- ✅ Create test configurations for different scenarios:
  - ✅ Create minimal configuration with only required parameters
  - ✅ Create comprehensive configuration with all possible parameters
  - ✅ Create configurations for different model types
  - ✅ Create configurations for different feature engineering approaches
  - ✅ Example minimal configuration:
    ```yaml
    # tests/configs/minimal_config.yml
    data:
      input_path: tests/data/sample_data.csv
      target_column: target
    
    features:
      text_columns: [description]
      numeric_columns: [service_life]
    
    model:
      type: random_forest
    ```
  
  - ✅ Example comprehensive configuration:
    ```yaml
    # tests/configs/comprehensive_config.yml
    data:
      input_path: tests/data/sample_data.csv
      target_column: target
      test_size: 0.3
      random_state: 42
      validation_size: 0.2
    
    features:
      text_columns: [description, manufacturer, model]
      numeric_columns: [service_life, installation_year]
      categorical_columns: [location, department]
      date_columns: [installation_date]
      transformations:
        - type: text_combiner
          columns: [description, manufacturer, model]
          output_column: combined_text
        - type: numeric_scaler
          columns: [service_life]
          method: standard
        - type: one_hot_encoder
          columns: [location, department]
    
    model:
      type: gradient_boosting
      params:
        n_estimators: 200
        max_depth: 10
        learning_rate: 0.1
      optimization:
        method: grid_search
        param_grid:
          n_estimators: [100, 200, 300]
          max_depth: [5, 10, 15]
        cv: 5
        scoring: f1_weighted
    
    evaluation:
      metrics: [accuracy, precision, recall, f1]
      cv: 5
    
    output:
      model_dir: tests/output/models
      results_dir: tests/output/results
      model_name: comprehensive_model
    ```

- ✅ Test with minimal configurations:
  - ✅ Verify that default values are used for missing parameters
  - ✅ Verify that the pipeline works with minimal input
  - ✅ Example test:
    ```python
    def test_pipeline_with_minimal_config():
        # Load minimal configuration
        config_path = "tests/configs/minimal_config.yml"
        
        # Create orchestrator with minimal configuration
        orchestrator = create_test_orchestrator_with_config(config_path)
        
        # Run training pipeline
        model, metrics = orchestrator.train_model(
            data_path="tests/data/sample_data.csv",
            output_dir="tests/output/models",
            model_name="minimal_model"
        )
        
        # Verify pipeline completed successfully
        assert model is not None
        assert metrics is not None
    ```

- ✅ Test with comprehensive configurations:
  - ✅ Verify that all parameters are used correctly
  - ✅ Verify that complex configurations work as expected
  - ✅ Example test:
    ```python
    def test_pipeline_with_comprehensive_config():
        # Load comprehensive configuration
        config_path = "tests/configs/comprehensive_config.yml"
        
        # Create orchestrator with comprehensive configuration
        orchestrator = create_test_orchestrator_with_config(config_path)
        
        # Run training pipeline
        model, metrics = orchestrator.train_model(
            data_path="tests/data/sample_data.csv",
            output_dir="tests/output/models",
            model_name="comprehensive_model"
        )
        
        # Verify pipeline completed successfully
        assert model is not None
        assert metrics is not None
        
        # Verify that optimization was performed
        assert "best_params" in orchestrator.context.get("optimization_results", {})
    ```

#### 2.3 Verify End-to-End Functionality (⏳ IN PROGRESS)
- ✅ Test with real data:
  - ✅ Create realistic test datasets with various characteristics
  - ✅ Include edge cases like missing values and outliers
  - ✅ Test with different data sizes (small, medium, large)
  - ✅ Example test data creation:
    ```python
    def create_realistic_test_data():
        """Create realistic test data for integration tests."""
        # Create a DataFrame with realistic equipment data
        data = pd.DataFrame({
            "equipment_tag": [f"EQ-{i:03d}" for i in range(100)],
            "manufacturer": np.random.choice(["Trane", "Carrier", "York", "Daikin"], 100),
            "model": [f"Model-{chr(65+i%26)}{i%100}" for i in range(100)],
            "description": [
                "Air Handling Unit with cooling coil",
                "Centrifugal Chiller for HVAC system",
                "Centrifugal Pump for chilled water",
                # ... more realistic descriptions
            ] * 25,
            "service_life": np.random.randint(10, 30, 100),
            "installation_year": np.random.randint(2000, 2023, 100),
            "location": np.random.choice(["Building A", "Building B", "Building C"], 100),
            "department": np.random.choice(["Facilities", "IT", "Production"], 100),
        })
        
        # Add some missing values
        for col in ["manufacturer", "model", "service_life"]:
            mask = np.random.choice([True, False], size=len(data), p=[0.05, 0.95])
            data.loc[mask, col] = np.nan
        
        # Add target columns for testing
        data["category_name"] = np.random.choice(["HVAC", "Plumbing", "Electrical"], 100)
        data["uniformat_code"] = np.random.choice(["D3010", "D2010", "D5010"], 100)
        data["mcaa_system_category"] = np.random.choice(["Mechanical", "Plumbing", "Electrical"], 100)
        
        # Save to CSV
        data.to_csv("tests/data/realistic_test_data.csv", index=False)
        
        return data
    ```

- ⏳ Verify outputs match expected results:
  - ⏳ Create expected output templates for comparison
  - ⏳ Verify that predictions match expected format and values
  - ⏳ Verify that evaluation metrics are within expected ranges
  - ⏳ Example verification:
    ```python
    def test_prediction_outputs_match_expected_format():
        # Run prediction pipeline
        predictions = run_test_prediction_pipeline()
        
        # Verify output format
        expected_columns = [
            "equipment_tag", "category_name", "uniformat_code",
            "mcaa_system_category", "Equipment_Type", "System_Subtype"
        ]
        for col in expected_columns:
            assert col in predictions.columns
        
        # Verify data types
        assert predictions["category_name"].dtype == "object"
        assert predictions["uniformat_code"].dtype == "object"
        
        # Verify value ranges
        assert set(predictions["category_name"].unique()).issubset(
            {"HVAC", "Plumbing", "Electrical", "Fire Protection", "Unknown"}
        )
    ```

- ⏳ Test error handling and recovery:
  - ⏳ Test with invalid input data
  - ⏳ Test with missing configuration files
  - ⏳ Test with incompatible model and data
  - ⏳ Verify appropriate error messages
  - ⏳ Example error handling test:
    ```python
    def test_pipeline_error_handling():
        # Create orchestrator
        orchestrator = create_test_orchestrator()
        
        # Test with nonexistent data file
        with pytest.raises(FileNotFoundError):
            orchestrator.train_model(
                data_path="nonexistent_file.csv",
                output_dir="tests/output/models"
            )
        
        # Test with invalid data (missing required columns)
        invalid_data = pd.DataFrame({"wrong_column": [1, 2, 3]})
        invalid_data.to_csv("tests/data/invalid_data.csv", index=False)
        
        with pytest.raises(ValueError) as excinfo:
            orchestrator.train_model(
                data_path="tests/data/invalid_data.csv",
                output_dir="tests/output/models"
            )
        assert "Missing required columns" in str(excinfo.value)
        
        # Verify context status reflects the error
        assert orchestrator.context.status == "error"
        assert "error_message" in orchestrator.context.data
    ```

**Files to Create/Modify:**
- `nexusml/tests/integration/test_training_pipeline.py` - End-to-end tests for training pipeline
- `nexusml/tests/integration/test_prediction_pipeline.py` - End-to-end tests for prediction pipeline
- `nexusml/tests/integration/test_evaluation_pipeline.py` - End-to-end tests for evaluation pipeline
- `nexusml/tests/integration/test_pipeline_configurations.py` - Tests for different configurations
- `nexusml/tests/integration/test_error_handling.py` - Tests for error handling and recovery
- `nexusml/tests/integration/conftest.py` - Common fixtures for integration tests
- `nexusml/tests/data/sample_training_data.csv` - Sample data for training tests
- `nexusml/tests/data/sample_prediction_data.csv` - Sample data for prediction tests
- `nexusml/tests/data/sample_evaluation_data.csv` - Sample data for evaluation tests
- `nexusml/tests/data/realistic_test_data.csv` - Realistic data for comprehensive testing
- `nexusml/tests/configs/minimal_config.yml` - Minimal configuration for testing
- `nexusml/tests/configs/comprehensive_config.yml` - Comprehensive configuration for testing
- `nexusml/tests/configs/feature_config.yml` - Feature engineering configuration for testing
- `nexusml/tests/configs/model_config.yml` - Model configuration for testing

### 3. Documentation (⏳ PENDING)

#### 3.1 Update Code Documentation (⏳ PENDING)
- ⏳ Add docstrings to all classes and methods:
  - ⏳ Use Google-style docstrings for consistency
  - ⏳ Include parameter descriptions with types
  - ⏳ Document return values and exceptions
  - ⏳ Add usage notes and warnings where appropriate
  - ⏳ Example docstring:
    ```python
    def train_model(
        self,
        data_path: str,
        feature_config_path: Optional[str] = None,
        test_size: float = 0.3,
        random_state: Optional[int] = None,
        output_dir: str = "outputs/models",
        model_name: str = "model",
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train a model using the specified data and configuration.
        
        This method loads data, performs feature engineering, builds and trains
        a model, evaluates its performance, and saves the trained model to disk.
        
        Args:
            data_path: Path to the input data file (CSV or Excel).
            feature_config_path: Path to the feature engineering configuration file.
                If None, uses default feature engineering.
            test_size: Proportion of the data to use for testing (0.0 to 1.0).
            random_state: Random seed for reproducibility. If None, uses a random seed.
            output_dir: Directory to save the trained model and related files.
            model_name: Name to use for the saved model files.
            
        Returns:
            A tuple containing:
                - The trained model object
                - A dictionary of evaluation metrics
                
        Raises:
            FileNotFoundError: If data_path or feature_config_path doesn't exist.
            ValueError: If test_size is not between 0 and 1.
            
        Example:
            >>> orchestrator = PipelineOrchestrator()
            >>> model, metrics = orchestrator.train_model(
            ...     data_path="data/equipment.csv",
            ...     feature_config_path="config/features.yml",
            ...     test_size=0.2,
            ...     output_dir="models",
            ...     model_name="equipment_classifier"
            ... )
            >>> print(f"Model accuracy: {metrics['accuracy']:.2f}")
        """
    ```

- ⏳ Include examples in docstrings:
  - ⏳ Add practical usage examples for each public method
  - ⏳ Show expected inputs and outputs
  - ⏳ Demonstrate common use cases
  - ⏳ Include edge cases and error handling examples
  - ⏳ Example:
    ```python
    def predict(
        self,
        model: Any,
        data: Union[pd.DataFrame, str],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model object.
            data: Input data as a DataFrame or path to a data file.
            output_path: Path to save prediction results. If None, results are not saved.
            
        Returns:
            DataFrame containing the original data with prediction columns added.
            
        Examples:
            # Predict using a DataFrame
            >>> df = pd.DataFrame({"description": ["Air handling unit", "Chiller"]})
            >>> predictions = orchestrator.predict(model, df)
            
            # Predict using a file path and save results
            >>> predictions = orchestrator.predict(
            ...     model,
            ...     "data/new_equipment.csv",
            ...     output_path="results/predictions.csv"
            ... )
            
            # Handle errors
            >>> try:
            ...     predictions = orchestrator.predict(model, "nonexistent.csv")
            ... except FileNotFoundError as e:
            ...     print(f"Error: {e}")
        """
    ```

- ⏳ Add type hints to all functions and methods:
  - ⏳ Use proper typing for parameters and return values
  - ⏳ Use Union, Optional, and other typing constructs as needed
  - ⏳ Include generics where appropriate
  - ⏳ Add TypeVar for generic types
  - ⏳ Example:
    ```python
    from typing import Dict, List, Optional, Tuple, TypeVar, Union, Any, Callable

    T = TypeVar('T')
    ModelType = TypeVar('ModelType')
    
    class ModelBuilder(Generic[ModelType]):
        def build_model(self, **kwargs: Any) -> ModelType:
            """Build a model with the specified parameters."""
            raise NotImplementedError
            
        def optimize_hyperparameters(
            self,
            model: ModelType,
            x_train: pd.DataFrame,
            y_train: Union[pd.Series, pd.DataFrame],
            param_grid: Optional[Dict[str, List[Any]]] = None,
            cv: int = 5,
            scoring: Union[str, Callable] = 'accuracy',
            **kwargs: Any
        ) -> ModelType:
            """Optimize model hyperparameters using cross-validation."""
            raise NotImplementedError
    ```

#### 3.2 Create Architecture Documentation (⏳ PENDING)
- ⏳ Document overall architecture:
  - ⏳ Create a high-level overview of the system architecture
  - ⏳ Explain the key components and their relationships
  - ⏳ Document the data flow through the system
  - ⏳ Explain the design principles and patterns used
  - ⏳ Example architecture overview:
    ```markdown
    # NexusML Architecture
    
    NexusML is designed as a modular, extensible framework for building and deploying
    machine learning pipelines. The architecture follows these key principles:
    
    - **Modularity**: Components are designed to be independent and reusable
    - **Extensibility**: New components can be added without modifying existing code
    - **Configurability**: Behavior can be controlled through configuration
    - **Testability**: Components can be tested in isolation
    
    ## Key Components
    
    1. **Configuration System**: Manages loading, validation, and access to configuration
    2. **Data Validation**: Validates input data against rules and schemas
    3. **Feature Engineering**: Transforms raw data into features for model training
    4. **Model Building**: Creates machine learning models based on configuration
    5. **Pipeline Orchestration**: Coordinates the execution of pipeline stages
    
    ## Data Flow
    
    1. Data is loaded from source (CSV, Excel, database)
    2. Data is validated against rules and schemas
    3. Features are engineered from raw data
    4. Data is split into training and testing sets
    5. Model is built and trained on training data
    6. Model is evaluated on testing data
    7. Model and metadata are saved to disk
    
    ## Extension Points
    
    NexusML provides several extension points for customization:
    
    - **Custom Validators**: Add new validation rules
    - **Custom Transformers**: Add new feature engineering transformations
    - **Custom Model Builders**: Add support for new model types
    - **Custom Pipeline Stages**: Add new stages to the pipeline
    - **Plugins**: Add complete new functionality through the plugin system
    ```

- ⏳ Create component diagrams:
  - ⏳ Use PlantUML or Mermaid to create diagrams
  - ⏳ Include class diagrams for key components
  - ⏳ Include sequence diagrams for key workflows
  - ⏳ Include component diagrams showing relationships
  - ⏳ Example class diagram:
    ```plantuml
    @startuml
    
    package "Configuration" {
      interface ConfigSection
      class ConfigurationManager
      class YamlDataConfig
      class YamlFeatureConfig
      class YamlModelConfig
      
      ConfigSection <|-- YamlDataConfig
      ConfigSection <|-- YamlFeatureConfig
      ConfigSection <|-- YamlModelConfig
      ConfigurationManager --> ConfigSection
    }
    
    package "Pipeline" {
      interface PipelineStage
      class BasePipeline
      class TrainingPipeline
      class PredictionPipeline
      class PipelineFactory
      class PipelineOrchestrator
      
      PipelineStage <|-- DataLoadingStage
      PipelineStage <|-- ValidationStage
      PipelineStage <|-- FeatureEngineeringStage
      
      BasePipeline <|-- TrainingPipeline
      BasePipeline <|-- PredictionPipeline
      
      PipelineFactory --> BasePipeline
      PipelineOrchestrator --> PipelineFactory
      BasePipeline o-- PipelineStage
    }
    
    @enduml
    ```
  
  - ⏳ Example sequence diagram:
    ```plantuml
    @startuml
    
    actor User
    participant PipelineOrchestrator
    participant TrainingPipeline
    participant DataLoadingStage
    participant FeatureEngineeringStage
    participant ModelBuildingStage
    participant ModelTrainingStage
    participant ModelEvaluationStage
    participant ModelSavingStage
    
    User -> PipelineOrchestrator: train_model(data_path, config_path)
    PipelineOrchestrator -> TrainingPipeline: create
    PipelineOrchestrator -> TrainingPipeline: run()
    
    TrainingPipeline -> DataLoadingStage: execute()
    DataLoadingStage --> TrainingPipeline: data
    
    TrainingPipeline -> FeatureEngineeringStage: execute(data)
    FeatureEngineeringStage --> TrainingPipeline: features
    
    TrainingPipeline -> ModelBuildingStage: execute(features)
    ModelBuildingStage --> TrainingPipeline: model
    
    TrainingPipeline -> ModelTrainingStage: execute(model, features)
    ModelTrainingStage --> TrainingPipeline: trained_model
    
    TrainingPipeline -> ModelEvaluationStage: execute(trained_model, features)
    ModelEvaluationStage --> TrainingPipeline: metrics
    
    TrainingPipeline -> ModelSavingStage: execute(trained_model, metrics)
    ModelSavingStage --> TrainingPipeline: model_path
    
    TrainingPipeline --> PipelineOrchestrator: result
    PipelineOrchestrator --> User: model, metrics
    
    @enduml
    ```

- ⏳ Document design decisions and trade-offs:
  - ⏳ Explain why certain design choices were made
  - ⏳ Document alternatives that were considered
  - ⏳ Explain trade-offs between different approaches
  - ⏳ Include performance considerations
  - ⏳ Example design decision documentation:
    ```markdown
    ## Design Decision: Configuration System
    
    ### Decision
    
    We chose to implement a configuration system based on YAML files with JSON Schema validation.
    
    ### Context
    
    The system needs to support complex configurations for different components,
    including data loading, feature engineering, model building, and evaluation.
    
    ### Alternatives Considered
    
    1. **JSON Configuration**: More structured but less readable
    2. **Python Configuration Files**: More flexible but less secure
    3. **Database Configuration**: More scalable but more complex
    
    ### Trade-offs
    
    - **YAML vs. JSON**: YAML is more readable and supports comments, but is less
      structured and can be error-prone. We mitigate this with JSON Schema validation.
      
    - **File-based vs. Database**: File-based configuration is simpler and works well
      with version control, but doesn't scale as well for large numbers of configurations.
      
    - **Validation vs. Flexibility**: Schema validation adds overhead but prevents
      many common errors and provides better error messages.
      
    ### Performance Considerations
    
    Configuration loading is cached to minimize performance impact. The validation
    overhead is only incurred once per configuration file.
    ```

#### 3.3 Create Usage Examples (⏳ PENDING)
- ⏳ Create example scripts for common tasks:
  - ⏳ Training a model with default settings
  - ⏳ Training a model with custom configuration
  - ⏳ Making predictions with a trained model
  - ⏳ Evaluating a model's performance
  - ⏳ Customizing the pipeline with new components
  - ⏳ Example script:
    ```python
    #!/usr/bin/env python
    """
    Example: Training a model with custom configuration
    
    This example demonstrates how to train a model using a custom configuration
    file for feature engineering and model parameters.
    """
    
    import logging
    from pathlib import Path
    
    from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("custom_training_example")
    
    # Create directories
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Train model with custom configuration
    model, metrics = orchestrator.train_model(
        data_path="data/equipment.csv",
        feature_config_path="config/custom_features.yml",
        model_config_path="config/custom_model.yml",
        test_size=0.2,
        random_state=42,
        output_dir=str(output_dir),
        model_name="custom_equipment_classifier",
    )
    
    # Display metrics
    logger.info("Model training completed with the following metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    # Make predictions with the trained model
    predictions = orchestrator.predict(
        model=model,
        data_path="data/new_equipment.csv",
        output_path="outputs/predictions.csv",
    )
    
    logger.info(f"Made predictions for {len(predictions)} items")
    logger.info(f"Predictions saved to: outputs/predictions.csv")
    ```

- ⏳ Add Jupyter notebooks with examples:
  - ⏳ Create interactive notebooks for exploring the library
  - ⏳ Include visualizations of results
  - ⏳ Provide step-by-step walkthroughs
  - ⏳ Include explanations of key concepts
  - ⏳ Example notebook structure:
    ```markdown
    # NexusML: Training and Evaluating a Model
    
    This notebook demonstrates how to train and evaluate a model using NexusML.
    
    ## 1. Setup
    
    First, let's import the necessary libraries and set up logging.
    
    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import logging
    
    from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    ```
    
    ## 2. Load and Explore Data
    
    Let's load the sample data and explore it.
    
    ```python
    # Load data
    data = pd.read_csv("data/equipment.csv")
    
    # Display basic information
    print(f"Data shape: {data.shape}")
    data.head()
    ```
    
    ## 3. Train a Model
    
    Now, let's train a model using the default configuration.
    
    ```python
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Train model
    model, metrics = orchestrator.train_model(
        data_path="data/equipment.csv",
        test_size=0.2,
        random_state=42,
        output_dir="outputs/models",
        model_name="equipment_classifier",
    )
    
    # Display metrics
    print("Model metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    ```
    
    ## 4. Visualize Results
    
    Let's visualize the model's performance.
    
    ```python
    # Create a bar chart of metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Model Performance Metrics")
    plt.ylim(0, 1)
    plt.show()
    ```
    ```

- ⏳ Create step-by-step tutorials:
  - ⏳ Provide detailed walkthroughs for common tasks
  - ⏳ Include screenshots and diagrams
  - ⏳ Explain concepts along with code
  - ⏳ Include troubleshooting tips
  - ⏳ Example tutorial:
    ```markdown
    # Tutorial: Building a Custom Pipeline
    
    This tutorial walks through the process of building a custom pipeline
    with NexusML, from data preparation to model deployment.
    
    ## Prerequisites
    
    - Python 3.8 or higher
    - NexusML installed (`pip install nexusml`)
    - Basic understanding of machine learning concepts
    
    ## Step 1: Prepare Your Data
    
    First, you need to prepare your data in a format that NexusML can use.
    NexusML supports CSV and Excel files with the following structure:
    
    - One row per sample
    - Columns for features and target variables
    - No missing values in critical columns
    
    Example:
    
    | equipment_tag | description                  | service_life | category_name |
    |---------------|------------------------------|--------------|---------------|
    | AHU-01        | Air Handling Unit with coil  | 20           | HVAC          |
    | CHW-01        | Centrifugal Chiller          | 25           | HVAC          |
    | P-01          | Centrifugal Pump             | 15           | Plumbing      |
    
    Save this data as `equipment.csv` in your project directory.
    
    ## Step 2: Create Configuration Files
    
    Next, create configuration files for your pipeline. Let's start with
    a feature engineering configuration:
    
    ```yaml
    # features.yml
    text_columns:
      - description
    
    numeric_columns:
      - service_life
    
    transformations:
      - type: text_normalizer
        columns: [description]
        lowercase: true
        remove_punctuation: true
        
      - type: numeric_scaler
        columns: [service_life]
        method: standard
    ```
    
    Now, create a model configuration:
    
    ```yaml
    # model.yml
    type: random_forest
    
    params:
      n_estimators: 100
      max_depth: 10
      random_state: 42
    
    optimization:
      method: grid_search
      param_grid:
        n_estimators: [50, 100, 200]
        max_depth: [5, 10, 15]
      cv: 5
      scoring: f1_weighted
    ```
    
    ## Step 3: Build Your Pipeline
    
    Create a Python script to build and run your pipeline:
    
    ```python
    #!/usr/bin/env python
    """
    Custom Pipeline Example
    """
    
    import logging
    from pathlib import Path
    
    from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("custom_pipeline")
    
    # Create directories
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    # Train model
    model, metrics = orchestrator.train_model(
        data_path="equipment.csv",
        feature_config_path="features.yml",
        model_config_path="model.yml",
        test_size=0.2,
        output_dir=str(output_dir),
        model_name="equipment_classifier",
    )
    
    # Display metrics
    logger.info("Model training completed with the following metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    ```
    
    ## Step 4: Run Your Pipeline
    
    Run your script:
    
    ```bash
    python custom_pipeline.py
    ```
    
    You should see output showing the progress of your pipeline and the final metrics.
    
    ## Step 5: Make Predictions
    
    Now, let's use the trained model to make predictions:
    
    ```python
    # Make predictions
    predictions = orchestrator.predict(
        model=model,
        data_path="new_equipment.csv",
        output_path="outputs/predictions.csv",
    )
    
    logger.info(f"Made predictions for {len(predictions)} items")
    logger.info(f"Predictions saved to: outputs/predictions.csv")
    ```
    
    ## Troubleshooting
    
    - **FileNotFoundError**: Make sure all file paths are correct
    - **ValueError: Missing required columns**: Check that your data has all required columns
    - **ImportError**: Make sure NexusML is installed correctly
    
    ## Next Steps
    
    - Try different model types (gradient_boosting, ensemble)
    - Customize feature engineering with different transformations
    - Implement custom validation rules
    ```

**Files to Create/Modify:**
- Update all source files with improved docstrings:
  - `nexusml/config/manager.py`
  - `nexusml/core/validation/interfaces.py`
  - `nexusml/core/feature_engineering/interfaces.py`
  - `nexusml/core/model_building/interfaces.py`
  - `nexusml/core/pipeline/stages/interfaces.py`
  - `nexusml/core/pipeline/factory.py`
  - `nexusml/core/pipeline/orchestrator.py`
  - All other implementation files

- Architecture documentation:
  - `docs/architecture/overview.md` - Overall architecture description
  - `docs/architecture/configuration.md` - Configuration system design
  - `docs/architecture/validation.md` - Validation system design
  - `docs/architecture/feature_engineering.md` - Feature engineering design
  - `docs/architecture/model_building.md` - Model building design
  - `docs/architecture/pipeline.md` - Pipeline design
  - `docs/architecture/diagrams/` - Directory for architecture diagrams
  - `docs/architecture/design_decisions.md` - Design decisions and trade-offs

- Component documentation:
  - `docs/components/configuration.md` - Configuration components
  - `docs/components/validation.md` - Validation components
  - `docs/components/feature_engineering.md` - Feature engineering components
  - `docs/components/model_building.md` - Model building components
  - `docs/components/pipeline.md` - Pipeline components
  - `docs/components/di.md` - Dependency injection system

- Example scripts:
  - `docs/examples/basic_training.py` - Basic model training
  - `docs/examples/custom_training.py` - Training with custom configuration
  - `docs/examples/prediction.py` - Making predictions
  - `docs/examples/evaluation.py` - Evaluating model performance
  - `docs/examples/custom_components.py` - Creating custom components
  - `docs/examples/pipeline_customization.py` - Customizing the pipeline

- Tutorials:
  - `docs/tutorials/getting_started.md` - Getting started guide
  - `docs/tutorials/data_preparation.md` - Data preparation guide
  - `docs/tutorials/feature_engineering.md` - Feature engineering guide
  - `docs/tutorials/model_building.md` - Model building guide
  - `docs/tutorials/pipeline_customization.md` - Pipeline customization guide
  - `docs/tutorials/troubleshooting.md` - Troubleshooting guide

- Jupyter notebooks:
  - `notebooks/01_introduction.ipynb` - Introduction to NexusML
  - `notebooks/02_data_exploration.ipynb` - Data exploration with NexusML
  - `notebooks/03_feature_engineering.ipynb` - Feature engineering with NexusML
  - `notebooks/04_model_training.ipynb` - Model training with NexusML
  - `notebooks/05_model_evaluation.ipynb` - Model evaluation with NexusML
  - `notebooks/06_prediction.ipynb` - Making predictions with NexusML
  - `notebooks/07_custom_components.ipynb` - Creating custom components

## Implementation Timeline

### Phase 1: Configuration Centralization (✅ COMPLETED)
- ✅ Enhance Config Module and Create Configuration Interfaces
- ✅ Update Path Management and Integration Testing
- ✅ Add Configuration Validation and Model Card Support

### Phase 2: Core Component Refactoring (✅ COMPLETED - 3 weeks)
- ✅ Week 1: Type Safety Improvements and Data Validation (Completed March 8, 2025)
- ✅ Week 2: Feature Engineering (Completed March 8, 2025)
- ✅ Week 3: Model Building and Training (Completed March 8, 2025)

### Phase 3: Pipeline Orchestration (✅ COMPLETED)
- ✅ Week 1: Pipeline Components and Dependency Injection (Completed March 9, 2025)
  - ✅ Pipeline Components: Interfaces and base implementations
  - ✅ Pipeline Components: Concrete implementations
  - ✅ Pipeline Components: Testing and verification
  - ✅ Dependency Injection: Register components
- ✅ Week 2: Pipeline Factory and Integration (Completed March 9, 2025)
  - ✅ Pipeline Factory: Implementation
  - ✅ Pipeline Types: Base, Training, Prediction, Evaluation
  - ✅ Extension Points: Registry and Plugin System
  - ✅ Unit Testing: All components tested and passing
  - ⚠️ Integration Testing: Basic verification complete, example scripts need updating

### Phase 4: Testing and Documentation (1 week) (⏳ IN PROGRESS)

#### 1.1 Complete Unit Tests (Day 1)
- ✅ Create test fixtures for common test scenarios
- ✅ Write tests for configuration components
- ✅ Write tests for validation components
- ✅ Write tests for feature engineering components
- ✅ Write tests for model building components
- ✅ Write tests for model training components
- ✅ Write tests for pipeline components
- ✅ Ensure test coverage meets 80% target

#### 1.2 Fix Example Scripts (Day 2) (✅ IN PROGRESS)
- ✅ Update `pipeline_orchestrator_example.py` to work with the new registry:
  - ✅ Update component registration to use the new registry pattern
  - ✅ Ensure proper dependency resolution for all components
  - ✅ Add error handling for component resolution failures
  - ✅ Verify all pipeline stages are properly connected
- ⏳ Fix feature engineering issues in `prediction_pipeline_example.py`:
  - ⏳ Fix handling of mixed data types (text and numeric)
  - ⏳ Update feature engineering stage to properly handle missing columns
  - ⏳ Ensure proper transformation of input data
  - ⏳ Add validation for input data format
  - ⏳ Fix MultiOutputClassifier handling in prediction stage

#### 1.3 Create Integration Tests (Day 3) (✅ IN PROGRESS)
- ✅ Create end-to-end test for training pipeline:
  - ✅ Test with minimal configuration (created `minimal_config.yml`)
  - ✅ Test with comprehensive configuration (created `comprehensive_config.yml`)
  - ✅ Verify model outputs match expected results
- ✅ Create end-to-end test for prediction pipeline:
  - ✅ Test with various input data formats (created `sample_prediction_data.csv`)
  - ✅ Verify prediction outputs match expected format
  - ✅ Test error handling and recovery
- ✅ Create end-to-end test for evaluation pipeline:
  - ✅ Test with different evaluation metrics
  - ✅ Verify evaluation results match expected values
  - ✅ Test with different model types

#### 1.4 Verification Steps (Day 4) (✅ IN PROGRESS)
- ✅ Create verification script to test all components:
  - ✅ Test component resolution from DI container
  - ✅ Test pipeline factory with different configurations
  - ✅ Test orchestrator with different pipeline types
  - ✅ Verify proper error handling and logging
- ⏳ Run verification script and fix any issues:
  - ⏳ Address any component resolution failures
  - ⏳ Fix any pipeline configuration issues
  - ⏳ Ensure proper error messages for invalid inputs
  - ⏳ Verify all components work together correctly

#### 1.5 Documentation and Examples (Day 5) (⏳ PENDING)
- ⏳ Update code documentation with comprehensive docstrings
- ⏳ Create architecture documentation:
  - ⏳ Document overall architecture
  - ⏳ Create component diagrams
  - ⏳ Document design decisions and trade-offs
- ⏳ Create usage examples and tutorials:
  - ⏳ Create example scripts for common tasks
  - ⏳ Add Jupyter notebooks with examples
  - ⏳ Create step-by-step tutorials

**Verification Test Script:**
```python
#!/usr/bin/env python
"""
NexusML Refactoring Verification Script

This script verifies that all components of the refactored NexusML library
work correctly together. It tests component resolution, pipeline creation,
and end-to-end functionality.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from nexusml.core.di.container import DIContainer
from nexusml.core.di.registration import register_all_components
from nexusml.core.pipeline.factory import PipelineFactory
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.config.manager import ConfigurationManager

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("verification_script")

def verify_component_resolution(logger):
    """Verify that all components can be resolved from the DI container."""
    logger.info("Verifying component resolution...")
    
    # Create a DI container
    container = DIContainer()
    
    # Register all components
    register_all_components(container)
    
    # Try to resolve key components
    components_to_verify = [
        "ConfigurationManager",
        "ConfigurableDataLoadingStage",
        "ConfigDrivenValidationStage",
        "SimpleFeatureEngineeringStage",
        "RandomSplittingStage",
        "ConfigDrivenModelBuildingStage",
        "StandardModelTrainingStage",
        "ClassificationEvaluationStage",
        "ModelCardSavingStage",
        "StandardPredictionStage"
    ]
    
    success = True
    for component_name in components_to_verify:
        try:
            component = container.resolve(eval(component_name))
            logger.info(f"✅ Successfully resolved {component_name}")
        except Exception as e:
            logger.error(f"❌ Failed to resolve {component_name}: {e}")
            success = False
    
    return success

def verify_pipeline_factory(logger):
    """Verify that the pipeline factory can create different pipeline types."""
    logger.info("Verifying pipeline factory...")
    
    # Create a component registry
    registry = ComponentRegistry()
    
    # Create a DI container
    container = DIContainer()
    
    # Register all components
    register_all_components(container)
    
    # Create a pipeline factory
    factory = PipelineFactory(registry, container)
    
    # Try to create different pipeline types
    pipeline_types = ["training", "prediction", "evaluation"]
    
    success = True
    for pipeline_type in pipeline_types:
        try:
            pipeline = factory.create_pipeline(pipeline_type)
            logger.info(f"✅ Successfully created {pipeline_type} pipeline")
        except Exception as e:
            logger.error(f"❌ Failed to create {pipeline_type} pipeline: {e}")
            success = False
    
    return success

def verify_end_to_end(logger):
    """Verify end-to-end functionality with a simple example."""
    logger.info("Verifying end-to-end functionality...")
    
    # Import the example script functions
    try:
        from examples.pipeline_orchestrator_example import create_orchestrator, train_model_example
        
        # Create orchestrator
        orchestrator = create_orchestrator()
        
        # Run a simple training example
        model = train_model_example(orchestrator, logger)
        
        if model:
            logger.info("✅ End-to-end verification successful")
            return True
        else:
            logger.error("❌ End-to-end verification failed: model training failed")
            return False
    except Exception as e:
        logger.error(f"❌ End-to-end verification failed: {e}")
        return False

def main():
    """Main function to run the verification script."""
    logger = setup_logging()
    logger.info("Starting NexusML Refactoring Verification")
    
    # Verify component resolution
    component_resolution_success = verify_component_resolution(logger)
    
    # Verify pipeline factory
    pipeline_factory_success = verify_pipeline_factory(logger)
    
    # Verify end-to-end functionality
    end_to_end_success = verify_end_to_end(logger)
    
    # Overall verification result
    if component_resolution_success and pipeline_factory_success and end_to_end_success:
        logger.info("✅ All verification tests passed!")
    else:
        logger.error("❌ Some verification tests failed")
        if not component_resolution_success:
            logger.error("  - Component resolution verification failed")
        if not pipeline_factory_success:
            logger.error("  - Pipeline factory verification failed")
        if not end_to_end_success:
            logger.error("  - End-to-end verification failed")
    
    logger.info("NexusML Refactoring Verification completed")

if __name__ == "__main__":
    main()
```

**Next Steps:**
1. ✅ Create the verification script in `nexusml/tests/verification_script.py` (COMPLETED)
2. ✅ Create test configurations and sample data (COMPLETED)
   - Created `minimal_config.yml`, `comprehensive_config.yml`, `feature_config.yml`, and `model_config.yml`
   - Created `sample_data.csv` and `sample_prediction_data.csv`
3. ⏳ Fix the issues in `prediction_pipeline_example.py` (IN PROGRESS)
   - Focus on feature engineering issues with mixed data types
   - Fix MultiOutputClassifier handling in prediction stage
4. ⏳ Run the verification script to ensure all components work together correctly
   - Execute: `python -m nexusml.tests.verification_script`
   - Fix any issues identified during verification
5. ⏳ Update documentation with the results of the verification
   - Document any workarounds or solutions to common issues
6. ⏳ Create comprehensive examples and tutorials for users
   - Focus on common use cases and workflows

## Migration Strategy

To minimize disruption to ongoing development, we'll use the following migration strategy:

1. **Parallel Implementation**: Implement new components alongside existing ones
2. **Adapter Pattern**: Create adapters to allow new components to work with existing code
3. **Feature Flags**: Use feature flags to gradually enable new components
4. **Incremental Adoption**: Allow teams to adopt new components at their own pace
5. **Deprecation Warnings**: Add deprecation warnings to old components
6. **Documentation**: Provide clear migration guides for developers

This approach allows for a smooth transition while ensuring that existing functionality continues to work throughout the refactoring process.

The ultimate goal of this refactoring work is to get the production model card system up and running, which will enable:

Standardized model training with proper validation and evaluation
Consistent model metadata and documentation through model cards
Reproducible model building and deployment processes
Better tracking and versioning of models in production
Improved model governance and compliance
