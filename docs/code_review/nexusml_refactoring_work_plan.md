# NexusML Refactoring Work Plan

## Executive Summary

This document outlines a comprehensive plan to refactor the NexusML pipeline system to adhere to SOLID principles and address several architectural issues. The refactoring aims to improve maintainability, extensibility, and reliability of the codebase.

## Current Issues and Motivations

### 1. Configuration Inconsistency

**Issue:** Different parts of the system use different configuration files or hardcoded values.

**Examples:**
- The validation function in `train_model_pipeline.py` has a hardcoded list of required columns
- The feature engineering in `train_model_pipeline_v2.py` uses a simplified implementation with hardcoded columns
- The `validate_data_from_config` function uses `production_data_config.yml` while other parts might use `feature_config.yml`

**Impact:**
- Inconsistent behavior between validation and feature engineering
- Misleading warnings about missing columns that aren't actually required
- Difficult to maintain as changes need to be made in multiple places

### 2. Validation-Feature Engineering Mismatch

**Issue:** Validation checks for columns that feature engineering doesn't use or vice versa.

**Examples:**
- Validation checks for `building_name`, `initial_cost`, and `condition_score` columns
- Feature engineering in `SimpleFeatureEngineer` only uses `description` and adds `combined_text` and `service_life`
- The training data preparation is hardcoded to use specific columns like `combined_text` and `service_life`

**Impact:**
- Confusing warnings that don't match actual requirements
- Potential for silent failures if required columns are missing
- Difficult to understand what columns are actually needed

### 3. Hardcoded Implementation

**Issue:** Many components have hardcoded logic instead of using configuration.

**Examples:**
- `SimpleFeatureEngineer` in `train_model_pipeline_v2.py` uses hardcoded columns and transformations
- The training data preparation in `train_model_pipeline.py` uses hardcoded column names
- The validation function has hardcoded lists of required and critical columns

**Impact:**
- Difficult to extend or modify behavior without changing code
- Changes to data structure require code changes in multiple places
- Lack of flexibility for different use cases

### 4. Lack of Dependency Injection

**Issue:** Components have hardcoded dependencies instead of using dependency injection.

**Examples:**
- The `validate_data` function directly imports and uses other modules
- The `train_model` function creates instances of classes directly
- The `SimpleFeatureEngineer` has hardcoded dependencies on logger

**Impact:**
- Difficult to test components in isolation
- Tight coupling between components
- Difficult to substitute implementations for different scenarios

### 5. Mixed Responsibilities

**Issue:** Classes and functions often handle multiple concerns.

**Examples:**
- The `train_model_pipeline.py` script handles validation, feature engineering, model training, and evaluation
- The `EquipmentClassifier` class handles model training, prediction, and attribute management
- The `validate_data` function handles both file existence checks and column validation

**Impact:**
- Code is difficult to understand and maintain
- Changes to one aspect often affect others
- Testing is more complex and less effective

## Proposed Work Plan

### Phase 1: Configuration Centralization

1. **Enhance Config Module**
   - Create a `ConfigurationManager` class that loads and manages all configuration
   - Implement methods to get specific configuration sections (data, features, etc.)
   - Add validation for configuration files

2. **Create Configuration Interfaces**
   - Define interfaces for different configuration types (`DataConfig`, `FeatureConfig`, etc.)
   - Implement concrete classes for each configuration type
   - Ensure backward compatibility with existing code

3. **Update Path Management**
   - Extend the existing path utilities to handle all file paths consistently
   - Create a `PathResolver` class to handle path resolution across different environments

### Phase 2: Core Component Refactoring

1. **Data Validation**
   - Create a `DataValidator` interface with validate method
   - Implement `ConfigDrivenValidator` that uses configuration for validation rules
   - Move validation logic out of pipeline scripts into dedicated classes

2. **Feature Engineering**
   - Create a proper `FeatureEngineer` interface
   - Implement `ConfigDrivenFeatureEngineer` that uses configuration for transformations
   - Split complex transformations into smaller, focused transformer classes

3. **Model Building and Training**
   - Create `ModelBuilder` and `ModelTrainer` interfaces
   - Implement concrete classes that follow Single Responsibility Principle
   - Use dependency injection for components like feature engineers

### Phase 3: Pipeline Orchestration

1. **Pipeline Components**
   - Define clear interfaces for each pipeline stage
   - Implement concrete classes for each stage
   - Ensure each class has a single responsibility

2. **Dependency Injection**
   - Create a proper DI container for the entire system
   - Register all components with the container
   - Use constructor injection for dependencies

3. **Pipeline Factory**
   - Create a `PipelineFactory` that builds pipelines from configuration
   - Support different pipeline types (training, prediction, etc.)
   - Allow extension with new components without modifying existing code

### Phase 4: Testing and Documentation

1. **Unit Tests**
   - Create unit tests for each component
   - Use mock objects for dependencies
   - Ensure high test coverage

2. **Integration Tests**
   - Create integration tests for the entire pipeline
   - Test with different configuration files
   - Verify end-to-end functionality

3. **Documentation**
   - Update code documentation with clear explanations
   - Create architecture documentation
   - Provide examples for common use cases

## SOLID Principles Application

### Single Responsibility Principle (SRP)

**Current Violation:** The `train_model_pipeline.py` script handles multiple responsibilities including validation, feature engineering, model training, and evaluation.

**Proposed Solution:** Split into separate classes with focused responsibilities:
- `DataValidator` for validation
- `FeatureEngineer` for feature engineering
- `ModelTrainer` for model training
- `ModelEvaluator` for evaluation

### Open/Closed Principle (OCP)

**Current Violation:** Adding new validation rules or feature transformations requires modifying existing code.

**Proposed Solution:** 
- Create interfaces for validation rules and feature transformations
- Allow new rules and transformations to be added without modifying existing code
- Use configuration to define behavior instead of hardcoding

### Liskov Substitution Principle (LSP)

**Current Violation:** The system doesn't use interfaces consistently, making it difficult to substitute implementations.

**Proposed Solution:**
- Define clear interfaces for all components
- Ensure all implementations adhere to interface contracts
- Use dependency injection to allow substitution of implementations

### Interface Segregation Principle (ISP)

**Current Violation:** Some classes have large, monolithic interfaces that force clients to depend on methods they don't use.

**Proposed Solution:**
- Create smaller, focused interfaces
- Split large classes into smaller ones with specific responsibilities
- Use composition over inheritance

### Dependency Inversion Principle (DIP)

**Current Violation:** High-level modules depend directly on low-level modules instead of abstractions.

**Proposed Solution:**
- Use dependency injection for all dependencies
- Define abstractions (interfaces) for all components
- Ensure high-level modules depend only on abstractions

## Implementation Examples

### ConfigurationManager

```python
# nexusml/config/manager.py
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Generic
import yaml

from nexusml.config import get_project_root

T = TypeVar('T')

class ConfigSection(Generic[T]):
    """Base class for configuration sections."""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.data.get(key, default)

class DataConfig(ConfigSection):
    """Configuration for data handling."""
    
    @property
    def required_columns(self) -> list:
        """Get the required columns."""
        return [col['name'] for col in self.data.get('required_columns', [])]
    
    @property
    def source_columns(self) -> list:
        """Get only the source columns (not derived during feature engineering)."""
        return [
            col['name'] for col in self.data.get('required_columns', [])
            if not col['name'].startswith(('Equipment_', 'Uniformat_', 'System_', 'combined_', 'service_life'))
        ]

class FeatureConfig(ConfigSection):
    """Configuration for feature engineering."""
    
    @property
    def text_combinations(self) -> list:
        """Get text combination configurations."""
        return self.data.get('text_combinations', [])
    
    @property
    def numeric_columns(self) -> list:
        """Get numeric column configurations."""
        return self.data.get('numeric_columns', [])
    
    @property
    def hierarchies(self) -> list:
        """Get hierarchy configurations."""
        return self.data.get('hierarchies', [])

class ConfigurationManager:
    """Manager for all configuration files."""
    
    def __init__(self):
        self.root = get_project_root()
        self.config_dir = self.root / "config"
        self.configs = {}
    
    def load_config(self, name: str) -> Dict[str, Any]:
        """Load a configuration file."""
        if name in self.configs:
            return self.configs[name]
        
        path = self.config_dir / f"{name}.yml"
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.configs[name] = config
        return config
    
    def get_config_section(self, name: str, section_class: Type[T]) -> T:
        """Get a typed configuration section."""
        config = self.load_config(name)
        return section_class(config)
    
    def get_data_config(self, name: str = "production_data_config") -> DataConfig:
        """Get the data configuration."""
        return self.get_config_section(name, DataConfig)
    
    def get_feature_config(self, name: str = "feature_config") -> FeatureConfig:
        """Get the feature configuration."""
        return self.get_config_section(name, FeatureConfig)
```

### DataValidator

```python
# nexusml/core/validation.py
from typing import Dict, List, Optional
import pandas as pd

from nexusml.config.manager import ConfigurationManager, DataConfig

class DataValidator:
    """Validates data against configuration requirements."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
    
    def validate(self, df: pd.DataFrame, config_name: str = "production_data_config") -> Dict:
        """
        Validate a DataFrame against configuration requirements.
        
        Args:
            df: DataFrame to validate
            config_name: Name of the configuration to use
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Get the data configuration
            data_config = self.config_manager.get_data_config(config_name)
            
            # Check required columns
            required_columns = data_config.source_columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return {
                    "valid": False,
                    "issues": [f"Missing required columns: {', '.join(missing_columns)}"],
                }
            
            # Check for missing values in critical columns
            critical_columns = ["equipment_tag", "category_name", "mcaa_system_category"]
            missing_values = {}
            
            for col in critical_columns:
                if col in df.columns:
                    missing_count = df[col].isna().sum()
                    if missing_count > 0:
                        missing_values[col] = missing_count
            
            if missing_values:
                issues = [
                    f"Missing values in {col}: {count}"
                    for col, count in missing_values.items()
                ]
                return {"valid": False, "issues": issues}
            
            # All checks passed
            return {"valid": True, "issues": []}
            
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Unexpected error during validation: {str(e)}"],
            }
```

### ConfigDrivenFeatureEngineer

```python
# nexusml/core/feature_engineering.py
from typing import Dict, List, Optional
import pandas as pd

from nexusml.config.manager import ConfigurationManager, FeatureConfig

class ConfigDrivenFeatureEngineer:
    """Feature engineer that uses configuration for transformations."""
    
    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        self.config_manager = config_manager or ConfigurationManager()
    
    def transform(self, df: pd.DataFrame, config_name: str = "production_data_config") -> pd.DataFrame:
        """
        Transform a DataFrame using configuration.
        
        Args:
            df: DataFrame to transform
            config_name: Name of the configuration to use
            
        Returns:
            Transformed DataFrame
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Get the feature configuration
        feature_config = self.config_manager.get_feature_config(config_name)
        
        # Apply text combinations
        result = self._apply_text_combinations(result, feature_config.text_combinations)
        
        # Apply numeric column transformations
        result = self._apply_numeric_transformations(result, feature_config.numeric_columns)
        
        # Apply hierarchies
        result = self._apply_hierarchies(result, feature_config.hierarchies)
        
        return result
    
    def _apply_text_combinations(self, df: pd.DataFrame, combinations: List[Dict]) -> pd.DataFrame:
        """Apply text combinations to create new columns."""
        for combo in combinations:
            name = combo.get('name')
            columns = combo.get('columns', [])
            separator = combo.get('separator', ' ')
            
            # Check if all required columns exist
            available_columns = [col for col in columns if col in df.columns]
            
            if available_columns:
                # Combine available columns
                df[name] = df[available_columns].fillna('').agg(separator.join, axis=1)
            else:
                # Create empty column if no source columns are available
                df[name] = "Unknown"
        
        return df
    
    def _apply_numeric_transformations(self, df: pd.DataFrame, numeric_configs: List[Dict]) -> pd.DataFrame:
        """Apply transformations to numeric columns."""
        for config in numeric_configs:
            name = config.get('name')
            new_name = config.get('new_name', name)
            fill_value = config.get('fill_value', 0)
            dtype = config.get('dtype', 'float')
            
            if name in df.columns:
                # Copy and convert the column
                df[new_name] = df[name].fillna(fill_value)
                if dtype == 'float':
                    df[new_name] = pd.to_numeric(df[new_name], errors='coerce').fillna(fill_value)
            else:
                # Create column with default value
                df[new_name] = fill_value
        
        return df
    
    def _apply_hierarchies(self, df: pd.DataFrame, hierarchies: List[Dict]) -> pd.DataFrame:
        """Apply hierarchical combinations to create new columns."""
        for hierarchy in hierarchies:
            new_col = hierarchy.get('new_col')
            parents = hierarchy.get('parents', [])
            separator = hierarchy.get('separator', '-')
            
            # Check if all parent columns exist
            available_parents = [col for col in parents if col in df.columns]
            
            if available_parents:
                # Combine available parent columns
                df[new_col] = df[available_parents].fillna('').agg(separator.join, axis=1)
            else:
                # Create empty column if no parent columns are available
                df[new_col] = "Unknown"
        
        return df
```

## Benefits of Refactoring

1. **Improved Maintainability**
   - Clear separation of concerns
   - Focused classes with single responsibilities
   - Consistent configuration management

2. **Enhanced Extensibility**
   - Easy to add new validation rules
   - Easy to add new feature transformations
   - Easy to add new pipeline components

3. **Better Testability**
   - Components can be tested in isolation
   - Dependencies can be mocked
   - Clear interfaces make testing easier

4. **Reduced Duplication**
   - Centralized configuration management
   - Reusable components
   - Consistent approach to common tasks

5. **Clearer Intent**
   - Code structure reflects business domain
   - Interfaces document expected behavior
   - Configuration separates policy from mechanism

## Conclusion

This refactoring work plan addresses the current issues in the NexusML pipeline system and provides a path to a more maintainable, extensible, and reliable codebase. By following SOLID principles and implementing the proposed changes, we can create a system that is easier to understand, modify, and extend.

The work is divided into phases to allow for incremental improvements and to minimize disruption to ongoing development. Each phase builds on the previous one and moves the system closer to the desired architecture.
