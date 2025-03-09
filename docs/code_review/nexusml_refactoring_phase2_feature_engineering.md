# NexusML Refactoring: Phase 2 - Feature Engineering Component

## Overview

This document summarizes the implementation of the Feature Engineering component for Phase 2 of the NexusML refactoring. The Feature Engineering component provides a comprehensive system for transforming data in the NexusML suite, following SOLID principles and improving type safety.

## Components Implemented

### 1. Feature Engineering Interfaces

- `FeatureTransformer`: Interface for feature transformers
- `ColumnTransformer`: Interface for column-specific transformers
- `ConfigurableTransformer`: Interface for configurable transformers
- `TransformerRegistry`: Interface for transformer registries
- `FeatureEngineer`: Interface for feature engineers
- `ConfigDrivenFeatureEngineer`: Interface for configuration-driven feature engineers

### 2. Base Implementations

- `BaseFeatureTransformer`: Base implementation of the FeatureTransformer interface
- `BaseColumnTransformer`: Base implementation of the ColumnTransformer interface
- `BaseConfigurableTransformer`: Base implementation of the ConfigurableTransformer interface
- `BaseFeatureEngineer`: Base implementation of the FeatureEngineer interface
- `BaseConfigDrivenFeatureEngineer`: Base implementation of the ConfigDrivenFeatureEngineer interface

### 3. Transformer Registry

- `DefaultTransformerRegistry`: Default implementation of the TransformerRegistry interface
- `register_transformer`: Function to register a transformer with the default registry
- `get_transformer_class`: Function to get a transformer class from the default registry
- `create_transformer`: Function to create a transformer instance from the default registry
- `get_registered_transformers`: Function to get all registered transformers from the default registry

### 4. Text Transformers

- `TextCombiner`: Transformer for combining multiple text columns into one
- `TextNormalizer`: Transformer for normalizing text in a column
- `TextTokenizer`: Transformer for tokenizing text in a column

### 5. Numeric Transformers

- `NumericCleaner`: Transformer for cleaning and transforming numeric columns
- `NumericScaler`: Transformer for scaling numeric columns
- `MissingValueHandler`: Transformer for handling missing values in numeric columns
- `OutlierDetector`: Transformer for detecting and handling outliers in numeric columns

### 6. Hierarchical Transformers

- `HierarchyBuilder`: Transformer for creating hierarchical category columns
- `HierarchyExpander`: Transformer for expanding a hierarchical column into its component parts
- `HierarchyFilter`: Transformer for filtering rows based on hierarchical column values

### 7. Categorical Transformers

- `ColumnMapper`: Transformer for mapping source columns to target columns
- `OneHotEncoder`: Transformer for one-hot encoding categorical columns
- `LabelEncoder`: Transformer for label encoding categorical columns
- `ClassificationSystemMapper`: Transformer for mapping equipment categories to classification system IDs
- `KeywordClassificationMapper`: Transformer for mapping equipment descriptions to classification system IDs

### 8. Configuration-Driven Feature Engineer

- `ConfigDrivenFeatureEngineer`: Feature engineer that creates and applies transformers based on a configuration

### 9. Backward Compatibility

- `GenericFeatureEngineer`: Adapter for backward compatibility with the old API
- `create_hierarchical_categories`: Function for backward compatibility with the old API
- `enhance_features`: Function for backward compatibility with the old API

### 10. Type Stubs

- Created type stubs for all feature engineering components to improve type safety

### 11. Tests

- Created unit tests for all feature engineering components
- Verified that all components work correctly

### 12. Example Script

- Created an example script that demonstrates how to use the feature engineering components

## SOLID Principles Implementation

### Single Responsibility Principle (SRP)

Each transformer and feature engineer has a single responsibility:
- `TextCombiner` only combines text columns
- `NumericCleaner` only cleans numeric columns
- `HierarchyBuilder` only builds hierarchical columns
- `ConfigDrivenFeatureEngineer` only creates and applies transformers based on a configuration

### Open/Closed Principle (OCP)

The feature engineering system is open for extension but closed for modification:
- New transformers can be added without modifying existing code
- New feature engineers can be added without modifying existing code
- The `TransformerRegistry` allows for dynamic registration of transformers

### Liskov Substitution Principle (LSP)

Transformers and feature engineers can be substituted for each other:
- All transformers implement the `FeatureTransformer` interface
- All feature engineers implement the `FeatureEngineer` interface
- Base classes provide default implementations that can be overridden

### Interface Segregation Principle (ISP)

Interfaces are focused and minimal:
- `FeatureTransformer` only defines methods for transforming data
- `ColumnTransformer` adds methods for column-specific transformers
- `ConfigurableTransformer` adds methods for configurable transformers

### Dependency Inversion Principle (DIP)

High-level modules depend on abstractions:
- Feature engineers depend on the `FeatureTransformer` interface, not concrete transformers
- The `ConfigDrivenFeatureEngineer` depends on the `TransformerRegistry` interface, not concrete registries
- The `GenericFeatureEngineer` adapter allows high-level code to depend on abstractions rather than legacy code

## Usage Examples

### Basic Usage

```python
from nexusml.core.feature_engineering import (
    TextCombiner,
    NumericCleaner,
    HierarchyBuilder,
    BaseFeatureEngineer,
)

# Create transformers
text_combiner = TextCombiner(
    columns=['Asset Category', 'Equip Name ID'],
    separator=' - ',
    new_column='Equipment_Type'
)

numeric_cleaner = NumericCleaner(
    column='Service Life',
    new_name='service_life_years',
    fill_value=0,
    dtype='int'
)

hierarchy_builder = HierarchyBuilder(
    parent_columns=['Asset Category', 'Equip Name ID'],
    new_column='Equipment_Hierarchy',
    separator='/'
)

# Create a feature engineer
feature_engineer = BaseFeatureEngineer()
feature_engineer.add_transformer(text_combiner)
feature_engineer.add_transformer(numeric_cleaner)
feature_engineer.add_transformer(hierarchy_builder)

# Apply the feature engineer
df_transformed = feature_engineer.fit_transform(df)
```

### Configuration-Driven Usage

```python
from nexusml.core.feature_engineering import ConfigDrivenFeatureEngineer

# Create a configuration
config = {
    "text_combinations": [
        {
            "columns": ["Asset Category", "Equip Name ID"],
            "separator": " - ",
            "name": "Equipment_Type"
        }
    ],
    "numeric_columns": [
        {
            "name": "Service Life",
            "new_name": "service_life_years",
            "fill_value": 0,
            "dtype": "int"
        }
    ],
    "hierarchies": [
        {
            "parents": ["Asset Category", "Equip Name ID"],
            "new_col": "Equipment_Hierarchy",
            "separator": "/"
        }
    ],
    "column_mappings": [
        {
            "source": "Manufacturer",
            "target": "equipment_manufacturer"
        },
        {
            "source": "Model",
            "target": "equipment_model"
        }
    ]
}

# Create a configuration-driven feature engineer
config_driven_fe = ConfigDrivenFeatureEngineer(config=config)

# Apply the configuration-driven feature engineer
df_transformed = config_driven_fe.fit_transform(df)
```

### Custom Transformer

```python
from nexusml.core.feature_engineering import (
    BaseColumnTransformer,
    register_transformer,
    create_transformer,
)

# Define a custom transformer
class ManufacturerNormalizer(BaseColumnTransformer):
    """
    Normalizes manufacturer names by converting to uppercase and removing special characters.
    """
    
    def __init__(
        self,
        column: str = "Manufacturer",
        new_column: str = "normalized_manufacturer",
        name: str = "ManufacturerNormalizer",
    ):
        """
        Initialize the manufacturer normalizer.
        
        Args:
            column: Name of the column containing manufacturer names.
            new_column: Name of the new column to create.
            name: Name of the transformer.
        """
        super().__init__([column], [new_column], name)
        self.column = column
        self.new_column = new_column
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize manufacturer names.
        
        Args:
            X: Input DataFrame to transform.
            
        Returns:
            Transformed DataFrame with normalized manufacturer names.
        """
        import re
        
        # Convert to uppercase and remove special characters
        X[self.new_column] = X[self.column].str.upper()
        X[self.new_column] = X[self.new_column].apply(
            lambda x: re.sub(r'[^A-Z0-9]', '', x) if isinstance(x, str) else x
        )
        
        return X

# Register the custom transformer
register_transformer("manufacturer_normalizer", ManufacturerNormalizer)

# Create an instance of the custom transformer
manufacturer_normalizer = create_transformer("manufacturer_normalizer")

# Apply the custom transformer
df_transformed = manufacturer_normalizer.fit_transform(df)
```

## Conclusion

The Feature Engineering component provides a comprehensive system for transforming data in the NexusML suite. It follows SOLID principles, improves type safety, and provides a flexible and extensible architecture for feature engineering. The component can be used to transform data in various ways, and can be extended with custom transformers and feature engineers.

## Next Steps

The next step in Phase 2 of the NexusML refactoring is to implement the Model Building and Training component. This component will build on the foundation laid by the Feature Engineering component and will follow the same SOLID principles and type safety improvements.