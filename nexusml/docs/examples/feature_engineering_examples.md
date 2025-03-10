# Feature Engineering Examples

This document provides documentation for the feature engineering examples in NexusML, which demonstrate different approaches to transforming raw data into features suitable for machine learning.

## Feature Engineering Example

### Overview

The `feature_engineering_example.py` script demonstrates how to use the feature engineering components in NexusML to transform raw data into features suitable for machine learning. It showcases different approaches, from using individual transformers to configuration-driven feature engineering.

### Key Features

- Individual feature transformers for specific transformations
- Composite feature engineering with multiple transformers
- Configuration-driven feature engineering
- Custom transformer creation and registration
- Simplified API for feature enhancement

### Usage

```python
# Run the example
python -m nexusml.examples.feature_engineering_example
```

### Code Walkthrough

The example demonstrates five different approaches to feature engineering:

#### Example 1: Using Individual Transformers

This approach applies transformers one by one to the data:

```python
# Create a text combiner
text_combiner = TextCombiner(
    columns=['Asset Category', 'Equip Name ID'],
    separator=' - ',
    new_column='Equipment_Type'
)

# Create a numeric cleaner
numeric_cleaner = NumericCleaner(
    column='Service Life',
    new_name='service_life_years',
    fill_value=0,
    dtype='int'
)

# Create a hierarchy builder
hierarchy_builder = HierarchyBuilder(
    parent_columns=['Asset Category', 'Equip Name ID'],
    new_column='Equipment_Hierarchy',
    separator='/'
)

# Apply the transformers in sequence
df_transformed = df.copy()
df_transformed = text_combiner.fit_transform(df_transformed)
df_transformed = numeric_cleaner.fit_transform(df_transformed)
df_transformed = hierarchy_builder.fit_transform(df_transformed)
```

#### Example 2: Using a Feature Engineer

This approach uses a `BaseFeatureEngineer` to manage multiple transformers:

```python
# Create a feature engineer
feature_engineer = BaseFeatureEngineer()

# Add transformers to the feature engineer
feature_engineer.add_transformer(text_combiner)
feature_engineer.add_transformer(numeric_cleaner)
feature_engineer.add_transformer(hierarchy_builder)

# Apply the feature engineer
df_transformed2 = feature_engineer.fit_transform(df.copy())
```

#### Example 3: Using a Configuration-Driven Feature Engineer

This approach uses a configuration dictionary to define transformations:

```python
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
df_transformed3 = config_driven_fe.fit_transform(df.copy())
```

#### Example 4: Creating a Custom Transformer

This approach demonstrates how to create and register a custom transformer:

```python
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
        """Initialize the manufacturer normalizer."""
        super().__init__([column], [new_column], name)
        self.column = column
        self.new_column = new_column
    
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize manufacturer names."""
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
df_transformed4 = manufacturer_normalizer.fit_transform(df.copy())
```

#### Example 5: Using the enhance_features Function

This approach uses a simplified function to apply a fitted feature engineer:

```python
# Create a feature engineer with the custom configuration
feature_engineer = ConfigDrivenFeatureEngineer(config=config)

# Fit the feature engineer
feature_engineer.fit(df.copy())

# Apply the enhance_features function with the fitted feature engineer
df_transformed5 = enhance_features(df.copy(), feature_engineer)
```

### Key Components

#### Base Classes

- **BaseFeatureTransformer**: Abstract base class for all feature transformers
- **BaseColumnTransformer**: Base class for transformers that operate on specific columns
- **BaseFeatureEngineer**: Composite transformer that manages multiple transformers

#### Built-in Transformers

- **TextCombiner**: Combines multiple text columns into a single column
- **NumericCleaner**: Cleans and standardizes numeric columns
- **HierarchyBuilder**: Creates hierarchical features from parent columns
- **ColumnMapper**: Maps columns from one name to another

#### Configuration-Driven Components

- **ConfigDrivenFeatureEngineer**: Creates transformers based on a configuration dictionary
- **enhance_features**: Simplified function to apply a fitted feature engineer

#### Transformer Registry

- **register_transformer**: Registers a transformer class with a name
- **create_transformer**: Creates a transformer instance by name

### Dependencies

- nexusml.core.feature_engineering: For feature engineering components
- pandas: For data manipulation
- numpy: For numerical operations

### Notes and Warnings

- The `fit_transform` method should be used for training data, while `transform` should be used for test or prediction data
- Custom transformers should inherit from `BaseFeatureTransformer` or `BaseColumnTransformer`
- Configuration-driven feature engineering is the recommended approach for most use cases
- Transformers are applied in the order they are added to the feature engineer
- Some transformers may require specific column types (e.g., numeric, string)

## Best Practices

### When to Use Each Approach

1. **Individual Transformers**: Use when you need fine-grained control over each transformation step or when you're exploring data and need to see intermediate results.

2. **BaseFeatureEngineer**: Use when you have a fixed set of transformations that you want to apply in a specific order.

3. **ConfigDrivenFeatureEngineer**: Use for production code where you want to define transformations in a configuration file or dictionary that can be easily modified without changing code.

4. **Custom Transformers**: Create when you need specialized transformations not covered by the built-in transformers.

5. **enhance_features Function**: Use in prediction pipelines where you've already fitted a feature engineer on training data.

### Feature Engineering Pipeline Design

When designing a feature engineering pipeline:

1. **Start Simple**: Begin with basic transformations and add complexity as needed
2. **Test Incrementally**: Verify each transformation step produces the expected output
3. **Consider Performance**: Some transformations may be computationally expensive
4. **Maintain Consistency**: Use the same transformations for training and prediction
5. **Document Transformations**: Keep track of what transformations are applied and why

### Integration with Machine Learning Pipeline

Feature engineering is typically part of a larger machine learning pipeline:

```
Data Loading → Feature Engineering → Model Training → Evaluation → Prediction
```

The feature engineering components in NexusML are designed to integrate seamlessly with the rest of the pipeline:

```python
# Example integration with pipeline
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator

# Create orchestrator (simplified)
orchestrator = PipelineOrchestrator()

# Train model with feature engineering
model, metrics = orchestrator.train_model(
    data_path="path/to/data.csv",
    feature_config=config,  # Feature engineering configuration
    model_type="random_forest",
    output_dir="outputs/models",
)
```

## Next Steps

After understanding feature engineering, you might want to explore:

1. **Model Building Examples**: Learn how to build and train models using the engineered features
2. **Pipeline Examples**: See how feature engineering fits into the complete machine learning pipeline
3. **Domain-Specific Examples**: Explore feature engineering techniques specific to equipment classification