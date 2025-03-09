"""
Feature Engineering Example

This script demonstrates how to use the feature engineering components in the NexusML suite.
"""

import pandas as pd
import numpy as np

from nexusml.core.feature_engineering import (
    # Base classes
    BaseFeatureTransformer,
    BaseColumnTransformer,
    BaseFeatureEngineer,
    
    # Transformers
    TextCombiner,
    NumericCleaner,
    HierarchyBuilder,
    ColumnMapper,
    
    # Config-driven feature engineer
    ConfigDrivenFeatureEngineer,
    enhance_features,
    
    # Registry
    register_transformer,
    create_transformer,
)


def main():
    """Run the feature engineering example."""
    print("NexusML Feature Engineering Example")
    print("===================================")
    
    # Create a sample DataFrame
    print("\nCreating sample DataFrame...")
    df = pd.DataFrame({
        'Asset Category': ['HVAC', 'Plumbing', 'Electrical', 'HVAC', 'Plumbing'],
        'Equip Name ID': ['AHU', 'Pump', 'Panel', 'Chiller', 'Fixture'],
        'Service Life': [15, 20, 30, 25, 10],
        'Manufacturer': ['Carrier', 'Grundfos', 'Square D', 'Trane', 'Kohler'],
        'Model': ['ABC123', 'XYZ456', 'DEF789', 'GHI012', 'JKL345'],
    })
    print(df)
    
    # Example 1: Using individual transformers
    print("\nExample 1: Using individual transformers")
    print("---------------------------------------")
    
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
    
    print(df_transformed)
    
    # Example 2: Using a feature engineer
    print("\nExample 2: Using a feature engineer")
    print("----------------------------------")
    
    # Create a feature engineer
    feature_engineer = BaseFeatureEngineer()
    
    # Add transformers to the feature engineer
    feature_engineer.add_transformer(text_combiner)
    feature_engineer.add_transformer(numeric_cleaner)
    feature_engineer.add_transformer(hierarchy_builder)
    
    # Apply the feature engineer
    df_transformed2 = feature_engineer.fit_transform(df.copy())
    
    print(df_transformed2)
    
    # Example 3: Using a configuration-driven feature engineer
    print("\nExample 3: Using a configuration-driven feature engineer")
    print("------------------------------------------------------")
    
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
    
    print(df_transformed3)
    
    # Example 4: Creating a custom transformer
    print("\nExample 4: Creating a custom transformer")
    print("---------------------------------------")
    
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
    df_transformed4 = manufacturer_normalizer.fit_transform(df.copy())
    
    print(df_transformed4)
    
    # Example 5: Using the enhance_features function
    print("\nExample 5: Using the enhance_features function")
    print("--------------------------------------------")
    
    # Create a feature engineer with the custom configuration
    feature_engineer = ConfigDrivenFeatureEngineer(config=config)
    
    # Fit the feature engineer
    feature_engineer.fit(df.copy())
    
    # Apply the enhance_features function with the fitted feature engineer
    df_transformed5 = enhance_features(df.copy(), feature_engineer)
    
    print(df_transformed5)


if __name__ == "__main__":
    main()
