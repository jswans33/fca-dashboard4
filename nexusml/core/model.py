"""
Enhanced Equipment Classification Model with EAV Integration

This module implements a machine learning pipeline for classifying equipment based on text descriptions
and numeric features, with integrated EAV (Entity-Attribute-Value) structure. Key features include:

1. Combined Text and Numeric Features:
   - Uses a ColumnTransformer to incorporate both text features (via TF-IDF) and numeric features
     (like service_life) into a single model.

2. Improved Handling of Imbalanced Classes:
   - Uses class_weight='balanced_subsample' in the RandomForestClassifier for handling imbalanced classes.

3. Better Evaluation Metrics:
   - Uses f1_macro scoring for hyperparameter optimization, which is more appropriate for
     imbalanced classes than accuracy.
   - Provides detailed analysis of "Other" category performance.

4. Feature Importance Analysis:
   - Analyzes the importance of both text and numeric features in classifying equipment.

5. EAV Integration:
   - Incorporates EAV structure for flexible equipment attributes
   - Uses classification systems (OmniClass, MasterFormat, Uniformat) for comprehensive taxonomy
   - Includes performance fields (service life, maintenance intervals) in feature engineering
   - Can predict missing attribute values based on equipment descriptions
"""

# Standard library imports
import os
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from nexusml.core.data_mapper import (
    map_predictions_to_master_db,
    map_staging_to_model_input,
)

# Local imports
from nexusml.core.data_preprocessing import load_and_preprocess_data
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.dynamic_mapper import DynamicFieldMapper
from nexusml.core.eav_manager import EAVManager, EAVTransformer
from nexusml.core.evaluation import (
    analyze_other_category_features,
    analyze_other_misclassifications,
    enhanced_evaluation,
)
from nexusml.core.feature_engineering import (
    GenericFeatureEngineer,
    create_hierarchical_categories,
    enhance_features,
    enhanced_masterformat_mapping,
)
from nexusml.core.model_building import build_enhanced_model


@injectable
class EquipmentClassifier:
    """
    Comprehensive equipment classifier with EAV integration.

    This class uses dependency injection to receive its dependencies,
    making it more testable and configurable.
    """

    @inject
    def __init__(
        self,
        model=None,
        feature_engineer: Optional[GenericFeatureEngineer] = None,
        eav_manager: Optional[EAVManager] = None,
        sampling_strategy: str = "direct",
    ):
        """
        Initialize the equipment classifier.

        Args:
            model: Trained ML model (if None, needs to be trained)
            feature_engineer: Feature engineering transformer (injected)
            eav_manager: EAV manager for attribute templates (injected)
            sampling_strategy: Strategy for handling class imbalance
        """
        self.model = model
        # Ensure we have a feature engineer and EAV manager
        self.feature_engineer = feature_engineer or GenericFeatureEngineer()
        self.eav_manager = eav_manager or EAVManager()
        self.sampling_strategy = sampling_strategy

    def train(
        self,
        data_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        sampling_strategy: str = "direct",
        **kwargs,
    ) -> None:
        """
        Train the equipment classifier.

        Args:
            data_path: Path to the training data
            feature_config_path: Path to the feature configuration
            sampling_strategy: Strategy for handling class imbalance (default: "direct")
            **kwargs: Additional parameters for training
        """
        # Use the provided sampling_strategy or fall back to self.sampling_strategy if it exists
        strategy = sampling_strategy
        if hasattr(self, 'sampling_strategy'):
            strategy = self.sampling_strategy
            
        # Train the model using the train_enhanced_model function
        self.model, self.df = train_enhanced_model(
            data_path=data_path,
            sampling_strategy=strategy,
            feature_config_path=feature_config_path,
            **kwargs,
        )

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from a file.

        Args:
            model_path: Path to the saved model file
        """
        import pickle

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        print(f"Model loaded from {model_path}")

    def predict(
        self, description: str, service_life: float = 0.0, asset_tag: str = ""
    ) -> Dict[str, Any]:
        """
        Predict equipment classifications from a description.

        Args:
            description: Text description of the equipment
            service_life: Service life value (optional)
            asset_tag: Asset tag for equipment (optional)

        Returns:
            Dictionary with classification results and master DB mappings
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Use the predict_with_enhanced_model function
        result = predict_with_enhanced_model(
            self.model, description, service_life, asset_tag
        )

        # Add EAV template for the predicted equipment type
        # Use category_name instead of Equipment_Category, and add Equipment_Category for backward compatibility
        if "category_name" in result:
            equipment_type = result["category_name"]
            result["Equipment_Category"] = (
                equipment_type  # Add for backward compatibility
            )
        else:
            equipment_type = "Unknown"
            result["Equipment_Category"] = equipment_type
            result["category_name"] = equipment_type

        # Create EAVManager if it doesn't exist
        if not hasattr(self, 'eav_manager') or self.eav_manager is None:
            self.eav_manager = EAVManager()
            
        # Generate attribute template
        try:
            result["attribute_template"] = self.eav_manager.generate_attribute_template(
                equipment_type
            )
        except Exception as e:
            # Provide a default attribute template if generation fails
            result["attribute_template"] = {
                "equipment_type": equipment_type,
                "classification": {},
                "required_attributes": {},
                "optional_attributes": {}
            }
            print(f"Warning: Could not generate attribute template: {e}")

        return result

    def predict_from_row(self, row: pd.Series) -> Dict[str, Any]:
        """
        Predict equipment classifications from a DataFrame row.

        This method is designed to work with rows that have already been processed
        by the feature engineering pipeline.

        Args:
            row: A pandas Series representing a row from a DataFrame

        Returns:
            Dictionary with classification results and master DB mappings
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Extract the description from the row
        if "combined_text" in row and row["combined_text"]:
            description = row["combined_text"]
        else:
            # Fallback to creating a combined description
            description = f"{row.get('equipment_tag', '')} {row.get('manufacturer', '')} {row.get('model', '')} {row.get('category_name', '')} {row.get('mcaa_system_category', '')}"

        # Extract service life
        service_life = 20.0
        if "service_life" in row and not pd.isna(row["service_life"]):
            service_life = float(row["service_life"])
        elif "condition_score" in row and not pd.isna(row["condition_score"]):
            service_life = float(row["condition_score"])

        # Extract asset tag
        asset_tag = ""
        if "equipment_tag" in row and not pd.isna(row["equipment_tag"]):
            asset_tag = str(row["equipment_tag"])

        # Instead of making predictions, use the actual values from the input data
        result = {
            "category_name": row.get("category_name", "Unknown"),
            "uniformat_code": row.get("uniformat_code", ""),
            "mcaa_system_category": row.get("mcaa_system_category", ""),
            "Equipment_Type": row.get("Equipment_Type", ""),
            "System_Subtype": row.get("System_Subtype", ""),
            "Asset Tag": asset_tag,
        }

        # Add MasterFormat prediction with enhanced mapping
        result["MasterFormat_Class"] = enhanced_masterformat_mapping(
            result["uniformat_code"],
            result["mcaa_system_category"],
            result["category_name"],
            (
                result["Equipment_Type"].split("-")[1]
                if "-" in result["Equipment_Type"]
                else None
            ),
        )

        # Add EAV template for the predicted equipment type
        equipment_type = result["category_name"]
        result["Equipment_Category"] = equipment_type  # Add for backward compatibility
        
        # Create EAVManager if it doesn't exist
        if not hasattr(self, 'eav_manager') or self.eav_manager is None:
            self.eav_manager = EAVManager()
            
        # Generate attribute template
        try:
            result["attribute_template"] = self.eav_manager.generate_attribute_template(
                equipment_type
            )
        except Exception as e:
            # Provide a default attribute template if generation fails
            result["attribute_template"] = {
                "equipment_type": equipment_type,
                "classification": {},
                "required_attributes": {},
                "optional_attributes": {}
            }
            print(f"Warning: Could not generate attribute template: {e}")

        # Map predictions to master database fields
        result["master_db_mapping"] = map_predictions_to_master_db(result)

        return result

    def predict_attributes(
        self, equipment_type: str, description: str
    ) -> Dict[str, Any]:
        """
        Predict attribute values for a given equipment type and description.

        Args:
            equipment_type: Type of equipment
            description: Text description of the equipment

        Returns:
            Dictionary with predicted attribute values
        """
        # This is a placeholder for attribute prediction
        # In a real implementation, this would use ML to predict attribute values
        # based on the description and equipment type
        
        # Create EAVManager if it doesn't exist
        if not hasattr(self, 'eav_manager') or self.eav_manager is None:
            self.eav_manager = EAVManager()
            
        try:
            template = self.eav_manager.get_equipment_template(equipment_type)
            required_attrs = template.get("required_attributes", [])
        except Exception as e:
            print(f"Warning: Could not get equipment template: {e}")
            template = {"required_attributes": []}
            required_attrs = []

        # Simple rule-based attribute prediction based on keywords in description
        predictions = {}

        # Extract numeric values with units from description
        import re

        # Look for patterns like "100 tons", "5 HP", etc.
        capacity_pattern = r"(\d+(?:\.\d+)?)\s*(?:ton|tons)"
        flow_pattern = r"(\d+(?:\.\d+)?)\s*(?:gpm|GPM)"
        pressure_pattern = r"(\d+(?:\.\d+)?)\s*(?:psi|PSI|psig|PSIG)"
        temp_pattern = r"(\d+(?:\.\d+)?)\s*(?:°F|F|deg F)"
        airflow_pattern = r"(\d+(?:\.\d+)?)\s*(?:cfm|CFM)"

        # Check for cooling capacity
        if "cooling_capacity_tons" in required_attrs:
            match = re.search(capacity_pattern, description)
            if match:
                predictions["cooling_capacity_tons"] = float(match.group(1))

        # Check for flow rate
        if "flow_rate_gpm" in required_attrs:
            match = re.search(flow_pattern, description)
            if match:
                predictions["flow_rate_gpm"] = float(match.group(1))

        # Check for pressure
        pressure_attrs = [attr for attr in required_attrs if "pressure" in attr]
        if pressure_attrs and re.search(pressure_pattern, description):
            match = re.search(pressure_pattern, description)
            if match:
                predictions[pressure_attrs[0]] = float(match.group(1))

        # Check for temperature
        temp_attrs = [attr for attr in required_attrs if "temp" in attr]
        if temp_attrs and re.search(temp_pattern, description):
            match = re.search(temp_pattern, description)
            if match:
                predictions[temp_attrs[0]] = float(match.group(1))

        # Check for airflow
        if "airflow_cfm" in required_attrs:
            match = re.search(airflow_pattern, description)
            if match:
                predictions["airflow_cfm"] = float(match.group(1))

        # Check for equipment types
        if "chiller_type" in required_attrs:
            if "centrifugal" in description.lower():
                predictions["chiller_type"] = "Centrifugal"
            elif "absorption" in description.lower():
                predictions["chiller_type"] = "Absorption"
            elif "screw" in description.lower():
                predictions["chiller_type"] = "Screw"
            elif "scroll" in description.lower():
                predictions["chiller_type"] = "Scroll"
            elif "reciprocating" in description.lower():
                predictions["chiller_type"] = "Reciprocating"

        if "pump_type" in required_attrs:
            if "centrifugal" in description.lower():
                predictions["pump_type"] = "Centrifugal"
            elif "positive displacement" in description.lower():
                predictions["pump_type"] = "Positive Displacement"
            elif "submersible" in description.lower():
                predictions["pump_type"] = "Submersible"
            elif "vertical" in description.lower():
                predictions["pump_type"] = "Vertical Turbine"

        # Add more attribute predictions as needed

        return predictions

    def fill_missing_attributes(
        self, equipment_type: str, attributes: Dict[str, Any], description: str
    ) -> Dict[str, Any]:
        """
        Fill in missing attributes using ML predictions and rules.

        Args:
            equipment_type: Type of equipment
            attributes: Dictionary of existing attribute name-value pairs
            description: Text description of the equipment

        Returns:
            Dictionary with filled attributes
        """
        # Create EAVManager if it doesn't exist
        if not hasattr(self, 'eav_manager') or self.eav_manager is None:
            self.eav_manager = EAVManager()
            
        try:
            return self.eav_manager.fill_missing_attributes(
                equipment_type, attributes, description, self
            )
        except Exception as e:
            print(f"Warning: Could not fill missing attributes: {e}")
            return attributes

    def validate_attributes(
        self, equipment_type: str, attributes: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Validate attributes against the template for an equipment type.

        Args:
            equipment_type: Type of equipment
            attributes: Dictionary of attribute name-value pairs

        Returns:
            Dictionary with validation results
        """
        # Create EAVManager if it doesn't exist
        if not hasattr(self, 'eav_manager') or self.eav_manager is None:
            self.eav_manager = EAVManager()
            
        try:
            return self.eav_manager.validate_attributes(equipment_type, attributes)
        except Exception as e:
            print(f"Warning: Could not validate attributes: {e}")
            return {"errors": [], "warnings": []}


def train_enhanced_model(
    data_path: Optional[str] = None,
    sampling_strategy: str = "direct",
    feature_config_path: Optional[str] = None,
    eav_manager: Optional[EAVManager] = None,
    feature_engineer: Optional[GenericFeatureEngineer] = None,
    **kwargs,
) -> Tuple[Any, pd.DataFrame]:
    """
    Train and evaluate the enhanced model with EAV integration

    Args:
        data_path: Path to the CSV file. Defaults to None, which uses the standard location.
        sampling_strategy: Strategy for handling class imbalance ("direct" is the only supported option for now)
        feature_config_path: Path to the feature configuration file. Defaults to None, which uses the standard location.
        eav_manager: EAVManager instance. If None, uses the one from the DI container.
        feature_engineer: GenericFeatureEngineer instance. If None, uses the one from the DI container.
        **kwargs: Additional parameters for the model

    Returns:
        tuple: (trained model, preprocessed dataframe)
    """
    # Get dependencies from DI container if not provided
    if eav_manager is None or feature_engineer is None:
        from nexusml.core.di.provider import ContainerProvider

        container = ContainerProvider().container

        if eav_manager is None:
            try:
                eav_manager = container.resolve(EAVManager)
            except Exception:
                # If EAVManager is not registered in the container, create it directly
                eav_manager = EAVManager()

        if feature_engineer is None:
            # Create a new feature engineer with the provided config path and EAV manager
            feature_engineer = GenericFeatureEngineer(
                config_path=feature_config_path, eav_manager=eav_manager
            )

    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)

    # 1.5. Map staging data columns to model input format
    print("Mapping staging data columns to model input format...")
    df = map_staging_to_model_input(df)

    # 2. Apply Generic Feature Engineering with EAV integration
    print("Applying Generic Feature Engineering with EAV integration...")
    df = feature_engineer.transform(df)

    # 3. Prepare training data - now including both text and numeric features
    # Create a DataFrame with both text and numeric features
    x = pd.DataFrame(
        {
            "combined_text": df["combined_text"],  # Using the name from config
            "service_life": df["service_life"],
        }
    )

    # Use hierarchical classification targets
    y = df[
        [
            "category_name",  # Use category_name instead of Equipment_Category
            "uniformat_code",  # Use uniformat_code instead of Uniformat_Class
            "mcaa_system_category",  # Use mcaa_system_category instead of System_Type
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # 4. Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # 5. Print class distribution information
    print("\nClass distribution information:")
    for col in y.columns:
        print(f"\nClass distribution for {col}:")
        print(y[col].value_counts())

    # 6. Build enhanced model with class_weight='balanced_subsample'
    print("\nBuilding enhanced model with balanced class weights...")
    model = build_enhanced_model(sampling_strategy=sampling_strategy, **kwargs)

    # 7. Train the model
    print("Training model...")
    model.fit(x_train, y_train)

    # 8. Evaluate with focus on "Other" categories
    print("Evaluating model...")
    y_pred_df = enhanced_evaluation(model, x_test, y_test)

    # 9. Analyze "Other" category features
    print("Analyzing 'Other' category features...")
    analyze_other_category_features(model, x_test, y_test, y_pred_df)

    # 10. Analyze misclassifications for "Other" categories
    print("Analyzing 'Other' category misclassifications...")
    analyze_other_misclassifications(x_test, y_test, y_pred_df)

    return model, df


def predict_with_enhanced_model(
    model: Any,
    description: str,
    service_life: float = 0.0,
    asset_tag: str = "",
    eav_manager: Optional[EAVManager] = None,
) -> dict:
    """
    Make predictions with enhanced detail for "Other" categories

    This function has been updated to work with the new pipeline structure that uses
    both text and numeric features.

    Args:
        model: Trained model pipeline
        description (str): Text description to classify
        service_life (float, optional): Service life value. Defaults to 0.0.
        asset_tag (str, optional): Asset tag for equipment. Defaults to "".
        eav_manager (Optional[EAVManager], optional): EAV manager instance. If None, uses the one from the DI container.

    Returns:
        dict: Prediction results with classifications and master DB mappings
    """
    # Get EAV manager from DI container if not provided
    if eav_manager is None:
        from nexusml.core.di.provider import ContainerProvider

        container = ContainerProvider().container
        try:
            eav_manager = container.resolve(EAVManager)
        except Exception:
            # If EAVManager is not registered in the container, create it directly
            eav_manager = EAVManager()

    # Create a DataFrame with the required structure for the pipeline
    input_data = pd.DataFrame(
        {"combined_text": [description], "service_life": [service_life]}
    )

    # Predict using the trained pipeline
    pred = model.predict(input_data)[0]

    # Extract predictions
    result = {
        "category_name": pred[0],  # Use category_name instead of Equipment_Category
        "uniformat_code": pred[1],  # Use uniformat_code instead of Uniformat_Class
        "mcaa_system_category": pred[
            2
        ],  # Use mcaa_system_category instead of System_Type
        "Equipment_Type": pred[3],
        "System_Subtype": pred[4],
        "Asset Tag": asset_tag,  # Add asset tag for master DB mapping
    }

    # Add MasterFormat prediction with enhanced mapping
    result["MasterFormat_Class"] = enhanced_masterformat_mapping(
        result["uniformat_code"],  # Use uniformat_code instead of Uniformat_Class
        result[
            "mcaa_system_category"
        ],  # Use mcaa_system_category instead of System_Type
        result["category_name"],  # Use category_name instead of Equipment_Category
        # Extract equipment subcategory if available
        (
            result["Equipment_Type"].split("-")[1]
            if "-" in result["Equipment_Type"]
            else None
        ),
    )

    # Add EAV template information
    try:
        equipment_type = result[
            "category_name"
        ]  # Use category_name instead of Equipment_Category

        # Get classification IDs
        classification_ids = eav_manager.get_classification_ids(equipment_type)

        # Only add these if they exist in the result
        if "omniclass_code" in result:
            result["OmniClass_ID"] = result["omniclass_code"]
        if "uniformat_code" in result:
            result["Uniformat_ID"] = result["uniformat_code"]

        # Get performance fields
        performance_fields = eav_manager.get_performance_fields(equipment_type)
        for field, info in performance_fields.items():
            result[field] = info.get("default", 0)

        # Get required attributes
        result["required_attributes"] = eav_manager.get_required_attributes(
            equipment_type
        )
    except Exception as e:
        print(f"Warning: Could not add EAV information to prediction: {e}")

    # Map predictions to master database fields
    result["master_db_mapping"] = map_predictions_to_master_db(result)

    return result


def visualize_category_distribution(
    df: pd.DataFrame, output_dir: str = "outputs"
) -> Tuple[str, str]:
    """
    Visualize the distribution of categories in the dataset

    Args:
        df (pd.DataFrame): DataFrame with category columns
        output_dir (str, optional): Directory to save visualizations. Defaults to "outputs".

    Returns:
        Tuple[str, str]: Paths to the saved visualization files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    equipment_category_file = f"{output_dir}/equipment_category_distribution.png"
    system_type_file = f"{output_dir}/system_type_distribution.png"

    # Generate visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="Equipment_Category")
    plt.title("Equipment Category Distribution")
    plt.tight_layout()
    plt.savefig(equipment_category_file)

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="System_Type")
    plt.title("System Type Distribution")
    plt.tight_layout()
    plt.savefig(system_type_file)

    return equipment_category_file, system_type_file


def train_model_from_any_data(
    df: pd.DataFrame, mapper: Optional[DynamicFieldMapper] = None
):
    """
    Train a model using data with any column structure.

    Args:
        df: Input DataFrame with arbitrary column names
        mapper: Optional DynamicFieldMapper instance

    Returns:
        Tuple: (trained model, transformed DataFrame)
    """
    # Create mapper if not provided
    if mapper is None:
        mapper = DynamicFieldMapper()

    # Map input fields to expected model fields
    mapped_df = mapper.map_dataframe(df)

    # Get the classification targets
    classification_targets = mapper.get_classification_targets()

    # Since train_enhanced_model expects a file path, we need to modify our approach
    # We'll use the core components of train_enhanced_model directly

    # Apply Generic Feature Engineering with EAV integration
    print("Applying Generic Feature Engineering with EAV integration...")
    eav_manager = EAVManager()
    feature_engineer = GenericFeatureEngineer(eav_manager=eav_manager)
    transformed_df = feature_engineer.transform(mapped_df)

    # Prepare training data - now including both text and numeric features
    # Create a DataFrame with both text and numeric features
    x = pd.DataFrame(
        {
            "combined_features": transformed_df[
                "combined_text"
            ],  # Using the name from config
            "service_life": transformed_df["service_life"],
        }
    )

    # Use hierarchical classification targets
    y = transformed_df[
        [
            "Equipment_Category",
            "Uniformat_Class",
            "System_Type",
            "Equipment_Type",
            "System_Subtype",
        ]
    ]

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # Build enhanced model
    print("Building enhanced model with balanced class weights...")
    model = build_enhanced_model(sampling_strategy="direct")

    # Train the model
    print("Training model...")
    model.fit(x_train, y_train)

    # Evaluate with focus on "Other" categories
    print("Evaluating model...")
    y_pred_df = enhanced_evaluation(model, x_test, y_test)

    return model, transformed_df


# Example usage
if __name__ == "__main__":
    # Create and train the equipment classifier
    classifier = EquipmentClassifier()
    classifier.train()

    # Example prediction with service life
    description = "Heat Exchanger for Chilled Water system with Plate and Frame design"
    service_life = 20.0  # Example service life in years
    prediction = classifier.predict(description, service_life)

    print("\nEnhanced Prediction with EAV Integration:")
    for key, value in prediction.items():
        if key != "attribute_template":  # Skip printing the full template
            print(f"{key}: {value}")

    print("\nAttribute Template:")
    template = prediction["attribute_template"]
    print(f"Equipment Type: {template['equipment_type']}")
    print(f"Classification: {template['classification']}")
    print("Required Attributes:")
    for attr, info in template["required_attributes"].items():
        print(f"  {attr}: {info}")

    # Visualize category distribution
    equipment_category_file, system_type_file = visualize_category_distribution(
        classifier.df
    )

    print("\nVisualizations saved to:")
    print(f"  - {equipment_category_file}")
    print(f"  - {system_type_file}")
