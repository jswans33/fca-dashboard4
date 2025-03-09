"""
Pipeline Stages Package

This package provides implementations of pipeline stages for the NexusML pipeline system.
Each stage represents a distinct step in the pipeline execution process and follows
the Single Responsibility Principle (SRP) from SOLID.
"""

# Import interfaces
from nexusml.core.pipeline.stages.interfaces import (
    DataLoadingStage,
    DataSplittingStage,
    FeatureEngineeringStage,
    ModelBuildingStage,
    ModelEvaluationStage,
    ModelSavingStage,
    ModelTrainingStage,
    PipelineStage,
    PredictionStage,
    ValidationStage,
)

# Import base implementations
from nexusml.core.pipeline.stages.base import (
    BaseDataLoadingStage,
    BaseDataSplittingStage,
    BaseFeatureEngineeringStage,
    BaseModelBuildingStage,
    BaseModelEvaluationStage,
    BaseModelSavingStage,
    BaseModelTrainingStage,
    BasePipelineStage,
    BasePredictionStage,
    BaseValidationStage,
)

# Import concrete implementations
from nexusml.core.pipeline.stages.data_loading import (
    CSVDataLoadingStage,
    ConfigurableDataLoadingStage,
    ExcelDataLoadingStage,
    SQLiteDataLoadingStage,
)

from nexusml.core.pipeline.stages.data_splitting import (
    ConfigDrivenDataSplittingStage,
    CrossValidationSplittingStage,
    RandomSplittingStage,
    StratifiedSplittingStage,
    TimeSeriesSplittingStage,
)

from nexusml.core.pipeline.stages.feature_engineering import (
    CompositeFeatureEngineeringStage,
    ConfigDrivenFeatureEngineeringStage,
    HierarchicalFeatureEngineeringStage,
    NumericFeatureEngineeringStage,
    SimpleFeatureEngineeringStage,
    TextFeatureEngineeringStage,
)

from nexusml.core.pipeline.stages.model_building import (
    ConfigDrivenModelBuildingStage,
    EnsembleModelBuildingStage,
    GradientBoostingModelBuildingStage,
    RandomForestModelBuildingStage,
)

from nexusml.core.pipeline.stages.model_evaluation import (
    ClassificationEvaluationStage,
    ConfigDrivenModelEvaluationStage,
    DetailedClassificationEvaluationStage,
)

from nexusml.core.pipeline.stages.model_saving import (
    ConfigDrivenModelSavingStage,
    ModelCardSavingStage,
    PickleModelSavingStage,
)

from nexusml.core.pipeline.stages.model_training import (
    ConfigDrivenModelTrainingStage,
    CrossValidationTrainingStage,
    GridSearchTrainingStage,
    RandomizedSearchTrainingStage,
    StandardModelTrainingStage,
)

from nexusml.core.pipeline.stages.prediction import (
    ConfigDrivenPredictionStage,
    ProbabilityPredictionStage,
    StandardPredictionStage,
    ThresholdPredictionStage,
)

from nexusml.core.pipeline.stages.validation import (
    ColumnValidationStage,
    CompositeValidationStage,
    ConfigDrivenValidationStage,
    DataFrameValidationStage,
    DataTypeValidationStage,
)

# Define __all__ to control what gets imported with "from nexusml.core.pipeline.stages import *"
__all__ = [
    # Interfaces
    "PipelineStage",
    "DataLoadingStage",
    "ValidationStage",
    "FeatureEngineeringStage",
    "DataSplittingStage",
    "ModelBuildingStage",
    "ModelTrainingStage",
    "ModelEvaluationStage",
    "ModelSavingStage",
    "PredictionStage",
    
    # Base implementations
    "BasePipelineStage",
    "BaseDataLoadingStage",
    "BaseValidationStage",
    "BaseFeatureEngineeringStage",
    "BaseDataSplittingStage",
    "BaseModelBuildingStage",
    "BaseModelTrainingStage",
    "BaseModelEvaluationStage",
    "BaseModelSavingStage",
    "BasePredictionStage",
    
    # Data loading stages
    "CSVDataLoadingStage",
    "ExcelDataLoadingStage",
    "SQLiteDataLoadingStage",
    "ConfigurableDataLoadingStage",
    
    # Data splitting stages
    "RandomSplittingStage",
    "StratifiedSplittingStage",
    "TimeSeriesSplittingStage",
    "CrossValidationSplittingStage",
    "ConfigDrivenDataSplittingStage",
    
    # Feature engineering stages
    "TextFeatureEngineeringStage",
    "NumericFeatureEngineeringStage",
    "HierarchicalFeatureEngineeringStage",
    "CompositeFeatureEngineeringStage",
    "SimpleFeatureEngineeringStage",
    "ConfigDrivenFeatureEngineeringStage",
    
    # Model building stages
    "RandomForestModelBuildingStage",
    "GradientBoostingModelBuildingStage",
    "EnsembleModelBuildingStage",
    "ConfigDrivenModelBuildingStage",
    
    # Model training stages
    "StandardModelTrainingStage",
    "CrossValidationTrainingStage",
    "GridSearchTrainingStage",
    "RandomizedSearchTrainingStage",
    "ConfigDrivenModelTrainingStage",
    
    # Model evaluation stages
    "ClassificationEvaluationStage",
    "DetailedClassificationEvaluationStage",
    "ConfigDrivenModelEvaluationStage",
    
    # Model saving stages
    "PickleModelSavingStage",
    "ModelCardSavingStage",
    "ConfigDrivenModelSavingStage",
    
    # Prediction stages
    "StandardPredictionStage",
    "ProbabilityPredictionStage",
    "ThresholdPredictionStage",
    "ConfigDrivenPredictionStage",
    
    # Validation stages
    "ColumnValidationStage",
    "DataTypeValidationStage",
    "CompositeValidationStage",
    "DataFrameValidationStage",
    "ConfigDrivenValidationStage",
]