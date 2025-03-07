# NexusML Refactoring Suggestions

This document provides concrete refactoring suggestions to improve the NexusML
codebase according to SOLID principles, with a focus on configuration
management, pipeline consolidation, and dependency injection.

## 1. Configuration Management Refactoring

### Current Issues

- Multiple configuration files scattered across the codebase
- Inconsistent loading mechanisms
- Lack of validation for configuration values
- Hardcoded fallback values in multiple places

### Proposed Solution: Unified Configuration System

#### 1.1 Create a Central Configuration Class

```python
# nexusml/core/config/configuration.py
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import json
from pydantic import BaseModel, Field, validator

class FeatureEngineeringConfig(BaseModel):
    """Configuration for feature engineering components."""
    text_combinations: list = Field(default_factory=list)
    numeric_columns: list = Field(default_factory=list)
    hierarchies: list = Field(default_factory=list)
    column_mappings: list = Field(default_factory=list)
    classification_systems: list = Field(default_factory=list)
    eav_integration: Dict[str, Any] = Field(default_factory=dict)

    @validator('text_combinations')
    def validate_text_combinations(cls, v):
        for combo in v:
            if 'columns' not in combo:
                raise ValueError("Each text combination must have 'columns' field")
        return v

class DataConfig(BaseModel):
    """Configuration for data preprocessing."""
    required_columns: list = Field(default_factory=list)
    training_data: Dict[str, Any] = Field(default_factory=dict)

    @validator('training_data')
    def validate_training_data(cls, v):
        if 'default_path' not in v:
            v['default_path'] = "ingest/data/eq_ids.csv"
        if 'encoding' not in v:
            v['encoding'] = "utf-8"
        return v

class ClassificationConfig(BaseModel):
    """Configuration for classification targets and field mappings."""
    classification_targets: list = Field(default_factory=list)
    input_field_mappings: list = Field(default_factory=list)

class ModelConfig(BaseModel):
    """Configuration for model building and training."""
    tfidf: Dict[str, Any] = Field(default_factory=dict)
    random_forest: Dict[str, Any] = Field(default_factory=dict)

    @validator('tfidf')
    def validate_tfidf(cls, v):
        defaults = {
            'max_features': 5000,
            'ngram_range': [1, 3],
            'min_df': 2,
            'max_df': 0.9,
            'use_idf': True,
            'sublinear_tf': True
        }
        for key, default_value in defaults.items():
            if key not in v:
                v[key] = default_value
        return v

    @validator('random_forest')
    def validate_random_forest(cls, v):
        defaults = {
            'n_estimators': 200,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced_subsample',
            'random_state': 42
        }
        for key, default_value in defaults.items():
            if key not in v:
                v[key] = default_value
        return v

class ReferenceConfig(BaseModel):
    """Configuration for reference data sources."""
    paths: Dict[str, str] = Field(default_factory=dict)
    file_patterns: Dict[str, str] = Field(default_factory=dict)
    column_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    hierarchies: Dict[str, list] = Field(default_factory=dict)
    defaults: Dict[str, Any] = Field(default_factory=dict)

class NexusMLConfig(BaseModel):
    """Master configuration for NexusML."""
    feature_engineering: FeatureEngineeringConfig = Field(default_factory=FeatureEngineeringConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    eav_templates_path: Optional[str] = None
    masterformat_mappings_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'NexusMLConfig':
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_default(cls) -> 'NexusMLConfig':
        """Create a configuration with default values."""
        return cls()

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(config_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False)

    def get_eav_templates(self) -> Dict[str, Any]:
        """Load EAV templates from the specified path."""
        if not self.eav_templates_path:
            return {}

        try:
            with open(self.eav_templates_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load EAV templates: {e}")
            return {}

    def get_masterformat_mappings(self) -> tuple:
        """Load MasterFormat mappings from the specified path."""
        primary_mapping = {}
        equipment_specific_mapping = {}

        if not self.masterformat_mappings_path:
            return primary_mapping, equipment_specific_mapping

        try:
            primary_path = Path(self.masterformat_mappings_path) / "masterformat_primary.json"
            equipment_path = Path(self.masterformat_mappings_path) / "masterformat_equipment.json"

            with open(primary_path, 'r') as f:
                primary_mapping = json.load(f)

            with open(equipment_path, 'r') as f:
                equipment_specific_mapping = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load MasterFormat mappings: {e}")

        return primary_mapping, equipment_specific_mapping
```

#### 1.2 Create a Configuration Provider

```python
# nexusml/core/config/provider.py
from pathlib import Path
from typing import Optional
import os
from .configuration import NexusMLConfig

class ConfigurationProvider:
    """Singleton provider for NexusML configuration."""

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigurationProvider, cls).__new__(cls)
            cls._instance._config = None
        return cls._instance

    def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize the configuration provider with a configuration file."""
        if config_path:
            self._config = NexusMLConfig.from_yaml(config_path)
        else:
            # Try to find configuration in standard locations
            standard_paths = [
                Path.cwd() / "nexusml_config.yml",
                Path.home() / ".nexusml" / "config.yml",
                Path(__file__).resolve().parent.parent.parent / "config" / "nexusml_config.yml"
            ]

            # Check environment variable
            env_path = os.environ.get("NEXUSML_CONFIG")
            if env_path:
                standard_paths.insert(0, Path(env_path))

            # Try each path
            for path in standard_paths:
                if path.exists():
                    self._config = NexusMLConfig.from_yaml(path)
                    break

            # If no configuration found, use defaults
            if self._config is None:
                self._config = NexusMLConfig.from_default()

    @property
    def config(self) -> NexusMLConfig:
        """Get the current configuration."""
        if self._config is None:
            self.initialize()
        return self._config

    def reset(self) -> None:
        """Reset the configuration to defaults."""
        self._config = NexusMLConfig.from_default()
```

#### 1.3 Update Configuration Usage in Components

Example for `GenericFeatureEngineer`:

```python
# nexusml/core/feature_engineering.py (partial update)
from nexusml.core.config.provider import ConfigurationProvider

class GenericFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    A generic feature engineering transformer that applies multiple transformations
    based on configuration.
    """

    def __init__(
        self,
        config_provider=None,
        eav_manager=None,
    ):
        """
        Initialize the transformer with configuration.

        Args:
            config_provider: Configuration provider. If None, uses the default.
            eav_manager: EAVManager instance. If None, creates a new one.
        """
        self.config_provider = config_provider or ConfigurationProvider()
        self.transformers = []
        self.eav_manager = eav_manager or EAVManager()

    def transform(self, X):
        """
        Transform the input DataFrame based on the configuration.

        Args:
            X: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        X = X.copy()
        config = self.config_provider.config.feature_engineering

        # 1. Apply column mappings
        if config.column_mappings:
            mapper = ColumnMapper(config.column_mappings)
            X = mapper.transform(X)

        # 2. Apply text combinations
        if config.text_combinations:
            for combo in config.text_combinations:
                combiner = TextCombiner(
                    columns=combo["columns"],
                    separator=combo.get("separator", " "),
                    new_column=combo.get("name", "combined_text"),
                )
                X = combiner.transform(X)

        # Continue with other transformations...

        return X
```

## 2. Pipeline Consolidation

### Current Issues

- Pipeline components are scattered across multiple files
- Inconsistent interfaces between components
- Lack of clear pipeline stages
- Difficult to customize or extend the pipeline

### Proposed Solution: Pipeline Factory with Dependency Injection

#### 2.1 Create Pipeline Interfaces

```python
# nexusml/core/pipeline/interfaces.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import pandas as pd

class DataLoader(ABC):
    """Interface for data loading components."""

    @abstractmethod
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from a source."""
        pass

class DataPreprocessor(ABC):
    """Interface for data preprocessing components."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        pass

class FeatureEngineer(ABC):
    """Interface for feature engineering components."""

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from the data."""
        pass

class ModelBuilder(ABC):
    """Interface for model building components."""

    @abstractmethod
    def build_model(self) -> Any:
        """Build a model."""
        pass

class ModelTrainer(ABC):
    """Interface for model training components."""

    @abstractmethod
    def train(self, model: Any, X: Any, y: Any) -> Any:
        """Train a model."""
        pass

class ModelEvaluator(ABC):
    """Interface for model evaluation components."""

    @abstractmethod
    def evaluate(self, model: Any, X: Any, y: Any) -> Dict[str, Any]:
        """Evaluate a model."""
        pass

class ModelSerializer(ABC):
    """Interface for model serialization components."""

    @abstractmethod
    def save(self, model: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a model."""
        pass

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load a model."""
        pass

class Predictor(ABC):
    """Interface for prediction components."""

    @abstractmethod
    def predict(self, model: Any, input_data: Any) -> Any:
        """Make predictions with a model."""
        pass
```

#### 2.2 Implement Concrete Pipeline Components

Example for `StandardDataLoader`:

```python
# nexusml/core/pipeline/components.py (partial)
from typing import Dict, Optional
import pandas as pd
from .interfaces import DataLoader
from nexusml.core.config.provider import ConfigurationProvider

class StandardDataLoader(DataLoader):
    """Standard implementation of data loading."""

    def __init__(self, config_provider=None):
        """
        Initialize the data loader.

        Args:
            config_provider: Configuration provider. If None, uses the default.
        """
        self.config_provider = config_provider or ConfigurationProvider()

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file. If None, uses the path from configuration.

        Returns:
            Loaded DataFrame
        """
        config = self.config_provider.config.data

        # Use default path if none provided
        if data_path is None:
            data_path = config.training_data.get("default_path", "ingest/data/eq_ids.csv")

        # Read CSV file using pandas
        encoding = config.training_data.get("encoding", "utf-8")
        fallback_encoding = config.training_data.get("fallback_encoding", "latin1")

        try:
            df = pd.read_csv(data_path, encoding=encoding)
        except UnicodeDecodeError:
            # Try with a different encoding if the primary one fails
            print(f"Warning: Failed to read with {encoding} encoding. Trying {fallback_encoding}.")
            df = pd.read_csv(data_path, encoding=fallback_encoding)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {data_path}. Please provide a valid path.")

        # Clean up column names (remove any leading/trailing whitespace)
        df.columns = [col.strip() for col in df.columns]

        return df
```

#### 2.3 Create a Pipeline Factory

```python
# nexusml/core/pipeline/factory.py
from typing import Any, Dict, Optional, Type
from .interfaces import (
    DataLoader, DataPreprocessor, FeatureEngineer,
    ModelBuilder, ModelTrainer, ModelEvaluator,
    ModelSerializer, Predictor
)
from .components import (
    StandardDataLoader, StandardDataPreprocessor,
    GenericFeatureEngineer, RandomForestModelBuilder,
    StandardModelTrainer, EnhancedModelEvaluator,
    PickleModelSerializer, StandardPredictor
)
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.eav_manager import EAVManager
from nexusml.core.reference.manager import ReferenceManager

class PipelineFactory:
    """Factory for creating pipeline components with dependency injection."""

    def __init__(self, config_provider=None):
        """
        Initialize the pipeline factory.

        Args:
            config_provider: Configuration provider. If None, uses the default.
        """
        self.config_provider = config_provider or ConfigurationProvider()
        self.components = {}

    def register_component(self, interface: Type, implementation: Type) -> None:
        """
        Register a component implementation for an interface.

        Args:
            interface: Interface class
            implementation: Implementation class
        """
        self.components[interface] = implementation

    def create_data_loader(self) -> DataLoader:
        """Create a data loader component."""
        implementation = self.components.get(DataLoader, StandardDataLoader)
        return implementation(config_provider=self.config_provider)

    def create_data_preprocessor(self) -> DataPreprocessor:
        """Create a data preprocessor component."""
        implementation = self.components.get(DataPreprocessor, StandardDataPreprocessor)
        return implementation(config_provider=self.config_provider)

    def create_feature_engineer(self, eav_manager=None) -> FeatureEngineer:
        """
        Create a feature engineer component.

        Args:
            eav_manager: EAVManager instance. If None, creates a new one.
        """
        implementation = self.components.get(FeatureEngineer, GenericFeatureEngineer)
        eav_manager = eav_manager or EAVManager(
            templates_path=self.config_provider.config.eav_templates_path
        )
        return implementation(config_provider=self.config_provider, eav_manager=eav_manager)

    def create_model_builder(self) -> ModelBuilder:
        """Create a model builder component."""
        implementation = self.components.get(ModelBuilder, RandomForestModelBuilder)
        return implementation(config_provider=self.config_provider)

    def create_model_trainer(self) -> ModelTrainer:
        """Create a model trainer component."""
        implementation = self.components.get(ModelTrainer, StandardModelTrainer)
        return implementation(config_provider=self.config_provider)

    def create_model_evaluator(self) -> ModelEvaluator:
        """Create a model evaluator component."""
        implementation = self.components.get(ModelEvaluator, EnhancedModelEvaluator)
        return implementation(config_provider=self.config_provider)

    def create_model_serializer(self) -> ModelSerializer:
        """Create a model serializer component."""
        implementation = self.components.get(ModelSerializer, PickleModelSerializer)
        return implementation(config_provider=self.config_provider)

    def create_predictor(self, eav_manager=None) -> Predictor:
        """
        Create a predictor component.

        Args:
            eav_manager: EAVManager instance. If None, creates a new one.
        """
        implementation = self.components.get(Predictor, StandardPredictor)
        eav_manager = eav_manager or EAVManager(
            templates_path=self.config_provider.config.eav_templates_path
        )
        return implementation(
            config_provider=self.config_provider,
            eav_manager=eav_manager
        )

    def create_complete_pipeline(self) -> Dict[str, Any]:
        """
        Create all pipeline components.

        Returns:
            Dictionary of pipeline components
        """
        # Create shared dependencies
        eav_manager = EAVManager(
            templates_path=self.config_provider.config.eav_templates_path
        )
        reference_manager = ReferenceManager(
            config_path=self.config_provider.config.reference
        )

        # Create pipeline components
        return {
            "data_loader": self.create_data_loader(),
            "data_preprocessor": self.create_data_preprocessor(),
            "feature_engineer": self.create_feature_engineer(eav_manager),
            "model_builder": self.create_model_builder(),
            "model_trainer": self.create_model_trainer(),
            "model_evaluator": self.create_model_evaluator(),
            "model_serializer": self.create_model_serializer(),
            "predictor": self.create_predictor(eav_manager),
            "eav_manager": eav_manager,
            "reference_manager": reference_manager
        }
```

#### 2.4 Create a Pipeline Orchestrator

```python
# nexusml/core/pipeline/orchestrator.py
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from .factory import PipelineFactory
from nexusml.core.config.provider import ConfigurationProvider

class PipelineOrchestrator:
    """Orchestrates the execution of the pipeline."""

    def __init__(self, factory=None, config_provider=None):
        """
        Initialize the pipeline orchestrator.

        Args:
            factory: Pipeline factory. If None, creates a new one.
            config_provider: Configuration provider. If None, uses the default.
        """
        self.config_provider = config_provider or ConfigurationProvider()
        self.factory = factory or PipelineFactory(config_provider=self.config_provider)
        self.pipeline = self.factory.create_complete_pipeline()

    def train_model(self, data_path: Optional[str] = None) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
        """
        Train a model using the pipeline.

        Args:
            data_path: Path to the training data. If None, uses the path from configuration.

        Returns:
            Tuple containing:
            - Trained model
            - Processed DataFrame
            - Dictionary with evaluation metrics
        """
        # Load data
        df = self.pipeline["data_loader"].load_data(data_path)

        # Preprocess data
        df = self.pipeline["data_preprocessor"].preprocess(df)

        # Engineer features
        df = self.pipeline["feature_engineer"].engineer_features(df)

        # Prepare data for training
        X = pd.DataFrame({
            "combined_features": df["combined_text"],
            "service_life": df["service_life"],
        })

        y = df[[
            "category_name",
            "uniformat_code",
            "mcaa_system_category",
            "Equipment_Type",
            "System_Subtype",
        ]]

        # Build model
        model = self.pipeline["model_builder"].build_model()

        # Train model
        model = self.pipeline["model_trainer"].train(model, X, y)

        # Evaluate model
        metrics = self.pipeline["model_evaluator"].evaluate(model, X, y)

        return model, df, metrics

    def save_model(self, model: Any, output_dir: str, model_name: str, metrics: Dict[str, Any]) -> Dict[str, str]:
        """
        Save a trained model.

        Args:
            model: Trained model
            output_dir: Directory to save the model
            model_name: Base name for the model file
            metrics: Evaluation metrics

        Returns:
            Dictionary with paths to saved files
        """
        return self.pipeline["model_serializer"].save(
            model,
            output_dir,
            metadata={
                "model_name": model_name,
                "metrics": metrics
            }
        )

    def predict(self, model: Any, description: str, service_life: float = 0.0, asset_tag: str = "") -> Dict[str, Any]:
        """
        Make predictions with a trained model.

        Args:
            model: Trained model
            description: Text description of the equipment
            service_life: Service life value
            asset_tag: Asset tag for equipment

        Returns:
            Dictionary with prediction results
        """
        return self.pipeline["predictor"].predict(
            model,
            {
                "description": description,
                "service_life": service_life,
                "asset_tag": asset_tag
            }
        )
```

#### 2.5 Update the Main Entry Points

```python
# nexusml/train_model_pipeline.py (partial update)
from nexusml.core.pipeline.orchestrator import PipelineOrchestrator
from nexusml.core.config.provider import ConfigurationProvider

def main():
    """Main function to run the model training pipeline."""
    # Parse command-line arguments
    args = parse_arguments()

    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting equipment classification model training pipeline")

    try:
        # Initialize configuration
        config_provider = ConfigurationProvider()
        config_provider.initialize(args.config)

        # Create pipeline orchestrator
        orchestrator = PipelineOrchestrator(config_provider=config_provider)

        # Train the model
        model, df, metrics = orchestrator.train_model(args.data_path)

        # Save the model
        save_paths = orchestrator.save_model(
            model,
            args.output_dir,
            args.model_name,
            metrics
        )

        # Generate visualizations if requested
        if args.visualize:
            viz_paths = generate_visualizations(
                model,
                df,
                args.output_dir,
                logger,
            )

        # Make a sample prediction
        sample_prediction = orchestrator.predict(
            model,
            "Heat Exchanger for Chilled Water system with Plate and Frame design",
            20.0
        )

        logger.info("Model training pipeline completed successfully")
        logger.info(f"Model saved to: {save_paths['model_path']}")
        logger.info(f"Metadata saved to: {save_paths['metadata_path']}")

    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}", exc_info=True)
        sys.exit(1)
```

## 3. Dependency Injection Improvements

### Current Issues

- Components create their dependencies internally
- Difficult to test components in isolation
- Tight coupling between components
- Limited flexibility for extension or customization

### Proposed Solution: Dependency Injection Container

#### 3.1 Create a Dependency Injection Container

```python
# nexusml/core/di/container.py
from typing import Any, Dict, Optional, Type

class DIContainer:
    """Simple dependency injection container."""

    def __init__(self):
        """Initialize the container."""
        self._services = {}
        self._factories = {}
        self._singletons = {}

    def register(self, interface: Type, implementation: Type) -> None:
        """
        Register a service implementation for an interface.

        Args:
            interface: Interface class
            implementation: Implementation class
        """
        self._services[interface] = implementation

    def register_factory(self, interface: Type, factory) -> None:
        """
        Register a factory function for an interface.

        Args:
            interface: Interface class
            factory: Factory function that creates an instance
        """
        self._factories[interface] = factory

    def register_instance(self, interface: Type, instance: Any) -> None:
        """
        Register a singleton instance for an interface.

        Args:
            interface: Interface class
            instance: Instance to use
        """
        self._singletons[interface] = instance

    def resolve(self, interface: Type) -> Any:
        """
        Resolve an implementation for an interface.

        Args:
            interface: Interface class

        Returns:
            Implementation instance
        """
        # Check if we have a singleton instance
        if interface in self._singletons:
            return self._singletons[interface]

        # Check if we have a factory
        if interface in self._factories:
            return self._factories[interface]()

        # Check if we have a service registration
        if interface in self._services:
            implementation = self._services[interface]
            # Create a new instance
            return implementation()

        # If no registration found, try to create an instance directly
        return interface()
```

#### 3.2 Create a Container Provider

```python
# nexusml/core/di/provider.py
from .container import DIContainer
from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.eav_manager import EAVManager
from nexusml.core.reference.manager import ReferenceManager
from nexusml.core.pipeline.interfaces import (
    DataLoader, DataPreprocessor, FeatureEngineer,
    ModelBuilder, ModelTrainer, ModelEvaluator,
    ModelSerializer, Predictor
)
from nexusml.core.pipeline.components import (
    StandardDataLoader, StandardDataPreprocessor,
    GenericFeatureEngineer, RandomForestModelBuilder,
    StandardModelTrainer, EnhancedModelEvaluator,
    PickleModelSerializer, StandardPredictor
)

class ContainerProvider:
    """Singleton provider for the DI container."""

    _instance = None
    _container = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContainerProvider, cls).__new__(cls)
            cls._instance._container = None
        return cls._instance

    def initialize(self) -> None:
        """Initialize the container with default registrations."""
        container = DIContainer()

        # Register configuration provider
        config_provider = ConfigurationProvider()
        container.register_instance(ConfigurationProvider, config_provider)

        # Register managers
        eav_manager = EAVManager(
            templates_path=config_provider.config.eav_templates_path
        )
        container.register_instance(EAVManager, eav_manager)

        reference_manager = ReferenceManager(
            config_path=config_provider.config.reference
        )
        container.register_instance(ReferenceManager, reference_manager)

        # Register pipeline components
        container.register(DataLoader, StandardDataLoader)
        container.register(DataPreprocessor, StandardDataPreprocessor)
        container.register(FeatureEngineer, GenericFeatureEngineer)
        container.register(ModelBuilder, RandomForestModelBuilder)
        container.register(ModelTrainer, StandardModelTrainer)
        container.register(ModelEvaluator, EnhancedModelEvaluator)
        container.register(ModelSerializer, PickleModelSerializer)
        container.register(Predictor, StandardPredictor)

        self._container = container

    @property
    def container(self) -> DIContainer:
        """Get the DI container."""
        if self._container is None:
            self.initialize()
        return self._container

    def reset(self) -> None:
        """Reset the container."""
        self._container = None
```

#### 3.3 Update Components to Use Dependency Injection

Example for `EquipmentClassifier`:

```python
# nexusml/core/model.py (partial update)
from nexusml.core.di.provider import ContainerProvider
from nexusml.core.pipeline.interfaces import (
    ModelBuilder, ModelTrainer, ModelEvaluator,
    ModelSerializer, Predictor
)

class EquipmentClassifier:
    """
    Comprehensive equipment classifier with EAV integration.
    """

    def __init__(
        self,
        model=None,
        feature_engineer=None,
        eav_manager=None,
        sampling_strategy="direct",
        container_provider=None,
    ):
        """
        Initialize the equipment classifier.

        Args:
            model: Trained ML model (if None, needs to be trained)
            feature_engineer: Feature engineering transformer
            eav_manager: EAV manager for attribute templates
            sampling_strategy: Strategy for handling class imbalance
            container_provider: DI container provider. If None, uses the default.
        """
        self.model = model
        self.container_provider = container_provider or ContainerProvider()

        # Resolve dependencies
        container = self.container_provider.container
        self.feature_engineer = feature_engineer or container.resolve(FeatureEngineer)
        self.eav_manager = eav_manager or container.resolve(EAVManager)
        self.sampling_strategy = sampling_strategy

        # Resolve pipeline components
        self.model_builder = container.resolve(ModelBuilder)
        self.model_trainer = container.resolve(ModelTrainer)
        self.model_evaluator = container.resolve(ModelEvaluator)
        self.model_serializer = container.resolve(ModelSerializer)
        self.predictor = container.resolve(Predictor)

    def train(
        self,
        data_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Train the equipment classifier.

        Args:
            data_path: Path to the training data
            feature_config_path: Path to the feature configuration
            **kwargs: Additional parameters for training
        """
        # Get data loader and preprocessor
        container = self.container_provider.container
        data_loader = container.resolve(DataLoader)
        data_preprocessor = container.resolve(DataPreprocessor)

        # Load and preprocess data
        df = data_loader.load_data(data_path)
        df = data_preprocessor.preprocess(df)

        # Apply feature engineering
        df = self.feature_engineer.engineer_features(df)
        self.df = df

        # Prepare data for training
        X = pd.DataFrame({
            "combined_features": df["combined_text"],
            "service_life": df["service_life"],
        })

        y = df[[
            "category_name",
            "uniformat_code",
            "mcaa_system_category",
            "Equipment_Type",
            "System_Subtype",
        ]]

        # Build model
        self.model = self.model_builder.build_model()

        # Train model
        self.model = self.model_trainer.train(self.model, X, y, **kwargs)
```

## 4. Implementation Plan

### Phase 1: Configuration Consolidation

1. Create the `NexusMLConfig` class and `ConfigurationProvider`
2. Create a unified configuration file template
3. Update components to use the new configuration system
4. Add validation for configuration values
5. Add tests for the configuration system

### Phase 2: Pipeline Refactoring

1. Define interfaces for pipeline components
2. Implement concrete pipeline components
3. Create the pipeline factory and orchestrator
4. Update entry points to use the new pipeline
5. Add tests for the pipeline components

### Phase 3: Dependency Injection

1. Create the DI container and provider
2. Update components to use dependency injection
3. Add tests for the DI system
4. Update documentation

### Phase 4: Integration and Testing

1. Integrate all changes
2. Add comprehensive tests
3. Update documentation
4. Create examples of customizing the pipeline

## 5. Benefits of the Refactoring

1. **Improved SOLID Principles Adherence**

   - Single Responsibility: Each component has a clear, focused responsibility
   - Open/Closed: Components can be extended without modification
   - Liskov Substitution: Interfaces ensure substitutability
   - Interface Segregation: Focused interfaces for specific responsibilities
   - Dependency Inversion: Dependencies are injected rather than created
     internally

2. **Enhanced Testability**

   - Components can be tested in isolation
   - Dependencies can be mocked or stubbed
   - Configuration can be controlled in tests

3. **Better Maintainability**

   - Clear separation of concerns
   - Consistent interfaces
   - Reduced duplication
   - Centralized configuration

4. **Increased Flexibility**

   - Components can be swapped or extended
   - Pipeline can be customized
   - Configuration is centralized and validated

5. **Improved Developer Experience**
   - Clear component responsibilities
   - Consistent interfaces
   - Dependency management through DI
   - Centralized configuration
