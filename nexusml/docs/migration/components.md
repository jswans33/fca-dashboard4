# NexusML Component Migration Guide

## Introduction

This document provides guidance on migrating components from the old NexusML
architecture to the new refactored architecture. It explains how to update
components to use the new interfaces, how to integrate with the dependency
injection system, how to use the adapter pattern for backward compatibility, and
how to test migrated components.

The new architecture follows the SOLID principles, particularly the Interface
Segregation Principle (ISP) and the Dependency Inversion Principle (DIP), making
the system more testable, maintainable, and extensible.

## Component Interfaces

The new architecture defines clear interfaces for all pipeline components in the
`nexusml.core.pipeline.interfaces` module. Each interface follows the Interface
Segregation Principle, defining a minimal set of methods that components must
implement.

### Key Interfaces

- **DataLoader**: Responsible for loading data from various sources
- **DataPreprocessor**: Responsible for cleaning and preparing data
- **FeatureEngineer**: Responsible for transforming raw data into features
- **ModelBuilder**: Responsible for creating and configuring models
- **ModelTrainer**: Responsible for training models
- **ModelEvaluator**: Responsible for evaluating models
- **ModelSerializer**: Responsible for saving and loading models
- **Predictor**: Responsible for making predictions

## Migration Process

The migration process consists of the following steps:

1. **Identify Components**: Identify the components in your code that need to be
   migrated
2. **Create New Components**: Create new components that implement the new
   interfaces
3. **Use Adapters**: Use adapters to maintain backward compatibility
4. **Update Dependencies**: Update dependencies to use dependency injection
5. **Register Components**: Register components with the component registry
6. **Test Components**: Test the migrated components

### Step 1: Identify Components

First, identify the components in your code that need to be migrated. This may
include:

- Data loading components
- Data preprocessing components
- Feature engineering components
- Model building components
- Model training components
- Model evaluation components
- Model serialization components
- Prediction components

### Step 2: Create New Components

Create new components that implement the new interfaces. Each component should
implement the appropriate interface from the `nexusml.core.pipeline.interfaces`
module.

**Example: Data Loader**

```python
from nexusml.core.pipeline.interfaces import DataLoader
import pandas as pd
from typing import Dict, Any, Optional

class CSVDataLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize a new CSVDataLoader.

        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file (if None, uses self.file_path)
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the CSV file cannot be found
            ValueError: If the CSV format is invalid
        """
        path = data_path or self.file_path
        if path is None:
            raise ValueError("No data path provided")

        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration
        """
        return {"file_path": self.file_path}
```

**Example: Feature Engineer**

```python
from nexusml.core.pipeline.interfaces import FeatureEngineer
import pandas as pd
from typing import List, Optional

class TextFeatureEngineer(FeatureEngineer):
    """Feature engineer for text data."""

    def __init__(self, text_columns: Optional[List[str]] = None, combined_column: str = "combined_text"):
        """
        Initialize a new TextFeatureEngineer.

        Args:
            text_columns: List of text columns to combine
            combined_column: Name of the combined text column
        """
        self.text_columns = text_columns or ["description"]
        self.combined_column = combined_column
        self._vectorizer = None

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features
            **kwargs: Additional arguments for feature engineering

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If features cannot be engineered
        """
        # Create a copy of the input data
        result = data.copy()

        # Combine text columns
        result[self.combined_column] = result[self.text_columns].fillna("").agg(" ".join, axis=1)

        return result

    def fit(self, data: pd.DataFrame, **kwargs) -> "TextFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the feature engineer cannot be fit to the data
        """
        # Create a copy of the input data
        result = data.copy()

        # Combine text columns
        result[self.combined_column] = result[self.text_columns].fillna("").agg(" ".join, axis=1)

        # Fit vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(**kwargs.get("vectorizer_params", {}))
        self._vectorizer.fit(result[self.combined_column])

        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If the data cannot be transformed
        """
        # Create a copy of the input data
        result = data.copy()

        # Combine text columns
        result[self.combined_column] = result[self.text_columns].fillna("").agg(" ".join, axis=1)

        # Transform data
        if self._vectorizer is not None:
            features = self._vectorizer.transform(result[self.combined_column])
            feature_names = self._vectorizer.get_feature_names_out()
            feature_df = pd.DataFrame(features.toarray(), columns=feature_names)
            result = pd.concat([result, feature_df], axis=1)

        return result
```

### Step 3: Use Adapters

Use adapters to maintain backward compatibility. Adapters allow old components
to be used in the new architecture and new components to be used in the old
architecture.

**Example: Adapter for Old Data Loader**

```python
from nexusml.core.pipeline.interfaces import DataLoader
import pandas as pd
from typing import Dict, Any, Optional

class OldDataLoaderAdapter(DataLoader):
    """Adapter for old data loader."""

    def __init__(self, old_loader):
        """
        Initialize a new OldDataLoaderAdapter.

        Args:
            old_loader: Old data loader to adapt
        """
        self.old_loader = old_loader

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data using the old data loader.

        Args:
            data_path: Path to the data file
            **kwargs: Additional arguments for data loading

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the data file cannot be found
            ValueError: If the data format is invalid
        """
        # Adapt old interface to new interface
        return self.old_loader.load(data_path)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration
        """
        # Provide configuration for old component
        return {}
```

**Example: Adapter for New Data Loader**

```python
class NewDataLoaderAdapter:
    """Adapter for new data loader."""

    def __init__(self, new_loader):
        """
        Initialize a new NewDataLoaderAdapter.

        Args:
            new_loader: New data loader to adapt
        """
        self.new_loader = new_loader

    def load(self, path):
        """
        Load data using the new data loader.

        Args:
            path: Path to the data file

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the data file cannot be found
            ValueError: If the data format is invalid
        """
        # Adapt new interface to old interface
        return self.new_loader.load_data(path)
```

### Step 4: Update Dependencies

Update dependencies to use dependency injection. This involves:

1. Identifying dependencies in your components
2. Passing dependencies through the constructor
3. Using the dependency injection container to resolve dependencies

**Example: Component with Dependencies**

```python
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor
import pandas as pd
from typing import Optional

class DataProcessor:
    """Component that processes data."""

    def __init__(self, data_loader: DataLoader, preprocessor: DataPreprocessor):
        """
        Initialize a new DataProcessor.

        Args:
            data_loader: Data loader to use
            preprocessor: Data preprocessor to use
        """
        self.data_loader = data_loader
        self.preprocessor = preprocessor

    def process_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Process data from the specified path.

        Args:
            data_path: Path to the data file
            **kwargs: Additional arguments for processing

        Returns:
            Processed DataFrame

        Raises:
            FileNotFoundError: If the data file cannot be found
            ValueError: If the data format is invalid
        """
        # Load data
        data = self.data_loader.load_data(data_path, **kwargs)

        # Preprocess data
        processed_data = self.preprocessor.preprocess(data, **kwargs)

        return processed_data
```

### Step 5: Register Components

Register components with the component registry. This involves:

1. Creating a component registry
2. Registering components with the registry
3. Setting default implementations

**Example: Registering Components**

```python
from nexusml.core.pipeline.registry import ComponentRegistry
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor, FeatureEngineer, ModelBuilder

# Create a registry
registry = ComponentRegistry()

# Register components
registry.register(DataLoader, "csv", CSVDataLoader)
registry.register(DataPreprocessor, "standard", StandardPreprocessor)
registry.register(FeatureEngineer, "text", TextFeatureEngineer)
registry.register(ModelBuilder, "random_forest", RandomForestModelBuilder)

# Set default implementations
registry.set_default_implementation(DataLoader, "csv")
registry.set_default_implementation(DataPreprocessor, "standard")
registry.set_default_implementation(FeatureEngineer, "text")
registry.set_default_implementation(ModelBuilder, "random_forest")
```

### Step 6: Test Components

Test the migrated components. This involves:

1. Writing unit tests for individual components
2. Writing integration tests for component interactions
3. Verifying that the components work as expected

**Example: Unit Test for Data Loader**

```python
import pytest
import pandas as pd
from nexusml.core.pipeline.interfaces import DataLoader
from nexusml.core.pipeline.components.data_loader import CSVDataLoader

def test_csv_data_loader():
    """Test the CSVDataLoader."""
    # Create a data loader
    data_loader = CSVDataLoader(file_path="tests/data/test_data.csv")

    # Load data
    data = data_loader.load_data()

    # Verify that data is loaded correctly
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert "description" in data.columns

def test_csv_data_loader_with_path():
    """Test the CSVDataLoader with a specific path."""
    # Create a data loader
    data_loader = CSVDataLoader()

    # Load data with a specific path
    data = data_loader.load_data(data_path="tests/data/test_data.csv")

    # Verify that data is loaded correctly
    assert isinstance(data, pd.DataFrame)
    assert len(data) > 0
    assert "description" in data.columns

def test_csv_data_loader_with_missing_file():
    """Test the CSVDataLoader with a missing file."""
    # Create a data loader
    data_loader = CSVDataLoader(file_path="tests/data/missing_file.csv")

    # Verify that FileNotFoundError is raised
    with pytest.raises(FileNotFoundError):
        data_loader.load_data()

def test_csv_data_loader_with_invalid_file():
    """Test the CSVDataLoader with an invalid file."""
    # Create a data loader
    data_loader = CSVDataLoader(file_path="tests/data/invalid_file.csv")

    # Verify that ValueError is raised
    with pytest.raises(ValueError):
        data_loader.load_data()
```

**Example: Integration Test for Component Interaction**

```python
import pytest
import pandas as pd
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor
from nexusml.core.pipeline.components.data_loader import CSVDataLoader
from nexusml.core.pipeline.components.data_preprocessor import StandardPreprocessor

def test_data_loader_with_preprocessor():
    """Test the interaction between DataLoader and DataPreprocessor."""
    # Create components
    data_loader = CSVDataLoader(file_path="tests/data/test_data.csv")
    preprocessor = StandardPreprocessor()

    # Load and preprocess data
    data = data_loader.load_data()
    processed_data = preprocessor.preprocess(data)

    # Verify that data is processed correctly
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) > 0
    assert "description" in processed_data.columns
    assert preprocessor.verify_required_columns(processed_data) is not None
```

## Common Migration Scenarios

### Scenario 1: Simple Component Migration

For simple components with no dependencies, the migration process is
straightforward:

1. Identify the component interface
2. Implement the interface
3. Register the component with the registry

**Example: Migrating a Simple Data Loader**

```python
# Old component
class OldDataLoader:
    def load(self, path):
        # Old implementation
        import pandas as pd
        return pd.read_csv(path)

# New component
from nexusml.core.pipeline.interfaces import DataLoader
import pandas as pd
from typing import Dict, Any, Optional

class NewDataLoader(DataLoader):
    """Data loader for CSV files."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize a new NewDataLoader.

        Args:
            file_path: Path to the CSV file
        """
        self.file_path = file_path

    def load_data(self, data_path: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file.

        Args:
            data_path: Path to the CSV file (if None, uses self.file_path)
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the CSV file cannot be found
            ValueError: If the CSV format is invalid
        """
        path = data_path or self.file_path
        if path is None:
            raise ValueError("No data path provided")

        try:
            return pd.read_csv(path, **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {path}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the data loader.

        Returns:
            Dictionary containing the configuration
        """
        return {"file_path": self.file_path}
```

### Scenario 2: Component with Dependencies

For components with dependencies, the migration process involves:

1. Identifying the component interface
2. Identifying the dependencies
3. Implementing the interface with dependencies
4. Registering the component with the registry

**Example: Migrating a Component with Dependencies**

```python
# Old component
class OldFeatureEngineer:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    def engineer_features(self, data):
        # Old implementation
        processed_data = self.preprocessor.preprocess(data)
        # Engineer features
        return processed_data

# New component
from nexusml.core.pipeline.interfaces import FeatureEngineer, DataPreprocessor
import pandas as pd
from typing import Optional

class NewFeatureEngineer(FeatureEngineer):
    """Feature engineer with dependencies."""

    def __init__(self, preprocessor: DataPreprocessor):
        """
        Initialize a new NewFeatureEngineer.

        Args:
            preprocessor: Data preprocessor to use
        """
        self.preprocessor = preprocessor

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features
            **kwargs: Additional arguments for feature engineering

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If features cannot be engineered
        """
        # Preprocess data
        processed_data = self.preprocessor.preprocess(data, **kwargs)

        # Engineer features
        # ...

        return processed_data

    def fit(self, data: pd.DataFrame, **kwargs) -> "NewFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the feature engineer cannot be fit to the data
        """
        # Preprocess data
        processed_data = self.preprocessor.preprocess(data, **kwargs)

        # Fit feature engineer
        # ...

        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If the data cannot be transformed
        """
        # Preprocess data
        processed_data = self.preprocessor.preprocess(data, **kwargs)

        # Transform data
        # ...

        return processed_data
```

### Scenario 3: Component with Complex Logic

For components with complex logic, the migration process involves:

1. Identifying the component interface
2. Breaking down the complex logic into smaller, more focused methods
3. Implementing the interface with the refactored logic
4. Registering the component with the registry

**Example: Migrating a Component with Complex Logic**

```python
# Old component
class OldModelBuilder:
    def build_model(self, **kwargs):
        # Old implementation with complex logic
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Create pipeline steps
        steps = []

        # Add preprocessing steps
        steps.append(("scaler", StandardScaler()))

        # Add feature extraction steps
        steps.append(("vectorizer", TfidfVectorizer()))

        # Add classifier
        steps.append(("classifier", RandomForestClassifier(**kwargs)))

        # Create pipeline
        pipeline = Pipeline(steps)

        return pipeline

# New component
from nexusml.core.pipeline.interfaces import ModelBuilder
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Optional

class NewModelBuilder(ModelBuilder):
    """Model builder with refactored logic."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None):
        """
        Initialize a new NewModelBuilder.

        Args:
            n_estimators: Number of estimators for the random forest
            max_depth: Maximum depth of the trees
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def build_model(self, **kwargs) -> Pipeline:
        """
        Build a machine learning model.

        Args:
            **kwargs: Configuration parameters for the model

        Returns:
            Configured model pipeline

        Raises:
            ValueError: If the model cannot be built with the given parameters
        """
        # Create pipeline steps
        steps = []

        # Add preprocessing steps
        steps.extend(self._create_preprocessing_steps(**kwargs))

        # Add feature extraction steps
        steps.extend(self._create_feature_extraction_steps(**kwargs))

        # Add classifier
        steps.extend(self._create_classifier_steps(**kwargs))

        # Create pipeline
        pipeline = Pipeline(steps)

        return pipeline

    def _create_preprocessing_steps(self, **kwargs) -> list:
        """
        Create preprocessing steps for the pipeline.

        Args:
            **kwargs: Configuration parameters for preprocessing

        Returns:
            List of preprocessing steps
        """
        from sklearn.preprocessing import StandardScaler

        steps = []
        steps.append(("scaler", StandardScaler()))
        return steps

    def _create_feature_extraction_steps(self, **kwargs) -> list:
        """
        Create feature extraction steps for the pipeline.

        Args:
            **kwargs: Configuration parameters for feature extraction

        Returns:
            List of feature extraction steps
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        steps = []
        steps.append(("vectorizer", TfidfVectorizer(**kwargs.get("vectorizer_params", {}))))
        return steps

    def _create_classifier_steps(self, **kwargs) -> list:
        """
        Create classifier steps for the pipeline.

        Args:
            **kwargs: Configuration parameters for the classifier

        Returns:
            List of classifier steps
        """
        from sklearn.ensemble import RandomForestClassifier

        steps = []
        steps.append(("classifier", RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", self.n_estimators),
            max_depth=kwargs.get("max_depth", self.max_depth),
            **kwargs.get("classifier_params", {})
        )))
        return steps

    def optimize_hyperparameters(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model.

        Args:
            model: Model pipeline to optimize
            x_train: Training features
            y_train: Training targets
            **kwargs: Additional arguments for hyperparameter optimization

        Returns:
            Optimized model pipeline

        Raises:
            ValueError: If hyperparameters cannot be optimized
        """
        from sklearn.model_selection import GridSearchCV

        # Define parameter grid
        param_grid = kwargs.get("param_grid", {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20, 30]
        })

        # Create grid search
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=kwargs.get("cv", 5),
            scoring=kwargs.get("scoring", "accuracy"),
            n_jobs=kwargs.get("n_jobs", -1)
        )

        # Fit grid search
        grid_search.fit(x_train, y_train)

        # Return best model
        return grid_search.best_estimator_
```

### Scenario 4: Component with State

For components that maintain state, the migration process involves:

1. Identifying the component interface
2. Identifying the state that needs to be maintained
3. Implementing the interface with proper state management
4. Registering the component with the registry

**Example: Migrating a Component with State**

```python
# Old component
class OldFeatureEngineer:
    def __init__(self):
        self.vectorizer = None
        self.is_fitted = False

    def fit(self, data):
        # Old implementation
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(data["description"])
        self.is_fitted = True
        return self

    def transform(self, data):
        # Old implementation
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted")
        features = self.vectorizer.transform(data["description"])
        return features

# New component
from nexusml.core.pipeline.interfaces import FeatureEngineer
import pandas as pd
from typing import List, Optional

class NewFeatureEngineer(FeatureEngineer):
    """Feature engineer with state."""

    def __init__(self, text_column: str = "description"):
        """
        Initialize a new NewFeatureEngineer.

        Args:
            text_column: Text column to use for feature engineering
        """
        self.text_column = text_column
        self._vectorizer = None
        self._is_fitted = False

    def engineer_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Engineer features from the input data.

        Args:
            data: Input DataFrame with raw features
            **kwargs: Additional arguments for feature engineering

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If features cannot be engineered
        """
        # Check if fitted
        if not self._is_fitted:
            # Fit and transform
            return self.fit(data, **kwargs).transform(data, **kwargs)
        else:
            # Transform only
            return self.transform(data, **kwargs)

    def fit(self, data: pd.DataFrame, **kwargs) -> "NewFeatureEngineer":
        """
        Fit the feature engineer to the input data.

        Args:
            data: Input DataFrame to fit to
            **kwargs: Additional arguments for fitting

        Returns:
            Self for method chaining

        Raises:
            ValueError: If the feature engineer cannot be fit to the data
        """
        # Create vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(**kwargs.get("vectorizer_params", {}))

        # Fit vectorizer
        self._vectorizer.fit(data[self.text_column].fillna(""))
        self._is_fitted = True

        return self

    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        Args:
            data: Input DataFrame to transform
            **kwargs: Additional arguments for transformation

        Returns:
            Transformed DataFrame

        Raises:
            ValueError: If the data cannot be transformed
        """
        # Check if fitted
        if not self._is_fitted:
            raise ValueError("Feature engineer not fitted")

        # Create a copy of the input data
        result = data.copy()

        # Transform data
        features = self._vectorizer.transform(result[self.text_column].fillna(""))
        feature_names = self._vectorizer.get_feature_names_out()
        feature_df = pd.DataFrame(features.toarray(), columns=feature_names)

        # Concatenate with original data
        result = pd.concat([result, feature_df], axis=1)

        return result
```

## Troubleshooting

### Missing Dependencies

If you're missing dependencies when creating components, you can use the
dependency injection container to resolve them.

```python
from nexusml.core.di.container import DIContainer
from nexusml.core.pipeline.interfaces import DataLoader, DataPreprocessor

# Create a container
container = DIContainer()

# Register dependencies
container.register(DataLoader, CSVDataLoader)
container.register(DataPreprocessor, StandardPreprocessor)

# Resolve dependencies
data_loader = container.resolve(DataLoader)
preprocessor = container.resolve(DataPreprocessor)

# Create component with dependencies
component = ComponentWithDependencies(data_loader, preprocessor)
```

### Interface Compatibility

If you're having trouble implementing an interface, check the interface
definition to ensure you're implementing all required methods.

```python
from nexusml.core.pipeline.interfaces import DataLoader
import inspect

# Check interface methods
interface_methods = [method for method in dir(DataLoader) if not method.startswith("_")]
print(f"Interface methods: {interface_methods}")

# Check if your component implements all interface methods
component_methods = [method for method in dir(YourComponent) if not method.startswith("_")]
missing_methods = set(interface_methods) - set(component_methods)
if missing_methods:
    print(f"Missing methods: {missing_methods}")
```

### Adapter Issues

If you're having trouble with adapters, check that you're correctly adapting
between the old and new interfaces.

```python
# Check old interface
old_component = OldComponent()
old_methods = [method for method in dir(old_component) if not method.startswith("_") and callable(getattr(old_component, method))]
print(f"Old methods: {old_methods}")

# Check new interface
from nexusml.core.pipeline.interfaces import DataLoader
new_methods = [method for method in dir(DataLoader) if not method.startswith("_")]
print(f"New methods: {new_methods}")

# Create adapter
adapter = OldComponentAdapter(old_component)

# Check adapter methods
adapter_methods = [method for method in dir(adapter) if not method.startswith("_") and callable(getattr(adapter, method))]
print(f"Adapter methods: {adapter_methods}")

# Check if adapter implements all new interface methods
missing_methods = set(new_methods) - set(adapter_methods)
if missing_methods:
    print(f"Missing methods: {missing_methods}")
```

### Testing Issues

If you're having trouble with tests, check that you're correctly setting up the
test environment.

```python
import pytest
import pandas as pd
from nexusml.core.pipeline.interfaces import DataLoader
from nexusml.core.pipeline.components.data_loader import CSVDataLoader

# Create a test fixture for the data loader
@pytest.fixture
def data_loader():
    """Create a data loader for testing."""
    return CSVDataLoader(file_path="tests/data/test_data.csv")

# Create a test fixture for test data
@pytest.fixture
def test_data():
    """Create test data for testing."""
    return pd.DataFrame({
        "description": ["Test description 1", "Test description 2"],
        "service_life": [10.0, 15.0]
    })

# Test the data loader
def test_data_loader(data_loader, test_data):
    """Test the data loader."""
    # Mock the load_data method to return test data
    data_loader.load_data = lambda *args, **kwargs: test_data

    # Load data
    data = data_loader.load_data()

    # Verify that data is loaded correctly
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 2
    assert "description" in data.columns
    assert "service_life" in data.columns
```

## Conclusion

Migrating components from the old NexusML architecture to the new refactored
architecture is a gradual process that can be done incrementally while
maintaining backward compatibility. By following the migration process outlined
in this document and using the provided examples, you can update your components
to use the new interfaces, integrate with the dependency injection system, and
maintain backward compatibility through adapters.

For more information about the new architecture, see the
[Pipeline Architecture](../architecture/pipeline.md) and
[Dependency Injection](../architecture/dependency_injection.md) documentation.
