"""
Base Feature Engineering Module

This module provides base implementations for feature engineering components in the NexusML suite.
These base classes implement common functionality and provide default behavior
where appropriate, following the Template Method pattern.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from nexusml.src.features.interfaces import (
    ColumnTransformer,
    ConfigurableTransformer,
    FeatureEngineer,
    FeatureTransformer,
)


class BaseFeatureTransformer(BaseEstimator, TransformerMixin, FeatureTransformer):
    """
    Base implementation of the FeatureTransformer interface.

    Provides common functionality for all feature transformers.
    """

    def __init__(self, name: str = "BaseFeatureTransformer"):
        """
        Initialize the feature transformer.

        Args:
            name: Name of the transformer.
        """
        self.name = name
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureTransformer":
        """
        Fit the transformer to the data.

        This base implementation simply marks the transformer as fitted.
        Subclasses should override this method to provide specific fitting logic.

        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).

        Returns:
            Self for method chaining.
        """
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted transformer.

        This base implementation returns the input data unchanged.
        Subclasses should override this method to provide specific transformation logic.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If the transformer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                f"Transformer '{self.name}' must be fitted before transform can be called"
            )

        return X.copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit the transformer to the data and transform it.

        Args:
            X: Input DataFrame to fit and transform.
            y: Target values (optional).

        Returns:
            Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features produced by this transformer.

        This base implementation returns an empty list.
        Subclasses should override this method to provide specific feature names.

        Returns:
            List of feature names.
        """
        return []


class BaseColumnTransformer(BaseFeatureTransformer, ColumnTransformer):
    """
    Base implementation of the ColumnTransformer interface.

    Provides common functionality for column-specific transformers.
    """

    def __init__(
        self,
        input_columns: List[str],
        output_columns: Optional[List[str]] = None,
        name: str = "BaseColumnTransformer",
    ):
        """
        Initialize the column transformer.

        Args:
            input_columns: Names of the input columns required by this transformer.
            output_columns: Names of the output columns produced by this transformer.
                If None, uses the input columns.
            name: Name of the transformer.
        """
        super().__init__(name)
        self.input_columns = input_columns
        self.output_columns = output_columns or input_columns

    def get_input_columns(self) -> List[str]:
        """
        Get the names of the input columns required by this transformer.

        Returns:
            List of input column names.
        """
        return self.input_columns

    def get_output_columns(self) -> List[str]:
        """
        Get the names of the output columns produced by this transformer.

        Returns:
            List of output column names.
        """
        return self.output_columns

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features produced by this transformer.

        Returns:
            List of feature names.
        """
        return self.get_output_columns()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted transformer.

        This implementation checks if the required input columns exist and
        calls the _transform method to perform the actual transformation.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If the transformer has not been fitted or required columns are missing.
        """
        if not self._is_fitted:
            raise ValueError(
                f"Transformer '{self.name}' must be fitted before transform can be called"
            )

        # Create a copy of the DataFrame to avoid modifying the original
        X = X.copy()

        # Check if all required input columns exist
        missing_columns = [col for col in self.input_columns if col not in X.columns]
        if missing_columns:
            # Handle missing columns according to the transformer's behavior
            return self._handle_missing_columns(X, missing_columns)

        # Perform the transformation
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Perform the actual transformation on the input data.

        This method should be implemented by subclasses to provide specific transformation logic.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        return X

    def _handle_missing_columns(
        self, X: pd.DataFrame, missing_columns: List[str]
    ) -> pd.DataFrame:
        """
        Handle missing input columns.

        This method should be implemented by subclasses to provide specific handling
        for missing input columns. The default implementation raises a ValueError.

        Args:
            X: Input DataFrame.
            missing_columns: List of missing column names.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        raise ValueError(
            f"Required columns {missing_columns} not found in input DataFrame"
        )


class BaseConfigurableTransformer(BaseFeatureTransformer, ConfigurableTransformer):
    """
    Base implementation of the ConfigurableTransformer interface.

    Provides common functionality for configurable transformers.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: str = "BaseConfigurableTransformer",
    ):
        """
        Initialize the configurable transformer.

        Args:
            config: Configuration dictionary. If None, uses an empty dictionary.
            name: Name of the transformer.
        """
        super().__init__(name)
        self.config = config or {}

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this transformer.

        Returns:
            Dictionary containing the configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration of this transformer.

        Args:
            config: Dictionary containing the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Validate the configuration
        self._validate_config(config)

        # Set the configuration
        self.config = config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration.

        This method should be implemented by subclasses to provide specific validation
        for the configuration. The default implementation does nothing.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        pass


class BaseFeatureEngineer(FeatureEngineer):
    """
    Base implementation of the FeatureEngineer interface.

    Provides common functionality for feature engineers.
    """

    def __init__(
        self,
        transformers: Optional[List[FeatureTransformer]] = None,
        name: str = "BaseFeatureEngineer",
    ):
        """
        Initialize the feature engineer.

        Args:
            transformers: List of transformers to use. If None, uses an empty list.
            name: Name of the feature engineer.
        """
        self.name = name
        self.transformers = transformers or []
        self._is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureEngineer":
        """
        Fit the feature engineer to the data.

        This method fits each transformer in sequence.

        Args:
            X: Input DataFrame to fit to.
            y: Target values (optional).

        Returns:
            Self for method chaining.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        X_transformed = X.copy()

        # Fit each transformer in sequence
        for transformer in self.transformers:
            X_transformed = transformer.fit_transform(X_transformed, y)

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data using the fitted feature engineer.

        This method applies each transformer in sequence.

        Args:
            X: Input DataFrame to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            ValueError: If the feature engineer has not been fitted.
        """
        if not self._is_fitted:
            raise ValueError(
                f"Feature engineer '{self.name}' must be fitted before transform can be called"
            )

        # Create a copy of the DataFrame to avoid modifying the original
        X_transformed = X.copy()

        # Apply each transformer in sequence
        for transformer in self.transformers:
            X_transformed = transformer.transform(X_transformed)

        return X_transformed

    def get_transformers(self) -> List[FeatureTransformer]:
        """
        Get the transformers used by this feature engineer.

        Returns:
            List of transformers.
        """
        return self.transformers

    def add_transformer(self, transformer: FeatureTransformer) -> None:
        """
        Add a transformer to this feature engineer.

        Args:
            transformer: Transformer to add.
        """
        self.transformers.append(transformer)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features produced by this feature engineer.

        Returns:
            List of feature names.
        """
        # Get the feature names from the last transformer
        if self.transformers:
            return self.transformers[-1].get_feature_names()

        return []


class BaseConfigDrivenFeatureEngineer(BaseFeatureEngineer, ConfigurableTransformer):
    """
    Base implementation of the ConfigDrivenFeatureEngineer interface.

    Provides common functionality for configuration-driven feature engineers.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        name: str = "BaseConfigDrivenFeatureEngineer",
    ):
        """
        Initialize the configuration-driven feature engineer.

        Args:
            config: Configuration dictionary. If None, uses an empty dictionary.
            name: Name of the feature engineer.
        """
        super().__init__([], name)
        self.config = config or {}

        # Create transformers from the configuration
        self.transformers = self.create_transformers_from_config()

    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this feature engineer.

        Returns:
            Dictionary containing the configuration.
        """
        return self.config

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set the configuration of this feature engineer.

        Args:
            config: Dictionary containing the configuration.

        Raises:
            ValueError: If the configuration is invalid.
        """
        # Validate the configuration
        self._validate_config(config)

        # Set the configuration
        self.config = config

        # Create transformers from the new configuration
        self.transformers = self.create_transformers_from_config()

        # Reset the fitted state
        self._is_fitted = False

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the configuration.

        This method should be implemented by subclasses to provide specific validation
        for the configuration. The default implementation does nothing.

        Args:
            config: Configuration to validate.

        Raises:
            ValueError: If the configuration is invalid.
        """
        pass

    def create_transformers_from_config(self) -> List[FeatureTransformer]:
        """
        Create transformers from the configuration.

        This method should be implemented by subclasses to provide specific logic
        for creating transformers from the configuration. The default implementation
        returns an empty list.

        Returns:
            List of transformers created from the configuration.
        """
        return []
