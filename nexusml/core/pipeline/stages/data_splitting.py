"""
Data Splitting Stage Module

This module provides implementations of the DataSplittingStage interface for
splitting data into training and testing sets.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from nexusml.config.manager import ConfigurationManager
from nexusml.core.pipeline.context import PipelineContext
from nexusml.core.pipeline.stages.base import BaseDataSplittingStage


class RandomSplittingStage(BaseDataSplittingStage):
    """
    Implementation of DataSplittingStage for random data splitting.
    """

    def __init__(
        self,
        name: str = "RandomSplitting",
        description: str = "Splits data randomly",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the random splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading splitting configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("engineered_data") or context.has("data")

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data randomly.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).
        """
        # Get feature columns (all columns except target columns)
        feature_columns = [col for col in data.columns if col not in target_columns]

        # Extract features and targets
        x = data[feature_columns]
        y = data[target_columns]

        # Get split parameters
        test_size = kwargs.get("test_size", self.config.get("test_size", 0.3))
        random_state = kwargs.get(
            "random_state", self.config.get("random_state", 42)
        )
        shuffle = kwargs.get("shuffle", self.config.get("shuffle", True))
        
        # Remove target_columns from kwargs to avoid duplicate parameter
        kwargs_copy = kwargs.copy()
        if 'target_columns' in kwargs_copy:
            del kwargs_copy['target_columns']
        if 'test_size' in kwargs_copy:
            del kwargs_copy['test_size']
        if 'random_state' in kwargs_copy:
            del kwargs_copy['random_state']
        if 'shuffle' in kwargs_copy:
            del kwargs_copy['shuffle']

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, shuffle=shuffle, **kwargs_copy
        )

        return x_train, x_test, y_train, y_test


class StratifiedSplittingStage(BaseDataSplittingStage):
    """
    Implementation of DataSplittingStage for stratified data splitting.
    """

    def __init__(
        self,
        name: str = "StratifiedSplitting",
        description: str = "Splits data with stratification",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the stratified splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading splitting configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("engineered_data") or context.has("data")

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data with stratification.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).

        Raises:
            ValueError: If stratification column is not specified or not found.
        """
        # Get feature columns (all columns except target columns)
        feature_columns = [col for col in data.columns if col not in target_columns]

        # Extract features and targets
        x = data[feature_columns]
        y = data[target_columns]

        # Get split parameters
        test_size = kwargs.get("test_size", self.config.get("test_size", 0.3))
        random_state = kwargs.get(
            "random_state", self.config.get("random_state", 42)
        )
        stratify_column = kwargs.get(
            "stratify_column", self.config.get("stratify_column")
        )

        # Get stratification values
        if stratify_column is None:
            # If no stratification column is specified, use the first target column
            if len(target_columns) > 0:
                stratify = y[target_columns[0]]
            else:
                raise ValueError("No stratification column specified")
        elif stratify_column in target_columns:
            # If stratification column is a target column, use it
            stratify = y[stratify_column]
        elif stratify_column in feature_columns:
            # If stratification column is a feature column, use it
            stratify = x[stratify_column]
        else:
            raise ValueError(f"Stratification column '{stratify_column}' not found")

        # Split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        return x_train, x_test, y_train, y_test


class TimeSeriesSplittingStage(BaseDataSplittingStage):
    """
    Implementation of DataSplittingStage for time series data splitting.
    """

    def __init__(
        self,
        name: str = "TimeSeriesSplitting",
        description: str = "Splits time series data",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the time series splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading splitting configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("engineered_data") or context.has("data")

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split time series data.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).

        Raises:
            ValueError: If time column is not specified or not found.
        """
        # Get feature columns (all columns except target columns)
        feature_columns = [col for col in data.columns if col not in target_columns]

        # Extract features and targets
        x = data[feature_columns]
        y = data[target_columns]

        # Get split parameters
        test_size = kwargs.get("test_size", self.config.get("test_size", 0.3))
        time_column = kwargs.get("time_column", self.config.get("time_column"))

        # Check if time column is specified
        if time_column is None:
            raise ValueError("Time column not specified")

        # Check if time column exists
        if time_column not in data.columns:
            raise ValueError(f"Time column '{time_column}' not found")

        # Sort data by time column
        sorted_indices = data[time_column].argsort()
        x_sorted = x.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]

        # Calculate split point
        split_point = int(len(data) * (1 - test_size))

        # Split the data
        x_train = x_sorted.iloc[:split_point]
        x_test = x_sorted.iloc[split_point:]
        y_train = y_sorted.iloc[:split_point]
        y_test = y_sorted.iloc[split_point:]

        return x_train, x_test, y_train, y_test


class CrossValidationSplittingStage(BaseDataSplittingStage):
    """
    Implementation of DataSplittingStage for cross-validation data splitting.
    """

    def __init__(
        self,
        name: str = "CrossValidationSplitting",
        description: str = "Splits data for cross-validation",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the cross-validation splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading splitting configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("engineered_data") or context.has("data")

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data for cross-validation.

        This method returns a single fold of the cross-validation splits.
        The fold index can be specified in kwargs.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).
        """
        # Get feature columns (all columns except target columns)
        feature_columns = [col for col in data.columns if col not in target_columns]

        # Extract features and targets
        x = data[feature_columns]
        y = data[target_columns]

        # Get split parameters
        n_splits = kwargs.get("n_splits", self.config.get("n_splits", 5))
        fold_index = kwargs.get("fold_index", self.config.get("fold_index", 0))
        random_state = kwargs.get(
            "random_state", self.config.get("random_state", 42)
        )
        cv_type = kwargs.get("cv_type", self.config.get("cv_type", "kfold"))
        stratify_column = kwargs.get(
            "stratify_column", self.config.get("stratify_column")
        )

        # Create cross-validator
        if cv_type == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = list(cv.split(x))
        elif cv_type == "stratified":
            # Get stratification values
            if stratify_column is None:
                # If no stratification column is specified, use the first target column
                if len(target_columns) > 0:
                    stratify = y[target_columns[0]]
                else:
                    raise ValueError("No stratification column specified")
            elif stratify_column in target_columns:
                # If stratification column is a target column, use it
                stratify = y[stratify_column]
            elif stratify_column in feature_columns:
                # If stratification column is a feature column, use it
                stratify = x[stratify_column]
            else:
                raise ValueError(f"Stratification column '{stratify_column}' not found")

            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            splits = list(cv.split(x, stratify))
        elif cv_type == "timeseries":
            cv = TimeSeriesSplit(n_splits=n_splits)
            splits = list(cv.split(x))
        else:
            raise ValueError(f"Unsupported cross-validation type: {cv_type}")

        # Check if fold index is valid
        if fold_index < 0 or fold_index >= len(splits):
            raise ValueError(
                f"Invalid fold index: {fold_index}. Must be between 0 and {len(splits) - 1}"
            )

        # Get the specified fold
        train_indices, test_indices = splits[fold_index]

        # Split the data
        x_train = x.iloc[train_indices]
        x_test = x.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]

        return x_train, x_test, y_train, y_test


class ConfigDrivenDataSplittingStage(BaseDataSplittingStage):
    """
    Implementation of DataSplittingStage that uses configuration for data splitting.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenDataSplitting",
        description: str = "Splits data based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the configuration-driven data splitting stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading splitting configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self._splitters = {
            "random": RandomSplittingStage(
                config=config, config_manager=config_manager
            ),
            "stratified": StratifiedSplittingStage(
                config=config, config_manager=config_manager
            ),
            "timeseries": TimeSeriesSplittingStage(
                config=config, config_manager=config_manager
            ),
            "cross_validation": CrossValidationSplittingStage(
                config=config, config_manager=config_manager
            ),
        }

    def validate_context(self, context: PipelineContext) -> bool:
        """
        Validate that the context contains all required data for this stage.

        Args:
            context: The pipeline context to validate.

        Returns:
            True if the context is valid, False otherwise.
        """
        return context.has("engineered_data") or context.has("data")

    def split_data(
        self, data: pd.DataFrame, target_columns: List[str], **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data based on configuration.

        Args:
            data: Input DataFrame to split.
            target_columns: List of target column names.
            **kwargs: Additional arguments for splitting.

        Returns:
            Tuple containing (x_train, x_test, y_train, y_test).
        """
        # Get the splitting type from kwargs or config
        splitting_type = kwargs.get(
            "splitting_type", self.config.get("splitting_type", "random")
        )

        # Get the appropriate splitter
        if splitting_type not in self._splitters:
            raise ValueError(f"Unsupported splitting type: {splitting_type}")

        splitter = self._splitters[splitting_type]

        # Split the data
        return splitter.split_data(data, target_columns, **kwargs)