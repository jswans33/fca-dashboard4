"""
Hyperparameter Optimizer Module

This module provides a HyperparameterOptimizer implementation that optimizes
hyperparameters for machine learning models.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.di.decorators import inject, injectable
from nexusml.core.model_building.base import BaseHyperparameterOptimizer

# Set up logging
logger = logging.getLogger(__name__)


@injectable
class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """
    Implementation of the HyperparameterOptimizer interface using GridSearchCV.
    
    This class optimizes hyperparameters for machine learning models using
    GridSearchCV based on configuration provided by the ConfigurationProvider.
    """
    
    def __init__(
        self,
        name: str = "GridSearchOptimizer",
        description: str = "Grid search hyperparameter optimizer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the GridSearchOptimizer.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        self._grid_search = None
        logger.info(f"Initialized {name}")
    
    def optimize(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model using grid search.
        
        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for optimization.
            
        Returns:
            Optimized model pipeline.
            
        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        try:
            logger.info(f"Optimizing hyperparameters for model using grid search on {len(x_train)} samples")
            
            # Extract grid search parameters from config and kwargs
            param_grid = kwargs.get("param_grid", {})
            cv = kwargs.get("cv", self.config.get("cv", 3))
            scoring = kwargs.get("scoring", self.config.get("scoring", "f1_macro"))
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))
            n_jobs = kwargs.get("n_jobs", self.config.get("n_jobs", None))
            
            if not param_grid:
                logger.warning("No parameter grid provided for hyperparameter optimization")
                return model
            
            # Log optimization information
            logger.info(f"X_train shape: {x_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_train columns: {x_train.columns.tolist()}")
            logger.info(f"y_train columns: {y_train.columns.tolist()}")
            logger.info(f"Parameter grid: {param_grid}")
            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            
            # Perform grid search
            if verbose:
                print(f"Performing grid search with {cv}-fold cross-validation...")
                print(f"X_train shape: {x_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"Parameter grid: {param_grid}")
            
            # Create and fit the grid search
            self._grid_search = GridSearchCV(
                model, param_grid=param_grid, cv=cv, scoring=scoring, 
                verbose=verbose, n_jobs=n_jobs, return_train_score=True
            )
            
            self._grid_search.fit(x_train, y_train)
            
            # Store the best parameters and score
            self._best_params = self._grid_search.best_params_
            self._best_score = self._grid_search.best_score_
            self._is_optimized = True
            
            # Print grid search results
            if verbose:
                print("\nGrid search results:")
                print(f"Best parameters: {self._best_params}")
                print(f"Best cross-validation score: {self._best_score:.4f}")
            
            # Log grid search results
            logger.info(f"Best parameters: {self._best_params}")
            logger.info(f"Best cross-validation score: {self._best_score:.4f}")
            
            return self._grid_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise ValueError(f"Error optimizing hyperparameters: {str(e)}") from e
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Get the cross-validation results from the grid search.
        
        Returns:
            Dictionary of cross-validation results.
            
        Raises:
            ValueError: If grid search has not been performed.
        """
        if self._grid_search is None:
            raise ValueError("Grid search has not been performed")
        
        # Convert numpy arrays to lists for serialization
        cv_results = {}
        for key, value in self._grid_search.cv_results_.items():
            if isinstance(value, np.ndarray):
                cv_results[key] = value.tolist()
            else:
                cv_results[key] = value
        
        return cv_results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_params
    
    def get_best_score(self) -> float:
        """
        Get the best score achieved during optimization.
        
        Returns:
            Best score.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_score


@injectable
class RandomizedSearchOptimizer(BaseHyperparameterOptimizer):
    """
    Implementation of the HyperparameterOptimizer interface using RandomizedSearchCV.
    
    This class optimizes hyperparameters for machine learning models using
    RandomizedSearchCV based on configuration provided by the ConfigurationProvider.
    """
    
    def __init__(
        self,
        name: str = "RandomizedSearchOptimizer",
        description: str = "Randomized search hyperparameter optimizer using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the RandomizedSearchOptimizer.
        
        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        super().__init__(name, description, config_provider)
        self._random_search = None
        logger.info(f"Initialized {name}")
    
    def optimize(
        self, model: Pipeline, x_train: pd.DataFrame, y_train: pd.DataFrame, **kwargs
    ) -> Pipeline:
        """
        Optimize hyperparameters for the model using randomized search.
        
        Args:
            model: Model pipeline to optimize.
            x_train: Training features.
            y_train: Training targets.
            **kwargs: Additional arguments for optimization.
            
        Returns:
            Optimized model pipeline.
            
        Raises:
            ValueError: If hyperparameters cannot be optimized.
        """
        try:
            logger.info(f"Optimizing hyperparameters for model using randomized search on {len(x_train)} samples")
            
            # Import RandomizedSearchCV here to avoid circular imports
            from sklearn.model_selection import RandomizedSearchCV
            
            # Extract randomized search parameters from config and kwargs
            param_distributions = kwargs.get("param_distributions", {})
            n_iter = kwargs.get("n_iter", self.config.get("n_iter", 10))
            cv = kwargs.get("cv", self.config.get("cv", 3))
            scoring = kwargs.get("scoring", self.config.get("scoring", "f1_macro"))
            verbose = kwargs.get("verbose", self.config.get("verbose", 1))
            n_jobs = kwargs.get("n_jobs", self.config.get("n_jobs", None))
            random_state = kwargs.get("random_state", self.config.get("random_state", 42))
            
            if not param_distributions:
                logger.warning("No parameter distributions provided for hyperparameter optimization")
                return model
            
            # Log optimization information
            logger.info(f"X_train shape: {x_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_train columns: {x_train.columns.tolist()}")
            logger.info(f"y_train columns: {y_train.columns.tolist()}")
            logger.info(f"Parameter distributions: {param_distributions}")
            logger.info(f"Number of iterations: {n_iter}")
            logger.info(f"Cross-validation folds: {cv}")
            logger.info(f"Scoring metric: {scoring}")
            
            # Perform randomized search
            if verbose:
                print(f"Performing randomized search with {n_iter} iterations and {cv}-fold cross-validation...")
                print(f"X_train shape: {x_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"Parameter distributions: {param_distributions}")
            
            # Create and fit the randomized search
            self._random_search = RandomizedSearchCV(
                model, param_distributions=param_distributions, n_iter=n_iter,
                cv=cv, scoring=scoring, verbose=verbose, n_jobs=n_jobs,
                random_state=random_state, return_train_score=True
            )
            
            self._random_search.fit(x_train, y_train)
            
            # Store the best parameters and score
            self._best_params = self._random_search.best_params_
            self._best_score = self._random_search.best_score_
            self._is_optimized = True
            
            # Print randomized search results
            if verbose:
                print("\nRandomized search results:")
                print(f"Best parameters: {self._best_params}")
                print(f"Best cross-validation score: {self._best_score:.4f}")
            
            # Log randomized search results
            logger.info(f"Best parameters: {self._best_params}")
            logger.info(f"Best cross-validation score: {self._best_score:.4f}")
            
            return self._random_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise ValueError(f"Error optimizing hyperparameters: {str(e)}") from e
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Get the cross-validation results from the randomized search.
        
        Returns:
            Dictionary of cross-validation results.
            
        Raises:
            ValueError: If randomized search has not been performed.
        """
        if self._random_search is None:
            raise ValueError("Randomized search has not been performed")
        
        # Convert numpy arrays to lists for serialization
        cv_results = {}
        for key, value in self._random_search.cv_results_.items():
            if isinstance(value, np.ndarray):
                cv_results[key] = value.tolist()
            else:
                cv_results[key] = value
        
        return cv_results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best parameters found during optimization.
        
        Returns:
            Dictionary of best parameters.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_params
    
    def get_best_score(self) -> float:
        """
        Get the best score achieved during optimization.
        
        Returns:
            Best score.
            
        Raises:
            ValueError: If optimization has not been performed.
        """
        if not self._is_optimized:
            raise ValueError("Hyperparameter optimization has not been performed")
        return self._best_score