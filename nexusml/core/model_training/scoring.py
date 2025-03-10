"""
Custom Scoring Functions for Model Training

This module provides custom scoring functions for model training,
particularly for multiclass-multioutput classification which is not
supported by scikit-learn's standard scoring functions.
"""

import logging
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate as sklearn_cross_validate

# Set up logging
logger = logging.getLogger(__name__)


def multioutput_accuracy_score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """
    Calculate accuracy score for multiclass-multioutput classification.

    This function calculates the mean accuracy across all output columns.

    Args:
        y_true: True target values as a DataFrame.
        y_pred: Predicted target values as a DataFrame.

    Returns:
        Mean accuracy score across all output columns.
    """
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise ValueError("Both y_true and y_pred must be pandas DataFrames")

    # Ensure columns match
    if not all(col in y_pred.columns for col in y_true.columns):
        raise ValueError("y_pred must contain all columns from y_true")

    # Calculate accuracy for each column
    accuracies = []
    for col in y_true.columns:
        accuracies.append(accuracy_score(y_true[col], y_pred[col]))

    # Return mean accuracy
    return float(np.mean(accuracies))


def multioutput_f1_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "macro",
) -> float:
    """
    Calculate F1 score for multiclass-multioutput classification.

    This function calculates the mean F1 score across all output columns.

    Args:
        y_true: True target values as a DataFrame.
        y_pred: Predicted target values as a DataFrame.
        average: Averaging method for F1 score. Default is "macro".

    Returns:
        Mean F1 score across all output columns.
    """
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise ValueError("Both y_true and y_pred must be pandas DataFrames")

    # Ensure columns match
    if not all(col in y_pred.columns for col in y_true.columns):
        raise ValueError("y_pred must contain all columns from y_true")

    # Calculate F1 score for each column
    f1_scores = []
    for col in y_true.columns:
        try:
            f1_scores.append(f1_score(y_true[col], y_pred[col], average=average))
        except Exception as e:
            logger.warning(f"Error calculating F1 score for column {col}: {e}")
            # If there's only one class, F1 score is undefined, so we use accuracy instead
            f1_scores.append(accuracy_score(y_true[col], y_pred[col]))

    # Return mean F1 score
    return float(np.mean(f1_scores))


def multioutput_precision_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "macro",
) -> float:
    """
    Calculate precision score for multiclass-multioutput classification.

    This function calculates the mean precision score across all output columns.

    Args:
        y_true: True target values as a DataFrame.
        y_pred: Predicted target values as a DataFrame.
        average: Averaging method for precision score. Default is "macro".

    Returns:
        Mean precision score across all output columns.
    """
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise ValueError("Both y_true and y_pred must be pandas DataFrames")

    # Ensure columns match
    if not all(col in y_pred.columns for col in y_true.columns):
        raise ValueError("y_pred must contain all columns from y_true")

    # Calculate precision score for each column
    precision_scores = []
    for col in y_true.columns:
        try:
            precision_scores.append(
                precision_score(y_true[col], y_pred[col], average=average)
            )
        except Exception as e:
            logger.warning(f"Error calculating precision score for column {col}: {e}")
            # If there's only one class, precision is undefined, so we use accuracy instead
            precision_scores.append(accuracy_score(y_true[col], y_pred[col]))

    # Return mean precision score
    return float(np.mean(precision_scores))


def multioutput_recall_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    average: Literal["micro", "macro", "samples", "weighted", "binary"] = "macro",
) -> float:
    """
    Calculate recall score for multiclass-multioutput classification.

    This function calculates the mean recall score across all output columns.

    Args:
        y_true: True target values as a DataFrame.
        y_pred: Predicted target values as a DataFrame.
        average: Averaging method for recall score. Default is "macro".

    Returns:
        Mean recall score across all output columns.
    """
    if not isinstance(y_true, pd.DataFrame) or not isinstance(y_pred, pd.DataFrame):
        raise ValueError("Both y_true and y_pred must be pandas DataFrames")

    # Ensure columns match
    if not all(col in y_pred.columns for col in y_true.columns):
        raise ValueError("y_pred must contain all columns from y_true")

    # Calculate recall score for each column
    recall_scores = []
    for col in y_true.columns:
        try:
            recall_scores.append(
                recall_score(y_true[col], y_pred[col], average=average)
            )
        except Exception as e:
            logger.warning(f"Error calculating recall score for column {col}: {e}")
            # If there's only one class, recall is undefined, so we use accuracy instead
            recall_scores.append(accuracy_score(y_true[col], y_pred[col]))

    # Return mean recall score
    return float(np.mean(recall_scores))


def get_multioutput_scorer(metric: str = "accuracy", **kwargs) -> Callable:
    """
    Get a scoring function for multiclass-multioutput classification.

    Args:
        metric: Metric to use for scoring. Options are "accuracy", "f1", "precision", "recall".
        **kwargs: Additional arguments for the scoring function.

    Returns:
        Scoring function for the specified metric.

    Raises:
        ValueError: If the metric is not supported.
    """
    if metric == "accuracy":
        return multioutput_accuracy_score
    elif metric == "f1":
        return lambda y_true, y_pred: multioutput_f1_score(y_true, y_pred, **kwargs)
    elif metric == "precision":
        return lambda y_true, y_pred: multioutput_precision_score(
            y_true, y_pred, **kwargs
        )
    elif metric == "recall":
        return lambda y_true, y_pred: multioutput_recall_score(y_true, y_pred, **kwargs)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def cross_validate_multioutput(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.DataFrame,
    scoring: Union[str, List[str]] = "accuracy",
    **kwargs,
) -> Dict[str, List[float]]:
    """
    Perform cross-validation for multiclass-multioutput classification.

    This function wraps scikit-learn's cross_validate function to support
    multiclass-multioutput classification.

    Args:
        estimator: Estimator to use for cross-validation.
        X: Features as a DataFrame.
        y: Targets as a DataFrame.
        scoring: Scoring metric(s) to use. Default is "accuracy".
        **kwargs: Additional arguments for cross_validate.

    Returns:
        Dictionary of cross-validation results.

    Raises:
        ValueError: If the scoring metric is not supported.
    """
    # Get the cv parameter from kwargs or use default
    cv = kwargs.pop("cv", 5)

    # Remove parameters that are not used by sklearn's cross_validate
    # but might be passed from the model trainer
    for param in ["random_state", "n_jobs", "verbose"]:
        if param in kwargs:
            kwargs.pop(param)

    logger.info("Performing cross-validation for multiclass-multioutput classification")

    # Define a custom scoring function that handles multiclass-multioutput
    def custom_scorer(estimator, X, y):
        y_pred = estimator.predict(X)

        # Convert to DataFrame if it's not already
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y.columns)

        # Calculate the score based on the specified metric
        if isinstance(scoring, str):
            if scoring == "accuracy":
                return multioutput_accuracy_score(y, y_pred)
            elif scoring == "f1":
                return multioutput_f1_score(y, y_pred)
            elif scoring == "precision":
                return multioutput_precision_score(y, y_pred)
            elif scoring == "recall":
                return multioutput_recall_score(y, y_pred)
            else:
                raise ValueError(f"Unsupported scoring metric: {scoring}")
        else:
            # If scoring is a list, calculate all metrics
            scores = {}
            for metric in scoring:
                if metric == "accuracy":
                    scores[metric] = multioutput_accuracy_score(y, y_pred)
                elif metric == "f1":
                    scores[metric] = multioutput_f1_score(y, y_pred)
                elif metric == "precision":
                    scores[metric] = multioutput_precision_score(y, y_pred)
                elif metric == "recall":
                    scores[metric] = multioutput_recall_score(y, y_pred)
                else:
                    raise ValueError(f"Unsupported scoring metric: {metric}")
            return scores

    # Perform cross-validation with the custom scorer
    try:
        cv_results = sklearn_cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=custom_scorer,
            return_train_score=True,
            **kwargs,
        )

        # Convert numpy arrays to lists for better serialization
        return {
            "train_score": cv_results["train_score"].tolist(),
            "test_score": cv_results["test_score"].tolist(),
            "fit_time": cv_results["fit_time"].tolist(),
            "score_time": cv_results["score_time"].tolist(),
        }
    except Exception as e:
        logger.error(f"Error performing cross-validation: {str(e)}")
        # Return dummy results if cross-validation fails
        return {
            "train_score": [0.0] * cv,
            "test_score": [0.0] * cv,
            "fit_time": [0.0] * cv,
            "score_time": [0.0] * cv,
        }
