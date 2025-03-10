"""
Model Evaluation Stage Module

This module provides implementations of the ModelEvaluationStage interface for
evaluating trained models and analyzing their performance.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from nexusml.config.manager import ConfigurationManager
from nexusml.src.pipeline.context import PipelineContext
from nexusml.src.pipeline.stages.base import BaseModelEvaluationStage


class ClassificationEvaluationStage(BaseModelEvaluationStage):
    """
    Implementation of ModelEvaluationStage for evaluating classification models.
    """

    def __init__(
        self,
        name: str = "ClassificationEvaluation",
        description: str = "Evaluates classification models",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the classification evaluation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading evaluation configuration.
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
        return (
            (context.has("trained_model") or context.has("model"))
            and context.has("x_test")
            and context.has("y_test")
        )

    def evaluate_model(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a classification model.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(x_test)

        # Convert to DataFrame if it's not already
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

        # Calculate metrics for each target column
        metrics = {}
        for col in y_test.columns:
            # Get the column values
            y_test_col = y_test[col]
            y_pred_col = y_pred[col]

            # Calculate metrics
            col_metrics = {
                "accuracy": accuracy_score(y_test_col, y_pred_col),
                "precision": precision_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "classification_report": classification_report(
                    y_test_col, y_pred_col, output_dict=True, zero_division=0
                ),
                "confusion_matrix": confusion_matrix(y_test_col, y_pred_col).tolist(),
            }
            metrics[col] = col_metrics

        # Add overall metrics
        metrics["overall"] = {
            "accuracy_mean": np.mean(
                [metrics[col]["accuracy"] for col in y_test.columns]
            ),
            "precision_mean": np.mean(
                [metrics[col]["precision"] for col in y_test.columns]
            ),
            "recall_mean": np.mean([metrics[col]["recall"] for col in y_test.columns]),
            "f1_score_mean": np.mean(
                [metrics[col]["f1_score"] for col in y_test.columns]
            ),
        }

        # Store predictions in the metrics
        metrics["predictions"] = y_pred.to_dict()

        return metrics


class DetailedClassificationEvaluationStage(BaseModelEvaluationStage):
    """
    Implementation of ModelEvaluationStage for detailed evaluation of classification models.
    """

    def __init__(
        self,
        name: str = "DetailedClassificationEvaluation",
        description: str = "Performs detailed evaluation of classification models",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the detailed classification evaluation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading evaluation configuration.
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
        return (
            (context.has("trained_model") or context.has("model"))
            and context.has("x_test")
            and context.has("y_test")
        )

    def evaluate_model(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Perform detailed evaluation of a classification model.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Make predictions
        y_pred = model.predict(x_test)

        # Convert to DataFrame if it's not already
        if not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

        # Calculate metrics for each target column
        metrics = {}
        for col in y_test.columns:
            # Get the column values
            y_test_col = y_test[col]
            y_pred_col = y_pred[col]

            # Get unique classes
            classes = sorted(list(set(y_test_col.unique()) | set(y_pred_col.unique())))

            # Calculate confusion metrics
            cm = confusion_matrix(y_test_col, y_pred_col, labels=classes)
            cm_dict = {
                "matrix": cm.tolist(),
                "classes": classes,
            }

            # Calculate per-class metrics
            class_metrics = {}
            for i, cls in enumerate(classes):
                # True positives: diagonal element for this class
                tp = cm[i, i]
                # False positives: sum of column minus true positives
                fp = np.sum(cm[:, i]) - tp
                # False negatives: sum of row minus true positives
                fn = np.sum(cm[i, :]) - tp
                # True negatives: sum of all elements minus tp, fp, and fn
                tn = np.sum(cm) - tp - fp - fn

                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                accuracy = (
                    (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                )

                class_metrics[cls] = {
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "true_negatives": int(tn),
                    "false_negatives": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "accuracy": float(accuracy),
                }

            # Calculate overall metrics
            col_metrics = {
                "accuracy": accuracy_score(y_test_col, y_pred_col),
                "precision": precision_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    y_test_col, y_pred_col, average="weighted", zero_division=0
                ),
                "classification_report": classification_report(
                    y_test_col, y_pred_col, output_dict=True, zero_division=0
                ),
                "confusion_matrix": cm_dict,
                "class_metrics": class_metrics,
            }
            metrics[col] = col_metrics

        # Add overall metrics
        metrics["overall"] = {
            "accuracy_mean": np.mean(
                [metrics[col]["accuracy"] for col in y_test.columns]
            ),
            "precision_mean": np.mean(
                [metrics[col]["precision"] for col in y_test.columns]
            ),
            "recall_mean": np.mean([metrics[col]["recall"] for col in y_test.columns]),
            "f1_score_mean": np.mean(
                [metrics[col]["f1_score"] for col in y_test.columns]
            ),
        }

        # Add error analysis
        metrics["error_analysis"] = self._analyze_errors(x_test, y_test, y_pred)

        # Store predictions in the metrics
        metrics["predictions"] = y_pred.to_dict()

        return metrics

    def _analyze_errors(
        self, x_test: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.

        Args:
            x_test: Test features.
            y_test: Test targets.
            y_pred: Model predictions.

        Returns:
            Dictionary of error analysis results.
        """
        error_analysis = {}

        # Find misclassified samples for each target column
        for col in y_test.columns:
            # Get indices of misclassified samples
            misclassified = y_test[col] != y_pred[col]
            misclassified_indices = misclassified[misclassified].index.tolist()

            # Get misclassified samples
            misclassified_samples = []
            for idx in misclassified_indices:
                sample = {
                    "index": int(idx),
                    "features": x_test.loc[idx].to_dict(),
                    "true_label": y_test.loc[idx, col],
                    "predicted_label": y_pred.loc[idx, col],
                }
                misclassified_samples.append(sample)

            # Limit the number of samples to avoid large results
            max_samples = 10
            if len(misclassified_samples) > max_samples:
                misclassified_samples = misclassified_samples[:max_samples]

            # Calculate error rate
            error_rate = misclassified.mean()

            error_analysis[col] = {
                "error_rate": float(error_rate),
                "misclassified_count": int(misclassified.sum()),
                "misclassified_samples": misclassified_samples,
            }

        return error_analysis


class ConfigDrivenModelEvaluationStage(BaseModelEvaluationStage):
    """
    Implementation of ModelEvaluationStage that uses configuration for model evaluation.
    """

    def __init__(
        self,
        name: str = "ConfigDrivenModelEvaluation",
        description: str = "Evaluates a model based on configuration",
        config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigurationManager] = None,
    ):
        """
        Initialize the configuration-driven model evaluation stage.

        Args:
            name: Stage name.
            description: Stage description.
            config: Configuration for the stage.
            config_manager: Configuration manager for loading evaluation configuration.
        """
        super().__init__(name, description, config)
        self.config_manager = config_manager or ConfigurationManager()
        self._evaluators = {
            "classification": ClassificationEvaluationStage(
                config=config, config_manager=config_manager
            ),
            "detailed_classification": DetailedClassificationEvaluationStage(
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
        return (
            (context.has("trained_model") or context.has("model"))
            and context.has("x_test")
            and context.has("y_test")
        )

    def evaluate_model(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a model based on configuration.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        # Get the evaluation type from kwargs or config
        evaluation_type = kwargs.get(
            "evaluation_type", self.config.get("evaluation_type", "classification")
        )

        # Get the appropriate evaluator
        if evaluation_type not in self._evaluators:
            raise ValueError(f"Unsupported evaluation type: {evaluation_type}")

        evaluator = self._evaluators[evaluation_type]

        # Evaluate the model
        return evaluator.evaluate_model(model, x_test, y_test, **kwargs)
