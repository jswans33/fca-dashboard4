"""
Model Evaluator Component

This module provides a standard implementation of the ModelEvaluator interface
that uses the unified configuration system from Work Chunk 1.
"""

import logging
from typing import Any, Dict, Optional

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

from nexusml.core.config.provider import ConfigurationProvider
from nexusml.core.pipeline.base import BaseModelEvaluator

# Set up logging
logger = logging.getLogger(__name__)


class EnhancedModelEvaluator(BaseModelEvaluator):
    """
    Enhanced implementation of the ModelEvaluator interface.

    This class evaluates models based on configuration provided by the
    ConfigurationProvider. It provides detailed metrics and analysis,
    with special focus on "Other" categories.
    """

    def __init__(
        self,
        name: str = "EnhancedModelEvaluator",
        description: str = "Enhanced model evaluator using unified configuration",
        config_provider: Optional[ConfigurationProvider] = None,
    ):
        """
        Initialize the EnhancedModelEvaluator.

        Args:
            name: Component name.
            description: Component description.
            config_provider: Configuration provider instance. If None, creates a new one.
        """
        # Initialize with empty config, we'll get it from the provider
        super().__init__(name, description, config={})
        self._config_provider = config_provider or ConfigurationProvider()

        # Create a default evaluation configuration
        self.config = {
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "detailed_report": True,
                "confusion_matrix": True,
                "other_category_analysis": True,
            }
        }

        # Try to update from configuration provider if available
        try:
            # Check if there's a classification section in the config
            if hasattr(self._config_provider.config, "classification"):
                classifier_config = (
                    self._config_provider.config.classification.model_dump()
                )
                if "evaluation" in classifier_config:
                    self.config.update(classifier_config["evaluation"])
                    logger.info(
                        "Updated evaluation configuration from classification section"
                    )
            logger.debug(f"Using evaluation configuration: {self.config}")
        except Exception as e:
            logger.warning(f"Could not load evaluation configuration: {e}")
            logger.info("Using default evaluation configuration")

        logger.info(f"Initialized {name}")

    def evaluate(
        self, model: Pipeline, x_test: pd.DataFrame, y_test: pd.DataFrame, **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.

        Args:
            model: Trained model pipeline to evaluate.
            x_test: Test features.
            y_test: Test targets.
            **kwargs: Additional arguments for evaluation.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            ValueError: If the model cannot be evaluated.
        """
        try:
            logger.info(f"Evaluating model on data with shape: {x_test.shape}")

            # Make predictions
            y_pred = model.predict(x_test)

            # Convert to DataFrame if it's not already
            if not isinstance(y_pred, pd.DataFrame):
                y_pred = pd.DataFrame(
                    y_pred, columns=y_test.columns, index=y_test.index
                )

            # Calculate metrics for each target column
            metrics = {}
            for col in y_test.columns:
                # Extract column values safely
                y_true_col = y_test.loc[:, col]
                y_pred_col = y_pred.loc[:, col]

                # Ensure they are pandas Series
                if not isinstance(y_true_col, pd.Series):
                    y_true_col = pd.Series(y_true_col)
                if not isinstance(y_pred_col, pd.Series):
                    y_pred_col = pd.Series(y_pred_col, index=y_true_col.index)

                col_metrics = self._calculate_metrics(y_true_col, y_pred_col)
                metrics[col] = col_metrics

                # Log summary metrics
                logger.info(f"Metrics for {col}:")
                logger.info(f"  Accuracy: {col_metrics['accuracy']:.4f}")
                logger.info(f"  F1 Score: {col_metrics['f1_macro']:.4f}")

            # Calculate overall metrics (average across all columns)
            metrics["overall"] = {
                "accuracy_mean": np.mean(
                    [metrics[col]["accuracy"] for col in y_test.columns]
                ),
                "f1_macro_mean": np.mean(
                    [metrics[col]["f1_macro"] for col in y_test.columns]
                ),
                "precision_macro_mean": np.mean(
                    [metrics[col]["precision_macro"] for col in y_test.columns]
                ),
                "recall_macro_mean": np.mean(
                    [metrics[col]["recall_macro"] for col in y_test.columns]
                ),
            }

            # Log overall metrics
            logger.info("Overall metrics:")
            logger.info(f"  Accuracy: {metrics['overall']['accuracy_mean']:.4f}")
            logger.info(f"  F1 Score: {metrics['overall']['f1_macro_mean']:.4f}")

            # Store predictions for further analysis
            metrics["predictions"] = y_pred

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise ValueError(f"Error evaluating model: {str(e)}") from e

    def analyze_predictions(
        self,
        model: Pipeline,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze model predictions in detail.

        Args:
            model: Trained model pipeline.
            x_test: Test features.
            y_test: Test targets.
            y_pred: Model predictions.
            **kwargs: Additional arguments for analysis.

        Returns:
            Dictionary of analysis results.

        Raises:
            ValueError: If predictions cannot be analyzed.
        """
        try:
            logger.info("Analyzing model predictions")

            analysis = {}

            # Analyze each target column
            for col in y_test.columns:
                col_analysis = self._analyze_column(col, x_test, y_test, y_pred)
                analysis[col] = col_analysis

                # Log summary of analysis
                logger.info(f"Analysis for {col}:")
                if "other_category" in col_analysis:
                    other = col_analysis["other_category"]
                    logger.info(f"  'Other' category metrics:")
                    logger.info(f"    Precision: {other['precision']:.4f}")
                    logger.info(f"    Recall: {other['recall']:.4f}")
                    logger.info(f"    F1 Score: {other['f1_score']:.4f}")

                # Log class distribution
                if "class_distribution" in col_analysis:
                    logger.info(f"  Class distribution:")
                    for cls, count in col_analysis["class_distribution"].items():
                        logger.info(f"    {cls}: {count}")

            # Analyze feature importance if the model supports it
            if hasattr(model, "named_steps") and "clf" in model.named_steps:
                clf = model.named_steps["clf"]
                if hasattr(clf, "estimators_"):
                    analysis["feature_importance"] = self._analyze_feature_importance(
                        model, x_test
                    )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing predictions: {str(e)}")
            raise ValueError(f"Error analyzing predictions: {str(e)}") from e

    def _calculate_metrics(
        self, y_true: pd.Series, y_pred: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate evaluation metrics for a single target column.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            Dictionary of metrics.
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

        # Calculate per-class metrics
        classes = sorted(set(y_true) | set(y_pred))
        per_class_metrics = {}

        for cls in classes:
            # True positives: predicted as cls and actually cls
            tp = ((y_true == cls) & (y_pred == cls)).sum()
            # False positives: predicted as cls but not actually cls
            fp = ((y_true != cls) & (y_pred == cls)).sum()
            # False negatives: not predicted as cls but actually cls
            fn = ((y_true == cls) & (y_pred != cls)).sum()

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            per_class_metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": (y_true == cls).sum(),
            }

        metrics["per_class"] = per_class_metrics

        return metrics

    def _analyze_column(
        self,
        column: str,
        x_test: pd.DataFrame,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Analyze predictions for a single target column.

        Args:
            column: Column name to analyze.
            x_test: Test features.
            y_test: Test targets.
            y_pred: Model predictions.

        Returns:
            Dictionary of analysis results.
        """
        analysis = {}

        # Class distribution
        y_true_dist = y_test[column].value_counts().to_dict()
        y_pred_dist = y_pred[column].value_counts().to_dict()

        analysis["class_distribution"] = {
            "true": y_true_dist,
            "predicted": y_pred_dist,
        }

        # Analyze "Other" category if present
        if "Other" in y_test[column].unique():
            other_indices = y_test[column] == "Other"

            # Calculate accuracy for "Other" category
            if other_indices.sum() > 0:
                other_accuracy = (
                    y_test[column][other_indices] == y_pred[column][other_indices]
                ).mean()

                # Calculate confusion metrics for "Other" category
                tp = ((y_test[column] == "Other") & (y_pred[column] == "Other")).sum()
                fp = ((y_test[column] != "Other") & (y_pred[column] == "Other")).sum()
                fn = ((y_test[column] == "Other") & (y_pred[column] != "Other")).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                analysis["other_category"] = {
                    "accuracy": float(other_accuracy),
                    "true_positives": int(tp),
                    "false_positives": int(fp),
                    "false_negatives": int(fn),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                }

                # Analyze misclassifications
                if fn > 0:
                    # False negatives: Actually "Other" but predicted as something else
                    fn_indices = (y_test[column] == "Other") & (
                        y_pred[column] != "Other"
                    )
                    fn_examples = []

                    for i in range(min(5, fn_indices.sum())):
                        idx = fn_indices[fn_indices].index[i]
                        fn_examples.append(
                            {
                                "index": idx,
                                "predicted_as": y_pred[column][idx],
                            }
                        )

                    analysis["other_category"]["false_negatives_examples"] = fn_examples

                if fp > 0:
                    # False positives: Predicted as "Other" but actually something else
                    fp_indices = (y_test[column] != "Other") & (
                        y_pred[column] == "Other"
                    )
                    fp_examples = []

                    for i in range(min(5, fp_indices.sum())):
                        idx = fp_indices[fp_indices].index[i]
                        fp_examples.append(
                            {
                                "index": idx,
                                "actual_class": y_test[column][idx],
                            }
                        )

                    analysis["other_category"]["false_positives_examples"] = fp_examples

        return analysis

    def _analyze_feature_importance(
        self, model: Pipeline, x_test: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze feature importance from the model.

        Args:
            model: Trained model pipeline.
            x_test: Test features.

        Returns:
            Dictionary of feature importance analysis.
        """
        feature_importance = {}

        try:
            # Extract the feature names
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                preprocessor = model.named_steps["preprocessor"]

                if hasattr(preprocessor, "transformers_"):
                    # Get feature names from transformers
                    feature_names = []

                    for name, transformer, column in preprocessor.transformers_:
                        if name == "text" and hasattr(transformer, "named_steps"):
                            if "tfidf" in transformer.named_steps:
                                tfidf = transformer.named_steps["tfidf"]
                                if hasattr(tfidf, "get_feature_names_out"):
                                    text_features = tfidf.get_feature_names_out()
                                    feature_names.extend(text_features)
                        elif name == "numeric":
                            if isinstance(column, list):
                                feature_names.extend(column)
                            else:
                                feature_names.append(column)

                    # Get feature importances from the model
                    if hasattr(model, "named_steps") and "clf" in model.named_steps:
                        clf = model.named_steps["clf"]

                        if hasattr(clf, "estimators_"):
                            # For each target column
                            for i, estimator in enumerate(clf.estimators_):
                                if hasattr(estimator, "feature_importances_"):
                                    importances = estimator.feature_importances_

                                    # Create a list of (feature, importance) tuples
                                    importance_tuples = []
                                    for j, importance in enumerate(importances):
                                        if j < len(feature_names):
                                            importance_tuples.append(
                                                (feature_names[j], importance)
                                            )
                                        else:
                                            importance_tuples.append(
                                                (f"feature_{j}", importance)
                                            )

                                    # Sort by importance (descending)
                                    importance_tuples.sort(
                                        key=lambda x: x[1], reverse=True
                                    )

                                    # Convert to dictionary
                                    target_importances = {}
                                    for feature, importance in importance_tuples[
                                        :20
                                    ]:  # Top 20 features
                                        target_importances[feature] = float(importance)

                                    feature_importance[f"target_{i}"] = (
                                        target_importances
                                    )
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {e}")

        return feature_importance
