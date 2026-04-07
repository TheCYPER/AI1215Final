"""Evaluation metrics for classification and regression tasks."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Any, Dict


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Compute regression metrics."""
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def compute_combined_score(accuracy: float, r2: float) -> float:
    """Kaggle combined score: 0.5 * Accuracy + 0.5 * R2."""
    return 0.5 * accuracy + 0.5 * r2
