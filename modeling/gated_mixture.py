"""Confidence-gated two-tier mixture of classifiers.

When the high-confidence model is sure, use it; otherwise fall back to a
specialist that's better on boundary samples.

Motivated by the analysis of our pure-TabNet stacking ensemble:
- Top-1 probability >= 0.9 bucket: 55.9% of samples, 93.23% accuracy
- Top-1 probability in [0.5, 0.7):  12.2% of samples, 60-68% accuracy
The errors are concentrated in the low-confidence zone, so a specialist
there (e.g. an ordinal-aware model) can lift overall accuracy without
touching the high-confidence correct predictions.

This class expects both sub-models to already be fitted. It's meant to be
constructed *after* the two base ensembles are trained, not as an ensemble
component that trains its own bases.
"""

from typing import Any, Dict, Optional

import numpy as np

from configs.config import TaskType
from modeling.base_model import BaseModel


class GatedMixtureClassifier(BaseModel):
    """Two-tier mixture by predicted top-1 confidence threshold."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        high_conf_model: Optional[BaseModel] = None,
        low_conf_model: Optional[BaseModel] = None,
        threshold: float = 0.8,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ):
        super().__init__(config)
        if task_type != TaskType.CLASSIFICATION:
            raise ValueError("GatedMixtureClassifier is classification-only")
        self.task_type = task_type
        self.high_conf_model_ = high_conf_model
        self.low_conf_model_ = low_conf_model
        self.threshold = float(threshold)

    def build_model(self, num_classes: int = 5, **kwargs) -> "GatedMixtureClassifier":
        if self.high_conf_model_ is None or self.low_conf_model_ is None:
            raise ValueError(
                "GatedMixtureClassifier requires both high_conf_model and "
                "low_conf_model to be constructed beforehand."
            )
        self.model_ = (self.high_conf_model_, self.low_conf_model_)
        return self

    def fit(self, X, y, **kwargs):
        # Fit both sub-models on the same data. In practice you'd pass
        # pre-fitted models; this is here for interface symmetry.
        if not getattr(self.high_conf_model_, "is_fitted_", False):
            self.high_conf_model_.fit(X, y, **kwargs)
        if not getattr(self.low_conf_model_, "is_fitted_", False):
            self.low_conf_model_.fit(X, y, **kwargs)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba_high = np.asarray(self.high_conf_model_.predict_proba(X), dtype=np.float64)
        top1 = proba_high.max(axis=1)
        mask_low = top1 < self.threshold
        if mask_low.any():
            proba_low = np.asarray(
                self.low_conf_model_.predict_proba(np.asarray(X)[mask_low]),
                dtype=np.float64,
            )
            proba_high[mask_low] = proba_low
        return proba_high

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    @staticmethod
    def tune_threshold(
        oof_proba_high: np.ndarray,
        oof_proba_low: np.ndarray,
        y_true: np.ndarray,
        grid: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Pick the threshold that maximizes OOF accuracy of the mixture.

        Inputs are aligned OOF predictions from both base models. Works
        without refitting anything — just grid search on the gating rule.
        """
        if grid is None:
            grid = np.arange(0.50, 0.98, 0.02)
        top1_high = oof_proba_high.max(axis=1)
        best = {"threshold": float(grid[0]), "accuracy": -1.0}
        pred_high = oof_proba_high.argmax(axis=1)
        pred_low = oof_proba_low.argmax(axis=1)
        for t in grid:
            mask_low = top1_high < t
            mixed_pred = np.where(mask_low, pred_low, pred_high)
            acc = float((mixed_pred == y_true).mean())
            if acc > best["accuracy"]:
                best = {"threshold": float(t), "accuracy": acc}
        return best
