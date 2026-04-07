"""Abstract base class for all models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """
    Consistent interface for all models.

    To add a new model: subclass BaseModel, implement build_model / fit / predict.
    Register it in modeling/__init__.py MODEL_REGISTRY.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_ = None
        self.is_fitted_ = False

    @abstractmethod
    def build_model(self, **kwargs):
        """Instantiate the underlying model. Returns self."""

    @abstractmethod
    def fit(self, X, y, **kwargs):
        """Train the model. Returns self."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return predictions."""

    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities (classification only)."""
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        raise NotImplementedError("Model does not support predict_proba")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Return feature importances if available."""
        if hasattr(self.model_, "feature_importances_"):
            return self.model_.feature_importances_
        return None

    def save(self, path: str):
        """Save model to disk."""
        import joblib
        joblib.dump(self.model_, path)

    def load(self, path: str):
        """Load model from disk."""
        import joblib
        self.model_ = joblib.load(path)
        self.is_fitted_ = True
        return self
